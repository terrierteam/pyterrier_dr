from typing import Optional, Sequence
import numpy as np
import torch
import pyterrier_alpha as pta
import pyterrier as pt
from .. import SimFn, infer_device
from . import FlexIndex
from .np_retr import NumpyScorer


class TorchScorer(NumpyScorer):
    def __init__(self,
        flex_index: FlexIndex,
        torch_vecs: torch.Tensor,
        *,
        num_results: Optional[int] = None
    ):
        self.flex_index = flex_index
        self.torch_vecs = torch_vecs
        self.num_results = num_results
        self.docnos, meta = flex_index.payload(return_dvecs=False)

    def score(self, query_vecs, docids):
        query_vecs = torch.from_numpy(query_vecs).to(self.torch_vecs)
        doc_vecs = self.torch_vecs[docids]
        if self.flex_index.sim_fn == SimFn.cos:
            query_vecs = query_vecs / torch.norm(2, query_vecs, dim=1, keepdims=True)
            doc_vecs = doc_vecs / torch.norm(2, doc_vecs, dim=1, keepdims=True)
            return (query_vecs @ doc_vecs.T).cpu().numpy()
        elif self.flex_index.sim_fn == SimFn.dot:
            return (query_vecs @ doc_vecs.T).cpu().numpy()
        else:
            raise ValueError(f'{self.flex_index.sim_fn} not supported')

    # transform inherited from NumpyScorer

class TorchRetriever(pt.Transformer):
    def __init__(
        self,
        flex_index: FlexIndex,
        torch_vecs: torch.Tensor,
        *,
        num_results: int = 1000,
        qbatch: int = 64,
        index_select: Optional[np.ndarray] = None,
        drop_query_vec: bool = False,
    ):
        self.flex_index = flex_index
        self.torch_vecs = torch_vecs
        self.num_results = num_results
        self.qbatch = qbatch
        self.drop_query_vec = drop_query_vec

        self.docnos, _ = flex_index.payload(return_dvecs=False)

        self.index_select = None
        if index_select is not None:
            self.index_select = torch.as_tensor(
                index_select,
                dtype=torch.long,
                device=torch_vecs.device
            )

    def fuse_rank_cutoff(self, k):
        if k < self.num_results:
            return TorchRetriever(
                self.flex_index,
                self.torch_vecs,
                num_results=k,
                qbatch=self.qbatch,
                index_select=self.index_select,
                drop_query_vec=self.drop_query_vec,
            )

    def transform(self, inp):
        pta.validate.query_frame(inp, extra_columns=['query_vec'])
        inp = inp.reset_index(drop=True)

        query_vecs = torch.from_numpy(
            np.stack(inp['query_vec'])
        ).to(self.torch_vecs)

        tv = (
            self.torch_vecs[self.index_select].T
            if self.index_select is not None
            else self.torch_vecs.T
        )

        result = pta.DataFrameBuilder(['docno', 'docid', 'score', 'rank'])

        it = range(0, query_vecs.shape[0], self.qbatch)
        if self.flex_index.verbose:
            it = pt.tqdm(it, desc='TorchRetriever', unit='qbatch')

        for start in it:
            batch = query_vecs[start:start+self.qbatch]

            if self.flex_index.sim_fn != SimFn.dot:
                raise ValueError(f'{self.flex_index.sim_fn} not supported')

            scores = batch @ tv

            if scores.shape[1] > self.num_results:
                scores, docids = scores.topk(self.num_results, dim=1)
            else:
                docids = scores.argsort(descending=True, dim=1)
                scores = torch.gather(scores, 1, docids)

            scores = scores.cpu().numpy()
            docids = docids.cpu().numpy()

            if self.index_select is not None:
                docids = self.index_select.cpu().numpy()[docids]

            for s, d in zip(scores, docids):
                result.extend({
                    'docno': self.docnos[d],
                    'docid': d,
                    'score': s,
                    'rank': np.arange(len(s)),
                })

        if self.drop_query_vec:
            inp = inp.drop(columns='query_vec')

        return result.to_df(inp)



def _torch_vecs(
    self,
    *,
    device: Optional[str] = None,
    fp16: bool = False
) -> torch.Tensor:
    """Return the indexed vectors as a pytorch tensor.

    .. caution::
        This method loads the entire index into memory on the provided device.
        If the index is too large to fit in memory, consider using :meth:`np_vecs`
        or :meth:`get_corpus_iter`.

    Args:
        device: The device to use for the tensor. If not provided, the default
            device is used (cuda if available, otherwise cpu).
        fp16: Whether to use half precision (fp16).

    Returns:
        :class:`torch.Tensor`: The indexed vectors.
    """
    device = infer_device(device)
    key = ('torch_vecs', device, fp16)

    if key not in self._cache:
        # Load numpy-backed vectors (memory-mapped)
        dvecs, _ = self.payload(return_docnos=False)

        # Important: frombuffer avoids an extra copy
        data = torch.frombuffer(
            dvecs,
            dtype=torch.float32
        ).reshape(*dvecs.shape)

        if fp16:
            data = data.half()

        self._cache[key] = data.to(device)

    return self._cache[key]
FlexIndex.torch_vecs = _torch_vecs


def _torch_scorer(self, *, num_results: Optional[int] = None, device: Optional[str] = None, fp16: bool = False):
    """Return a scorer that uses pytorch to score (re-rank) results using indexed vectors.

    The returned :class:`pyterrier.Transformer` expects a DataFrame with columns ``qid``, ``query_vec`` and ``docno``.
    (If an internal ``docid`` column is provided, this will be used to speed up vector lookups.)

    .. caution::
        This method loads the entire index into memory on the provided device. If the index is too large to fit in memory,
        consider using a different scorer that does not fully load the index into memory, like :meth:`np_scorer`.

    Args:
        num_results: The number of results to return per query. If not provided, all resuls from the original fram are returned.
        device: The device to use for scoring. If not provided, the default device is used (cuda if available, otherwise cpu).
        fp16: Whether to use half precision (fp16) for scoring.

    Returns:
        :class:`~pyterrier.Transformer`: A transformer that scores query vectors with pytorch.
    """
    return TorchScorer(self, self.torch_vecs(device=device, fp16=fp16), num_results=num_results)
FlexIndex.torch_scorer = _torch_scorer


def _torch_retriever(self,
    *,
    num_results: int = 1000,
    device: Optional[str] = None,
    fp16: bool = False,
    qbatch: int = 64,
    drop_query_vec: bool = False
    index_select: Optional[np.ndarray] = None,
):
    """Return a retriever that uses pytorch to perform brute-force retrieval results using the indexed vectors.

    The returned :class:`pyterrier.Transformer` expects a DataFrame with columns ``qid``, ``query_vec``.

    .. caution::
        This method loads the entire index into memory on the provided device. If the index is too large to fit in memory,
        consider using a different retriever that does not fully load the index into memory, like :meth:`np_retriever`.

    Args:
        num_results: The number of results to return per query.
        device: The device to use for scoring. If not provided, the default device is used (cuda if available, otherwise cpu).
        fp16: Whether to use half precision (fp16) for scoring.
        qbatch: The number of queries to score in each batch.
        drop_query_vec: Whether to drop the query vector from the output.
        index_select: Optional list or array of document ids to restrict retrieval to.
            If provided, retrieval is performed only over this subset of the index,
            which is internally converted to a torch tensor on the target device.

    Returns:
        :class:`~pyterrier.Transformer`: A transformer that retrieves using pytorch.
    """
    return TorchRetriever(self, self.torch_vecs(device=device, fp16=fp16), num_results=num_results, qbatch=qbatch, drop_query_vec=drop_query_vec, index_select=index_select)

FlexIndex.torch_retriever = _torch_retriever
