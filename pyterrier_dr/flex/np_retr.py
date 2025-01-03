from typing import Optional
import pyterrier as pt
import numpy as np
import pandas as pd
from .. import SimFn
from ..indexes import RankedLists
from . import FlexIndex
import pyterrier_alpha as pta


class NumpyRetriever(pt.Transformer):
    def __init__(self,
        flex_index: FlexIndex,
        *,
        num_results: int = 1000,
        batch_size: Optional[int] = None,
        drop_query_vec: bool = False
    ):
        self.flex_index = flex_index
        self.num_results = num_results
        self.batch_size = batch_size or 4096
        self.drop_query_vec = drop_query_vec

    def fuse_rank_cutoff(self, k):
        if k < self.num_results:
            return NumpyRetriever(self.flex_index, num_results=k, batch_size=self.batch_size, drop_query_vec=self.drop_query_vec)

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.query_frame(inp, extra_columns=['query_vec'])
        inp = inp.reset_index(drop=True)
        query_vecs = np.stack(inp['query_vec'])
        docnos, dvecs, config = self.flex_index.payload()
        if self.flex_index.sim_fn == SimFn.cos:
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        elif self.flex_index.sim_fn == SimFn.dot:
            pass # nothing to do
        else:
            raise ValueError(f'{self.flex_index.sim_fn} not supported')
        num_q = query_vecs.shape[0]
        ranked_lists = RankedLists(self.num_results, num_q)
        batch_it = range(0, dvecs.shape[0], self.batch_size)
        if self.flex_index.verbose:
            batch_it = pt.tqdm(batch_it, desc='NumpyRetriever scoring', unit='docbatch')
        for idx_start in batch_it:
            doc_batch = dvecs[idx_start:idx_start+self.batch_size].T
            if self.flex_index.sim_fn == SimFn.cos:
                doc_batch = doc_batch / np.linalg.norm(doc_batch, axis=0, keepdims=True)
            scores = query_vecs @ doc_batch
            dids = np.arange(idx_start, idx_start+doc_batch.shape[1], dtype='i4').reshape(1, -1).repeat(num_q, axis=0)
            ranked_lists.update(scores, dids)

        result = pta.DataFrameBuilder(['docno', 'docid', 'score', 'rank'])
        for scores, dids in zip(*ranked_lists.results()):
            result.extend({
                'docno': docnos.fwd[dids],
                'docid': dids,
                'score': scores,
                'rank': np.arange(len(scores)),
            })

        if self.drop_query_vec:
            inp = inp.drop(columns='query_vec')
        return result.to_df(inp)


class NumpyVectorLoader(pt.Transformer):
    def __init__(self, flex_index: FlexIndex):
        self.flex_index = flex_index

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        docids = self.flex_index._load_docids(inp)
        dvecs, config = self.flex_index.payload(return_docnos=False)
        return inp.assign(doc_vec=list(dvecs[docids]))
    
    def fuse_rank_cutoff(self, k):
        return pt.RankCutoff(k) >> self

class NumpyScorer(pt.Transformer):
    def __init__(self, flex_index: FlexIndex, *, num_results: Optional[int] = None):
        self.flex_index = flex_index
        self.num_results = num_results

    def score(self, query_vecs, docids):
        dvecs, config = self.flex_index.payload(return_docnos=False)
        doc_vecs = dvecs[docids]
        if self.flex_index.sim_fn == SimFn.cos:
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
            doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            return query_vecs @ doc_vecs.T
        elif self.flex_index.sim_fn == SimFn.dot:
            return query_vecs @ doc_vecs.T
        else:
            raise ValueError(f'{self.flex_index.sim_fn} not supported')

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        with pta.validate.any(inp) as v:
            v.columns(includes=['query_vec', 'docno'])
            v.columns(includes=['query_vec', 'docid'])
        inp = inp.reset_index(drop=True)

        res_idxs = []
        res_scores = []
        res_ranks = []
        for qid, df in pt.tqdm(inp.groupby('qid')):
            docids = self.flex_index._load_docids(df)
            query_vecs = df['query_vec'].iloc[0].reshape(1, -1)
            scores = self.score(query_vecs, docids).reshape(-1)
            index = df.index
            if self.num_results is not None and scores.shape[0] > self.num_results:
                idxs = np.argpartition(scores, -self.flex_index.num_results)[-self.num_results:]
                scores = scores[idxs]
                index = index[idxs]
            idxs = (-scores).argsort()
            res_scores.append(scores[idxs])
            res_idxs.append(index[idxs])
            res_ranks.append(np.arange(idxs.shape[0]))
        res_idxs = np.concatenate(res_idxs)
        res_scores = np.concatenate(res_scores)
        res_ranks = np.concatenate(res_ranks)
        res = pd.DataFrame({k: inp[k][res_idxs] for k in inp.columns if k not in ['score', 'rank']})
        res = res.assign(score=res_scores, rank=res_ranks)
        return res

def _np_vecs(self) -> np.ndarray:
    """Return the indexed vectors.

    Returns:
        :class:`numpy.ndarray`: The indexed vectors as a memory-mapped numpy array.
    """
    dvecs, meta = self.payload(return_docnos=False)
    return dvecs
FlexIndex.np_vecs = _np_vecs

def _np_retriever(self, *, num_results: int = 1000, batch_size: Optional[int] = None, drop_query_vec: bool = False):
    """Return a retriever that uses numpy to perform a brute force search over the index.

    The returned transformer expects a DataFrame with columns ``qid`` and ``query_vec``. It outpus
    a result frame containing the retrieved results.

    Args:
        num_results: The number of results to return per query.
        batch_size: The number of documents to score in each batch.
        drop_query_vec: Whether to drop the query vector from the output.

    Returns:
        :class:`~pyterrier.Transformer`: A retriever that uses numpy to perform a brute force search.
    """
    return NumpyRetriever(self, num_results=num_results, batch_size=batch_size, drop_query_vec=drop_query_vec)
FlexIndex.np_retriever = _np_retriever
FlexIndex.retriever = _np_retriever # default retriever


def _np_vec_loader(self):
    """Return a transformer that loads indexed vectors.

    The returned transformer expects a DataFrame with columns ``docno``. It outputs a frame that
    includes a column ``doc_vec``, which contains the indexed vectors. For example:

    .. code-block:: python
        :caption: Load vectors from a ``FlexIndex``

        index = FexIndex.from_hf('macavaney/msmarco-passage.tasb.flex')
        loader = index.vec_loader()
        loader(pd.DataFrame([
            {"docno": "5"},
            {"docno": "100"},
            {"docno": "74356"},
        ]))
        # docno                                            doc_vec
        #     5  [-0.09343405, 0.12045559, -0.25184962, 0.15029...
        #   100  [-0.11527929, 0.63400555, -0.0877756, -0.26490...
        # 74356  [0.15367049, 0.16049547, -0.012261144, -0.2588...

    Returns:
        :class:`~pyterrier.Transformer`: A transformer that loads indexed vectors.
    """
    return NumpyVectorLoader(self)
FlexIndex.np_vec_loader = _np_vec_loader
FlexIndex.vec_loader = _np_vec_loader # default vec_loader


def _np_scorer(self, *, num_results: Optional[int] = None) -> pt.Transformer:
    """Return a scorer that uses numpy to score (re-rank) results using indexed vectors.

    The returned transformer expects a DataFrame with columns ``qid``, ``query_vec`` and ``docno``.
    (If an internal ``docid`` column is provided, this will be used to speed up vector lookups.)

    This method uses memory-mapping to avoid loading the entire index into memory at once.

    Args:
        num_results: The number of results to return per query. If not provided, all resuls from the original fram are returned.

    Returns:
        :class:`~pyterrier.Transformer`: A transformer that scores query vectors with numpy.
    """
    return NumpyScorer(self, num_results=num_results)
FlexIndex.np_scorer = _np_scorer
FlexIndex.scorer = _np_scorer # default scorer
