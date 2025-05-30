from typing import Optional
import math
import os
import pyterrier as pt
import numpy as np
import ir_datasets
import pyterrier_alpha as pta
import pyterrier_dr
from . import FlexIndex

logger = ir_datasets.log.easy()


class ScannRetriever(pt.Indexer):
    def __init__(self, flex_index, scann_index, num_results=1000, leaves_to_search=None, qbatch=64, drop_query_vec=False):
        self.flex_index = flex_index
        self.scann_index = scann_index
        self.leaves_to_search = leaves_to_search
        self.num_results = num_results
        self.qbatch = qbatch
        self.drop_query_vec = drop_query_vec

    def fuse_rank_cutoff(self, k):
        return None # disable fusion for ANN
        if k < self.num_results:
            return ScannRetriever(self.flex_index, self.scann_index, num_results=k, leaves_to_search=self.leaves_to_search, qbatch=self.qbatch, drop_query_vec=self.drop_query_vec)

    def transform(self, inp):
        pta.validate.query_frame(inp, extra_columns=['query_vec'])
        inp = inp.reset_index(drop=True)
        docnos, config = self.flex_index.payload(return_dvecs=False)
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = query_vecs.copy()

        result = pta.DataFrameBuilder(['docno', 'docid', 'score', 'rank'])
        num_q = query_vecs.shape[0]
        QBATCH = self.qbatch
        for qidx in range(0, num_q, QBATCH):
            dids, scores = self.scann_index.search_batched(query_vecs[qidx:qidx+QBATCH], leaves_to_search=self.leaves_to_search, final_num_neighbors=self.num_results)
            for s, d in zip(scores, dids):
                mask = d != -1
                d = d[mask]
                s = s[mask]
                result.extend({
                    'docno': docnos.fwd[d],
                    'docid': d,
                    'score': s,
                    'rank': np.arange(d.shape[0]),
                })

        if self.drop_query_vec:
            inp = inp.drop(columns='query_vec')
        return result.to_df(inp)


def _scann_retriever(self,
    *,
    n_leaves: Optional[int] = None,
    leaves_to_search: int = 1,
    num_results: int = 1000,
    train_sample: Optional[int] = None,
    drop_query_vec=False
):
    """Returns a retriever over a ScaNN (Scalable Nearest Neighbors) index.

    Args:
        n_leaves (int, optional): Number of leaves in the ScaNN index. Defaults to approximatley sqrt(doc_count).
        leaves_to_search (int, optional): Number of leaves to search. Defaults to 1. The higher the value, the more accurate the search.
        num_results (int, optional): Number of results to return. Defaults to 1000.
        train_sample (int, optional): Number of training samples. Defaults to ``n_leaves*39``.
        drop_query_vec (bool, optional): Whether to drop the query vector from the output.

    Returns:
        :class:`~pyterrier.Transformer`: A transformer that retrieves using ScaNN.

    .. note::
        This method requires the ``scann`` package. Install it via ``pip install scann``.

    .. cite.dblp:: conf/icml/GuoSLGSCK20
    """
    pyterrier_dr.util.assert_scann()
    import scann
    dvecs, meta, = self.payload(return_docnos=False)

    if n_leaves is None:
        # rule of thumb: sqrt(doc_count) (from <https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md>)
        n_leaves = math.ceil(math.sqrt(meta['doc_count']))
        # we'll shift it to the nearest power of 2
        n_leaves = int(1 << math.ceil(math.log2(n_leaves)))

    if train_sample is None:
        train_sample = n_leaves * 39
        train_sample = min(train_sample, meta['doc_count'])
    elif 0 <= train_sample <= 1:
        train_sample = math.ceil(meta['doc_count'] * train_sample)

    key = ('scann', n_leaves, train_sample)
    index_name = f'scann_leaves-{n_leaves}_train-{train_sample}.scann'
    if key not in self._cache:
        if not os.path.exists(self.index_path/index_name):
            with logger.duration(f'building scann index with {n_leaves} leaves'):
                searcher = scann.ScannBuilder(dvecs, 10, "dot_product") # neighbours=10; doesn't seem to affect the model (?)
                searcher = searcher.tree(num_leaves=n_leaves, num_leaves_to_search=leaves_to_search, training_sample_size=train_sample, quantize_centroids=True)
                searcher = searcher.score_brute_force()
                searcher = searcher.create_pybind()
            with logger.duration('saving scann index'):
                (self.index_path/index_name).mkdir()
                searcher.serialize(str(self.index_path/index_name))
            self._cache[key] = searcher
        else:
            with logger.duration('reading index'):
                self._cache[key] = scann.scann_ops_pybind.load_searcher(dvecs, str(self.index_path/index_name))
    return ScannRetriever(self, self._cache[key], num_results=num_results, leaves_to_search=leaves_to_search, drop_query_vec=drop_query_vec)
FlexIndex.scann_retriever = _scann_retriever
