import pandas as pd
import math
import os
import pyterrier as pt
import itertools
import numpy as np
import ir_datasets
from . import FlexIndex

logger = ir_datasets.log.easy()


class ScannRetriever(pt.Indexer):
    def __init__(self, flex_index, scann_index, leaves_to_search=None, qbatch=64):
        self.flex_index = flex_index
        self.scann_index = scann_index
        self.leaves_to_search = leaves_to_search
        self.qbatch = qbatch

    def transform(self, inp):
        inp = inp.reset_index(drop=True)
        assert all(f in inp.columns for f in ['qid', 'query_vec'])
        docnos, config = self.flex_index.payload(return_dvecs=False)
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = query_vecs.copy()
        idxs = []
        res = {'docid': [], 'score': [], 'rank': []}
        num_q = query_vecs.shape[0]
        QBATCH = self.qbatch
        for qidx in range(0, num_q, QBATCH):
            dids, scores = self.scann_index.search_batched(query_vecs[qidx:qidx+QBATCH], leaves_to_search=self.leaves_to_search, final_num_neighbors=self.flex_index.num_results)
            for i, (s, d) in enumerate(zip(scores, dids)):
                mask = d != -1
                d = d[mask]
                s = s[mask]
                res['docid'].append(d)
                res['score'].append(s)
                res['rank'].append(np.arange(d.shape[0]))
                idxs.extend(itertools.repeat(qidx+i, d.shape[0]))
        res = {k: np.concatenate(v) for k, v in res.items()}
        res['docno'] = docnos.fwd[res['docid']]
        for col in inp.columns:
            if col != 'query_vec':
                res[col] = inp[col][idxs].values
        return pd.DataFrame(res)


def _scann_retriever(self, n_leaves=None, leaves_to_search=1, train_sample=None):
        import scann
        assert not hasattr(scann.scann_ops_pybind, 'builder'), "scann==1.0.0 required; install from wheel here: <https://github.com/google-research/google-research/blob/master/scann/docs/releases.md#scann-wheel-archive>"
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
        return ScannRetriever(self, self._cache[key], leaves_to_search=leaves_to_search)
FlexIndex.scann_retriever = _scann_retriever
