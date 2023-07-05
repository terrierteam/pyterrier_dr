import pandas as pd
import itertools
import numpy as np
import pyterrier as pt
from .. import SimFn
from . import FlexIndex
import ir_datasets

logger = ir_datasets.log.easy()

class LadrPreemptive(pt.Transformer):
    def __init__(self, flex_index, graph, dense_scorer, hops=1):
        self.flex_index = flex_index
        self.graph = graph
        self.dense_scorer = dense_scorer
        self.hops = hops

    def transform(self, inp):
        assert 'query_vec' in inp.columns and 'qid' in inp.columns
        assert 'docno' in inp.columns
        docnos, config = self.flex_index.payload(return_dvecs=False)

        res = {'qid': [], 'docid': [], 'score': []}
        it = iter(inp.groupby('qid'))
        if self.flex_index.verbose:
            it = logger.pbar(it)
        for qid, df in it:
            docids = docnos.inv[df['docno'].values]
            lx_docids = docids
            ext_docids = [docids]
            for _ in range(self.hops):
                docids = self.graph.edges_data[docids].reshape(-1)
                ext_docids.append(docids)
            ext_docids = np.unique(np.concatenate(ext_docids))
            query_vecs = df['query_vec'].iloc[0].reshape(1, -1)
            scores = self.dense_scorer.score(query_vecs, ext_docids)
            scores = scores.reshape(-1)
            if scores.shape[0] > self.flex_index.num_results:
                idxs = np.argpartition(scores, -self.flex_index.num_results)[-self.flex_index.num_results:]
            else:
                idxs = np.arange(scores.shape[0])
            docids, scores = ext_docids[idxs], scores[idxs]
            res['qid'].extend(itertools.repeat(qid, len(docids)))
            res['docid'].append(docids)
            res['score'].append(scores)
        res['docid'] = np.concatenate(res['docid'])
        res['score'] = np.concatenate(res['score'])
        res['docno'] = docnos.fwd[res['docid']]
        res = pd.DataFrame(res)
        res = pt.model.add_ranks(res)
        return res

def _pre_ladr(self, k=16, hops=1, dense_scorer=None):
    graph = self.corpus_graph(k) if isinstance(k, int) else k
    return LadrPreemptive(self, graph, hops=hops, dense_scorer=dense_scorer or self.scorer())
FlexIndex.ladr = _pre_ladr # TODO: remove this alias later
FlexIndex.pre_ladr = _pre_ladr

class LadrAdaptive(pt.Transformer):
    def __init__(self, flex_index, graph, dense_scorer, depth=100, max_hops=None):
        self.flex_index = flex_index
        self.graph = graph
        self.dense_scorer = dense_scorer
        self.depth = depth
        self.max_hops = max_hops

    def transform(self, inp):
        assert 'query_vec' in inp.columns and 'qid' in inp.columns
        assert 'docno' in inp.columns
        docnos, config = self.flex_index.payload(return_dvecs=False)

        res = {'qid': [], 'docid': [], 'score': []}
        it = iter(inp.groupby('qid'))
        if self.flex_index.verbose:
            it = logger.pbar(it)
        for qid, df in it:
            query_vecs = df['query_vec'].iloc[0].reshape(1, -1)
            docids = np.unique(docnos.inv[df['docno'].values])
            scores = self.dense_scorer.score(query_vecs, docids).reshape(-1)
            scores = scores.reshape(-1)
            rnd = 0
            while self.max_hops is None or rnd < self.max_hops:
                if scores.shape[0] > self.depth:
                    dids = docids[np.argpartition(scores, -self.depth)[-self.depth:]]
                else:
                    dids = docids
                neighbour_dids = np.unique(self.graph.edges_data[dids].reshape(-1))
                new_neighbour_dids = np.setdiff1d(neighbour_dids, docids, assume_unique=True)
                if new_neighbour_dids.shape[0] == 0:
                    break
                neighbour_scores = self.dense_scorer.score(query_vecs, new_neighbour_dids).reshape(-1)
                cat_dids = np.concatenate([docids, new_neighbour_dids])
                idxs = np.argsort(cat_dids)
                docids = cat_dids[idxs]
                scores = np.concatenate([scores, neighbour_scores])[idxs]
                rnd += 1
            if scores.shape[0] > self.flex_index.num_results:
                idxs = np.argpartition(scores, -self.flex_index.num_results)[-self.flex_index.num_results:]
            else:
                idxs = np.arange(scores.shape[0])
            res['qid'].extend(itertools.repeat(qid, len(idxs)))
            res['docid'].append(docids[idxs])
            res['score'].append(scores[idxs])
        res['docid'] = np.concatenate(res['docid'])
        res['score'] = np.concatenate(res['score'])
        res['docno'] = docnos.fwd[res['docid']]
        res = pd.DataFrame(res)
        res = pt.model.add_ranks(res)
        return res

def _ada_ladr(self, k=16, dense_scorer=None, depth=100, max_hops=None):
    graph = self.corpus_graph(k) if isinstance(k, int) else k
    return LadrAdaptive(self, graph, dense_scorer=dense_scorer or self.scorer(), depth=depth, max_hops=max_hops)
FlexIndex.ada_ladr = _ada_ladr
