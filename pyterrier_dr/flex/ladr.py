import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta
from . import FlexIndex


class LadrPreemptive(pt.Transformer):
    def __init__(self, flex_index, graph, dense_scorer, hops=1, drop_query_vec=False):
        self.flex_index = flex_index
        self.graph = graph
        self.dense_scorer = dense_scorer
        self.hops = hops
        self.drop_query_vec = drop_query_vec

    def transform(self, inp):
        pta.validate.result_frame(inp, extra_columns=['query_vec'])
        docnos, config = self.flex_index.payload(return_dvecs=False)

        qcols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']
        if not self.drop_query_vec:
            qcols += ['query_vec']
        all_results = pta.DataFrameBuilder(qcols + ['docno', 'score', 'rank'])

        it = iter(inp.groupby('qid'))
        if self.flex_index.verbose:
            it = pt.tqdm(it)
        for qid, df in it:
            qdata = {col: [df[col].iloc[0]] for col in qcols}
            docids = docnos.inv[df['docno'].values]
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
            idxs = np.argsort(-scores)
            docids, scores = docids[idxs], scores[idxs]
            all_results.extend(dict(
                **qdata,
                docno=docnos.fwd[docids],
                score=scores,
                rank=np.arange(len(scores)),
            ))
        return all_results.to_df()


def _pre_ladr(self, k=16, hops=1, dense_scorer=None, drop_query_vec=False):
    graph = self.corpus_graph(k) if isinstance(k, int) else k
    return LadrPreemptive(self, graph, hops=hops, dense_scorer=dense_scorer or self.scorer(), drop_query_vec=drop_query_vec)
FlexIndex.ladr = _pre_ladr # TODO: remove this alias later
FlexIndex.pre_ladr = _pre_ladr

class LadrAdaptive(pt.Transformer):
    def __init__(self, flex_index, graph, dense_scorer, depth=100, max_hops=None, drop_query_vec=False):
        self.flex_index = flex_index
        self.graph = graph
        self.dense_scorer = dense_scorer
        self.depth = depth
        self.max_hops = max_hops
        self.drop_query_vec = drop_query_vec

    def transform(self, inp):
        pta.validate.result_frame(inp, extra_columns=['query_vec'])
        docnos, config = self.flex_index.payload(return_dvecs=False)

        qcols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']
        if not self.drop_query_vec:
            qcols += ['query_vec']
        all_results = pta.DataFrameBuilder(qcols + ['docno', 'score', 'rank'])

        it = iter(inp.groupby('qid'))
        if self.flex_index.verbose:
            it = pt.tqdm(it)
        for qid, df in it:
            qdata = {col: [df[col].iloc[0]] for col in qcols}
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
            docids, scores = docids[idxs], scores[idxs]
            idxs = np.argsort(-scores)
            docids, scores = docids[idxs], scores[idxs]
            all_results.extend(dict(
                **qdata,
                docno=docnos.fwd[docids],
                score=scores,
                rank=np.arange(len(scores)),
            ))
        return all_results.to_df()

def _ada_ladr(self, k=16, dense_scorer=None, depth=100, max_hops=None, drop_query_vec=False):
    graph = self.corpus_graph(k) if isinstance(k, int) else k
    return LadrAdaptive(self, graph, dense_scorer=dense_scorer or self.scorer(), depth=depth, max_hops=max_hops, drop_query_vec=drop_query_vec)
FlexIndex.ada_ladr = _ada_ladr
