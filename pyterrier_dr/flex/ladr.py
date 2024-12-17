from typing import Optional, Union
import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta
from . import FlexIndex


class LadrPreemptive(pt.Transformer):
    def __init__(self, flex_index, graph, dense_scorer, num_results=1000, hops=1, drop_query_vec=False, budget=False):
        self.flex_index = flex_index
        self.graph = graph
        self.dense_scorer = dense_scorer
        self.num_results = num_results
        self.hops = hops
        self.drop_query_vec = drop_query_vec
        self.budget = budget

    def transform(self, inp):
        pta.validate.result_frame(inp, extra_columns=['query_vec'])
        docnos, config = self.flex_index.payload(return_dvecs=False)

        qcols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']
        if not self.drop_query_vec:
            qcols += ['query_vec']
        all_results = pta.DataFrameBuilder(qcols + ['docno', 'score', 'rank'])

        budget = self.budget
        if budget == True: # noqa: E712 truth value not okay here because budget can also be int
            budget = self.num_results
        elif budget == False: # noqa: E712 truth value not okay here because budget can also be int
            budget = None

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
            if budget is None:
                ext_docids = np.unique(np.concatenate(ext_docids))
            else:
                # apply budget if needed. We want to prioritize the documents that came from the documents in the
                # initial set (and then by the neighbors of those documents, etc), so partition by the index and take
                # budget results in that order.
                ext_docids, idxs = np.unique(np.concatenate(ext_docids), return_index=True)
                if ext_docids.shape[0] > budget:
                    ext_docids = ext_docids[np.argpartition(idxs, budget)[:budget]]
            query_vecs = df['query_vec'].iloc[0].reshape(1, -1)
            scores = self.dense_scorer.score(query_vecs, ext_docids)
            scores = scores.reshape(-1)
            if scores.shape[0] > self.num_results:
                idxs = np.argpartition(scores, -self.num_results)[-self.num_results:]
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

    def fuse_rank_cutoff(self, k):
        if k < self.num_results and not self.budget:
            return LadrPreemptive(self.flex_index, self.graph, self.dense_scorer,
                                  num_results=k, hops=self.hops, drop_query_vec=self.drop_query_vec, budget=self.budget)

def _pre_ladr(self,
    k: int = 16,
    *,
    hops: int = 1,
    num_results: int = 1000,
    dense_scorer: Optional[pt.Transformer] = None,
    drop_query_vec: bool = False,
    budget: Union[bool, int] = False,
) -> pt.Transformer:
    """Returns a proactive LADR (Lexicaly-Accelerated Dense Retrieval) transformer.

    Args:
        k (int): The number of neighbours in the corpus graph.
        hops (int): The number of hops to consider. Defaults to 1.
        num_results (int): The number of results to return per query.
        dense_scorer (:class:`~pyterrier.Transformer`, optional): The dense scorer to use. Defaults to :meth:`np_scorer`.
        drop_query_vec (bool): Whether to drop the query vector from the output.
        budget (bool or int): The maximum number of vectors to score. If ``False``, no maximum is applied. If ``True``, the budget is set to ``num_results``. If an integer, this value is used as the budget.

    Returns:
        :class:`~pyterrier.Transformer`: A proactive LADR transformer.

    .. cite.dblp:: conf/sigir/KulkarniMGF23
    """
    graph = self.corpus_graph(k) if isinstance(k, int) else k
    return LadrPreemptive(self, graph, num_results=num_results, hops=hops, dense_scorer=dense_scorer or self.scorer(), drop_query_vec=drop_query_vec, budget=budget)
FlexIndex.ladr = _pre_ladr # TODO: remove this alias later
FlexIndex.pre_ladr = _pre_ladr
FlexIndex.ladr_proactive = _pre_ladr

class LadrAdaptive(pt.Transformer):
    def __init__(self, flex_index, graph, dense_scorer, num_results=1000, depth=100, max_hops=None, drop_query_vec=False, budget=False):
        self.flex_index = flex_index
        self.graph = graph
        self.dense_scorer = dense_scorer
        self.num_results = num_results
        self.depth = depth
        self.max_hops = max_hops
        self.drop_query_vec = drop_query_vec
        self.budget = budget

    def fuse_rank_cutoff(self, k):
        if k < self.num_results and not self.budget:
            return LadrAdaptive(self.flex_index, self.graph, self.dense_scorer,
                                num_results=k, depth=self.depth, max_hops=self.max_hops, drop_query_vec=self.drop_query_vec, budget=self.budget)

    def transform(self, inp):
        pta.validate.result_frame(inp, extra_columns=['query_vec'])
        docnos, config = self.flex_index.payload(return_dvecs=False)

        qcols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']
        if not self.drop_query_vec:
            qcols += ['query_vec']
        all_results = pta.DataFrameBuilder(qcols + ['docno', 'score', 'rank'])

        budget = self.budget
        if budget == True: # noqa: E712 truth value not okay here because budget can also be int
            budget = self.num_results
        elif budget == False: # noqa: E712 truth value not okay here because budget can also be int
            budget = None

        it = iter(inp.groupby('qid'))
        if self.flex_index.verbose:
            it = pt.tqdm(it)
        for qid, df in it:
            qdata = {col: [df[col].iloc[0]] for col in qcols}
            query_vecs = df['query_vec'].iloc[0].reshape(1, -1)
            docids = np.unique(docnos.inv[df['docno'].values])
            if budget is not None and docids.shape[0] > budget:
                # apply budget if needed
                docids = docids[:budget]
            scores = self.dense_scorer.score(query_vecs, docids).reshape(-1)
            scores = scores.reshape(-1)
            rnd = 0
            while (self.max_hops is None or rnd < self.max_hops) and (budget is None or scores.shape[0] < budget):
                if scores.shape[0] > self.depth:
                    dids = docids[np.argpartition(scores, -self.depth)[-self.depth:]]
                else:
                    dids = docids
                neighbour_dids = np.unique(self.graph.edges_data[dids].reshape(-1))
                new_neighbour_dids = np.setdiff1d(neighbour_dids, docids, assume_unique=True)
                if new_neighbour_dids.shape[0] == 0:
                    break
                if budget is not None and new_neighbour_dids.shape[0] + scores.shape[0] > budget:
                    # apply budget if needed
                    new_neighbour_dids = new_neighbour_dids[:budget - scores.shape[0]]
                neighbour_scores = self.dense_scorer.score(query_vecs, new_neighbour_dids).reshape(-1)
                cat_dids = np.concatenate([docids, new_neighbour_dids])
                idxs = np.argsort(cat_dids)
                docids = cat_dids[idxs]
                scores = np.concatenate([scores, neighbour_scores])[idxs]
                rnd += 1
            if scores.shape[0] > self.num_results:
                idxs = np.argpartition(scores, -self.num_results)[-self.num_results:]
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

def _ada_ladr(self,
    k: int = 16,
    *,
    depth: int = 100,
    num_results: int = 1000,
    dense_scorer: Optional[pt.Transformer] = None,
    max_hops: Optional[int] = None,
    drop_query_vec: bool = False,
    budget: Union[bool, int] = False,
) -> pt.Transformer:
    """Returns an adaptive LADR (Lexicaly-Accelerated Dense Retrieval) transformer.

    Args:
        k (int): The number of neighbours in the corpus graph.
        depth (int): The depth of the ranked list to consider for convergence.
        num_results (int): The number of results to return per query.
        dense_scorer (:class:`~pyterrier.Transformer`, optional): The dense scorer to use. Defaults to :meth:`np_scorer`.
        max_hops (int, optional): The maximum number of hops to consider. Defaults to ``None`` (no limit).
        drop_query_vec (bool): Whether to drop the query vector from the output.
        budget (bool or int): The maximum number of vectors to score. If ``False``, no maximum is applied. If ``True``, the budget is set to ``num_results``. If an integer, this value is used as the budget.

    Returns:
        :class:`~pyterrier.Transformer`: An adaptive LADR transformer.

    .. cite.dblp:: conf/sigir/KulkarniMGF23
    """
    graph = self.corpus_graph(k) if isinstance(k, int) else k
    return LadrAdaptive(self, graph, num_results=num_results, dense_scorer=dense_scorer or self.scorer(), depth=depth, max_hops=max_hops, drop_query_vec=drop_query_vec, budget=budget)
FlexIndex.ada_ladr = _ada_ladr
FlexIndex.ladr_adaptive = _ada_ladr
