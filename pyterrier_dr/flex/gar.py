import pyterrier as pt
import heapq
import pyterrier_alpha as pta
from . import FlexIndex
import numpy as np
from pyterrier_dr import SimFn

class FlexGar(pt.Transformer):
    def __init__(self, flex_index, graph, score_fn, batch_size=128, num_results=1000, drop_query_vec=False):
        self.flex_index = flex_index
        self.docnos, self.dvecs, _ = flex_index.payload()
        self.score_fn = score_fn
        self.graph = graph
        self.batch_size = batch_size
        self.num_results = num_results
        self.drop_query_vec = drop_query_vec

    def transform(self, inp):
        pta.validate.result_frame(inp, extra_columns=['query_vec', 'score'])

        qcols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']
        if not self.drop_query_vec:
            qcols += ['query_vec']
        all_results = pta.DataFrameBuilder(qcols + ['docno', 'score', 'rank'])

        for qid, inp in inp.groupby('qid'):
            qvec = inp['query_vec'].iloc[0].reshape(1, -1)
            qdata = {col: [inp[col].iloc[0]] for col in qcols}
            initial_heap = list(zip(-inp['score'], self.docnos.inv[inp['docno']]))
            heapq.heapify(initial_heap)
            results = {}
            frontier_heap = []
            frontier_leftover = []
            i = 0
            while len(results) < self.num_results and (initial_heap or frontier_heap or frontier_leftover):
                size = min(self.batch_size, self.num_results - len(results))
                batch = []
                if i % 2 == 0 and initial_heap:
                    while len(batch) < size and initial_heap:
                        did = heapq.heappop(initial_heap)[1]
                        if did not in results:
                            batch.append(did)
                elif i % 2 == 1 and (frontier_heap or frontier_leftover):
                    while len(batch) < size and (frontier_heap or frontier_leftover):
                        if frontier_leftover:
                            frontier_leftover, e = frontier_leftover[size-len(batch):], frontier_leftover[:size-len(batch)]
                            batch.extend(e)
                        else:
                            _, did = heapq.heappop(frontier_heap)
                            frontier_leftover.extend(d for d in self.graph.neighbours(did) if d not in results)
                dvecs = self.dvecs[np.array(batch).astype(int)]
                if self.score_fn == SimFn.dot:
                    scores = (qvec @ dvecs.T).reshape(-1)
                for did, score in zip(batch, scores):
                    results[did] = score
                    heapq.heappush(frontier_heap, (-score, did))
                i += 1
            d, s = zip(*sorted(results.items(), key=lambda x: (-x[1], x[0])))
            all_results.extend(dict(
                **qdata,
                docno=self.docnos.fwd[d],
                score=s,
                rank=np.arange(len(s)),
            ))
        return all_results.to_df()



def _gar(self, k=16, batch_size=128, num_results=1000):
    return FlexGar(self, self.corpus_graph(k), SimFn.dot, batch_size=batch_size, num_results=num_results)
FlexIndex.gar = _gar
