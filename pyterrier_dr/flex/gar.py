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


def _gar(self,
    k: int = 16,
    *,
    batch_size: int = 128,
    num_results: int = 1000
) -> pt.Transformer:
    """Returns a retriever that uses a corpus graph to search over a FlexIndex.

    Args:
        k (int): Number of neighbours in the corpus graph. Defaults to 16.
        batch_size (int): Batch size for retrieval. Defaults to 128.
        num_results (int): Number of results per query to return. Defaults to 1000.

    Returns:
        :class:`~pyterrier.Transformer`: A retriever that uses a corpus graph to search over a FlexIndex.

    .. code-block:: bibtex
        :caption: GAR Citation
        :class: citation

        @inproceedings{DBLP:conf/cikm/MacAvaneyTM22,
          author       = {Sean MacAvaney and
                          Nicola Tonellotto and
                          Craig Macdonald},
          editor       = {Mohammad Al Hasan and
                          Li Xiong},
          title        = {Adaptive Re-Ranking with a Corpus Graph},
          booktitle    = {Proceedings of the 31st {ACM} International Conference on Information
                          {\\&} Knowledge Management, Atlanta, GA, USA, October 17-21, 2022},
          pages        = {1491--1500},
          publisher    = {{ACM}},
          year         = {2022},
          url          = {https://doi.org/10.1145/3511808.3557231},
          doi          = {10.1145/3511808.3557231},
          timestamp    = {Wed, 19 Oct 2022 17:09:02 +0200},
          biburl       = {https://dblp.org/rec/conf/cikm/MacAvaneyTM22.bib},
          bibsource    = {dblp computer science bibliography, https://dblp.org}
        }
    """
    return FlexGar(self, self.corpus_graph(k), SimFn.dot, batch_size=batch_size, num_results=num_results)
FlexIndex.gar = _gar
