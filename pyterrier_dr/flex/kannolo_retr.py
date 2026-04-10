

import pyterrier as pt
import pandas as pd
from pyterrier_dr.flex.core import FlexIndex
from pyterrier_dr.util import assert_kannolo
import numpy as np

class KannoloRetriever(pt.Transformer):
    def __init__(self, kindex, docnos, *args, num_results: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.kindex = kindex
        self.num_results = num_results
        self.docnos = docnos

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pt.validate.query_frame(df, extra_columns=['query_vec'])
        if len(df) == 0:
            return pd.DataFrame(columns= df.columns.to_list() + ['docno', 'score', 'rank'])

        qvecs = np.vstack(df['query_vec'].values).flatten()
        all_scores, all_docids = self.kindex.search(qvecs, k=self.num_results)
        out = []
        for position, (score, docid) in enumerate(zip(all_scores, all_docids)):
            qid_offset = position // self.num_results
            out.append((df.iloc[qid_offset]['qid'], df.iloc[qid_offset]['query'], self.docnos[docid], score))

        return pt.model.add_ranks( pd.DataFrame(out, columns=['qid', 'query', 'docno', 'score']) )

def _kannolo_retr_hsnw(self, ef_construction: int = 32, num_results: int = 1000) -> pt.Transformer:
    """Returns a retriever that searchers over a `kannolo` <https://github.com/TusKANNy/kannolo/>_ 
    HSWN index.

    Args:
        num_results (int): the number of results to return per query

    .. note::
        This transformer requires the ``kannolo`` package to be installed. Installation instructions are available
        in the `kannolo repository <https://github.com/TusKANNy/kannolo>`__.
        By using this method, you agree to cite the kANNolo paper (ECIR 2025) in any kind of material you produce 
        where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, 
        article, poster, presentation, or documentation. For more on the kANNolo Citation License and the BibTex 
        see https://github.com/TusKANNy/kannolo?tab=readme-ov-file#citation-license

    .. cite.dblp conf/ecir/DelfinoEMNRV25

    """
    #TODO: expose ef_construction and ef_search and any other relevant parameters
    assert_kannolo()
    _citation_license()
    from kannolo import DensePlainHNSW
    
    assert len(self) < num_results, "Number of results must be less than the number of documents in the index"
    docnos, dvecs, meta = self.payload(return_docnos=True, return_dvecs=True)
    # hack because kannolo doesn't know about memory-mapped numpy arrays,
    # but they are compatible with np.ndarray -- this causes the index to be
    # loaded into memory
    dvecs = dvecs.view(np.ndarray).flatten() 
    kindex = DensePlainHNSW.build_from_array(dvecs, dim=meta['vec_size'], metric="dotproduct")
    return KannoloRetriever(kindex, docnos, num_results=num_results)

FlexIndex.kannolo_hnsw_retriever = _kannolo_retr_hsnw

def _citation_license():
    print(
        """
This uses the TusKANNy kannolo library from Pisa, Tuscany, Italy — ISTI-CNR · University of Pisa. 

By using this method, you agree to cite the kANNolo paper (ECIR 2025) in any kind of material you produce 
where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, 
article, poster, presentation, or documentation. For more on the kANNolo Citation License and the BibTex see
https://github.com/TusKANNy/kannolo?tab=readme-ov-file#citation-license
""")
