

import pyterrier as pt
import pandas as pd
from pyterrier_dr.flex.core import FlexIndex
from pyterrier_dr.util import assert_kannolo
import numpy as np

class KannoloRetriever(pt.Transformer):
    def __init__(self, kindex, docnos, *args, num_results: int = 1000, early_exit_threshold=None, ef_search: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        self.kindex = kindex
        self.num_results = num_results
        self.docnos = docnos
        self.early_exit_threshold = early_exit_threshold
        self.ef_search = ef_search

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pt.validate.query_frame(df, extra_columns=['query_vec'])
        if len(df) == 0:
            return pd.DataFrame(columns= df.columns.to_list() + ['docno', 'score', 'rank'])

        qvecs = np.vstack(df['query_vec'].values).flatten()
        all_scores, all_docids = self.kindex.search(qvecs, k=self.num_results, ef_search=self.ef_search, early_exit_threshold=self.early_exit_threshold)
        out = []
        for position, (score, docid) in enumerate(zip(all_scores, all_docids)):
            qid_offset = position // self.num_results
            out.append((df.iloc[qid_offset]['qid'], df.iloc[qid_offset]['query'], self.docnos[docid], score))

        return pt.model.add_ranks( pd.DataFrame(out, columns=['qid', 'query', 'docno', 'score']) )

def _kannolo_retr_hsnw(self, m: int = 32, ef_construction: int = 200, ef_search: int = 100, num_results: int = 1000, early_exit_threshold : float = None) -> pt.Transformer:
    """Returns a retriever that searchers over a `kannolo` <https://github.com/TusKANNy/kannolo/>_ 
    HSWN index.

    Args:
        m (int): the number of nearest neighbors to consider during construction
        ef_construction (int): the size of the dynamic list used during construction
        ef_search (int): the size of the dynamic list used during search
        num_results (int): the number of results to return per query
        early_exit_threshold (float, optional): if set, the search will stop early 
            if the score of the last retrieved document is below this threshold. This 
            can be used to speed up search at the cost of potentially missing relevant documents.

    .. note::
        This transformer requires the ``kannolo`` package to be installed. Installation instructions are available
        in the `kannolo repository <https://github.com/TusKANNy/kannolo>`__.
        By using this method, you agree to cite the kANNolo paper (ECIR 2025) in any kind of material you produce 
        where it was used to conduct a search or experimentation, whether be it a research paper, dissertation, 
        article, poster, presentation, or documentation. For more on the kANNolo Citation License and the BibTex 
        see https://github.com/TusKANNy/kannolo?tab=readme-ov-file#citation-license

    .. cite.dblp conf/ecir/DelfinoEMNRV25

    """
    assert_kannolo()
    _citation_license()
    from kannolo import DensePlainHNSW
    
    assert len(self) < num_results, "Number of results must be less than the number of documents in the index"
    docnos, dvecs, meta = self.payload(return_docnos=True, return_dvecs=True)
    # we need to provide 1D array to kannolo, so we flatten the 2D array of document vectors into a 1D array.
    # this doesnt result in the index being loaded into memory.
    dvecs = dvecs.view(np.ndarray).reshape(-1)
    kindex = DensePlainHNSW.build_from_array(dvecs, m=m, ef_construction=ef_construction, dim=meta['vec_size'], metric="dotproduct")
    return KannoloRetriever(kindex, docnos, num_results=num_results, ef_search=ef_search, early_exit_threshold=early_exit_threshold)

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
