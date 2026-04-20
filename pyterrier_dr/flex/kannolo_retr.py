

import pyterrier as pt
import pandas as pd
import pyterrier_alpha as pta
from pyterrier_dr.flex.core import FlexIndex
from pyterrier_dr.util import assert_kannolo
import numpy as np
import os

class KannoloRetriever(pt.Transformer):
    def __init__(self, kindex, docnos, *args, num_results: int = 1000, early_exit_threshold=None, ef_search: int = 100, drop_query_vec: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kindex = kindex
        self.num_results = num_results
        self.docnos = docnos
        self.early_exit_threshold = early_exit_threshold
        self.ef_search = ef_search
        self.drop_query_vec = drop_query_vec

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pta.validate.query_frame(df, extra_columns=['query_vec'])
        df = df.reset_index(drop=True)
        result = pta.DataFrameBuilder(['docno', 'score', 'rank'])
        if len(df) == 0:
            return result.to_df(df)

        qvecs = np.vstack(df['query_vec'].values).flatten()
        k = min(self.num_results, len(self.docnos))
        all_scores, all_docids = self.kindex.search(qvecs, k=k, ef_search=self.ef_search, early_exit_threshold=self.early_exit_threshold)
        all_scores = np.asarray(all_scores)
        all_docids = np.asarray(all_docids)
        if all_scores.ndim == 1:
            all_scores = all_scores.reshape(len(df), -1)
            all_docids = all_docids.reshape(len(df), -1)

        for scores, docids in zip(all_scores, all_docids):
            scores = scores.reshape(-1)
            docids = docids.reshape(-1)
            mask = docids != -1
            docids = docids[mask]
            scores = scores[mask]
            result.extend({
                'docno': [self.docnos[docid] for docid in docids],
                'score': scores,
                'rank': np.arange(docids.shape[0]),
            })

        if self.drop_query_vec:
            df = df.drop(columns='query_vec')
        return result.to_df(df)

def _kannolo_retr_hsnw(self, m: int = 32, ef_construction: int = 200, ef_search: int = 100, num_results: int = 1000, early_exit_threshold : float = None, drop_query_vec: bool = False) -> pt.Transformer:
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
        drop_query_vec (bool, optional): whether to drop the query vector from the output

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
    
    docnos, dvecs, meta = self.payload(return_docnos=True, return_dvecs=True)
    index_name = f'kannolo_hnsw-{m}_ef-{ef_construction}'
    index_path = os.path.join(str(self.index_path), index_name)
    if index_name not in self._cache:
        if not os.path.exists(index_path):
            # we need to provide 1D array to kannolo, so we flatten the 2D array of document vectors into a 1D array.
            # this doesnt result in the index being loaded into memory.
            dvecs = dvecs.view(np.ndarray).reshape(-1)
            kindex = DensePlainHNSW.build_from_array(dvecs, m=m, ef_construction=ef_construction, dim=meta['vec_size'], metric="dotproduct")
            kindex.save(index_path)
        else:
            kindex = DensePlainHNSW.load(index_path, metric="dotproduct")
        self._cache[index_name] = kindex
    else:
        kindex = self._cache[index_name]

    return KannoloRetriever(kindex, docnos, num_results=num_results, ef_search=ef_search, early_exit_threshold=early_exit_threshold, drop_query_vec=drop_query_vec)

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
