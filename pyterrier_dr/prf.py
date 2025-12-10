import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


class VectorPrf(pt.Transformer):
    """
    Performs a Rocchio-esque PRF by linearly combining the query_vec column with 
    the doc_vec column of the top k documents.

    Arguments:
     - alpha: weight of original query_vec
     - beta: weight of doc_vec
     - k: number of pseudo-relevant feedback documents

    Expected Input Columns: ``['qid', 'query_vec', 'docno', 'doc_vec']``

    Output Columns: ``['qid', 'query_vec']`` (Any other query columns from the input are also pulled included in the output.)

    Example::
    
            prf_pipe = model >> index >> index.vec_loader() >> pyterrier_dr.VectorPrf() >> index 

    .. cite.dblp:: journals/tois/0009MZKZ23
    """
    def __init__(self,
        *,
        alpha: float = 1,
        beta: float = 0.2,
        k: int = 3
    ):
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def compile(self) -> pt.Transformer:
        return pt.RankCutoff(self.k) >> self

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Performs Vector PRF on the input dataframe."""
        pta.validate.result_frame(inp, extra_columns=['query_vec', 'doc_vec'])

        query_cols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']
        if len(inp) == 0:
            return pd.DataFrame([], columns=query_cols + ['query_vec'])

        # get the docvectors for the top k docs
        doc_vecs = np.stack([ row.doc_vec for row in inp.head(self.k).itertuples() ])
        # combine their average and add to the query
        query_vec = self.alpha * inp['query_vec'].iloc[0] + self.beta * np.mean(doc_vecs, axis=0)
        # generate new query dataframe with the existing query columns and the new query_vec
        return pd.DataFrame([[inp[c].iloc[0] for c in query_cols] + [query_vec]], columns=query_cols + ['query_vec'])

    def __repr__(self):
        return f"VectorPrf(alpha={self.alpha}, beta={self.beta}, k={self.k})"


class AveragePrf(pt.Transformer):
    """
    Performs Average PRF (as described by Li et al.) by averaging the query_vec column with 
    the doc_vec column of the top k documents.

    Arguments:
     - k: number of pseudo-relevant feedback documents

    Expected Input Columns: ``['qid', 'query_vec', 'docno', 'doc_vec']``

    Output Columns: ``['qid', 'query_vec']`` (Any other query columns from the input are also pulled included in the output.)

    Example::
    
            prf_pipe = model >> index >> index.vec_loader() >> pyterrier_dr.AveragePrf() >> index 

    .. cite.dblp:: journals/tois/0009MZKZ23
    """
    def __init__(self,
        *,
        k: int = 3
    ):
        self.k = k

    def compile(self) -> pt.Transformer:
        return pt.RankCutoff(self.k) >> self
    
    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Performs Average PRF on the input dataframe."""
        pta.validate.result_frame(inp, extra_columns=['query_vec', 'doc_vec'])

        query_cols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']

        if len(inp) == 0:
            return pd.DataFrame([], columns=query_cols + ['query_vec'])

        # get the docvectors for the top k docs and the query_vec
        all_vecs = np.stack([inp['query_vec'].iloc[0]] + [row.doc_vec for row in inp.head(self.k).itertuples()])
        # combine their average and add to the query
        query_vec = np.mean(all_vecs, axis=0)
        # generate new query dataframe with the existing query columns and the new query_vec
        return pd.DataFrame([[inp[c].iloc[0] for c in query_cols] + [query_vec]], columns=query_cols + ['query_vec'])

    def __repr__(self):
        return f"AveragePrf(k={self.k})"
