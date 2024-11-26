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
    
            prf_pipe = model >> index >> index.vec_loader() >> pyterier_dr.vector_prf() >> index 

    .. code-block:: bibtex
        :caption: Citation

        @article{DBLP:journals/tois/0009MZKZ23,
          author       = {Hang Li and
                          Ahmed Mourad and
                          Shengyao Zhuang and
                          Bevan Koopman and
                          Guido Zuccon},
          title        = {Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers:
                          Successes and Pitfalls},
          journal      = {{ACM} Trans. Inf. Syst.},
          volume       = {41},
          number       = {3},
          pages        = {62:1--62:40},
          year         = {2023},
          url          = {https://doi.org/10.1145/3570724},
          doi          = {10.1145/3570724},
          timestamp    = {Fri, 21 Jul 2023 22:26:51 +0200},
          biburl       = {https://dblp.org/rec/journals/tois/0009MZKZ23.bib},
          bibsource    = {dblp computer science bibliography, https://dblp.org}
        }
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

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Performs Vector PRF on the input dataframe."""
        pta.validate.result_frame(inp, extra_columns=['query_vec', 'doc_vec'])

        query_cols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']

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
    
            prf_pipe = model >> index >> index.vec_loader() >> pyterier_dr.average_prf() >> index 

    .. code-block:: bibtex
        :caption: Citation

        @article{DBLP:journals/tois/0009MZKZ23,
          author       = {Hang Li and
                          Ahmed Mourad and
                          Shengyao Zhuang and
                          Bevan Koopman and
                          Guido Zuccon},
          title        = {Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers:
                          Successes and Pitfalls},
          journal      = {{ACM} Trans. Inf. Syst.},
          volume       = {41},
          number       = {3},
          pages        = {62:1--62:40},
          year         = {2023},
          url          = {https://doi.org/10.1145/3570724},
          doi          = {10.1145/3570724},
          timestamp    = {Fri, 21 Jul 2023 22:26:51 +0200},
          biburl       = {https://dblp.org/rec/journals/tois/0009MZKZ23.bib},
          bibsource    = {dblp computer science bibliography, https://dblp.org}
        }
    """
    def __init__(self,
        *,
        k: int = 3
    ):
        self.k = k

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Performs Average PRF on the input dataframe."""
        pta.validate.result_frame(inp, extra_columns=['query_vec', 'doc_vec'])

        query_cols = [col for col in inp.columns if col.startswith('q') and col != 'query_vec']

        # get the docvectors for the top k docs and the query_vec
        all_vecs = np.stack([inp['query_vec'].iloc[0]] + [row.doc_vec for row in inp.head(self.k).itertuples()])
        # combine their average and add to the query
        query_vec = np.mean(all_vecs, axis=0)
        # generate new query dataframe with the existing query columns and the new query_vec
        return pd.DataFrame([[inp[c].iloc[0] for c in query_cols] + [query_vec]], columns=query_cols + ['query_vec'])

    def __repr__(self):
        return f"AveragePrf(k={self.k})"
