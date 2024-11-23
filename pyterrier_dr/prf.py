import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


def vector_prf(*, alpha : float = 1, beta : float = 0.2, k : int = 3):
    """
    Performs a Rocchio-esque PRF by linearly combining the query_vec column with 
    the doc_vec column of the top k documents.

    Arguments:
     - alpha: weight of original query_vec
     - beta: weight of doc_vec
     - k: number of pseudo-relevant feedback documents

    Expected Input: ['qid', 'query', 'query_vec', 'doc_vec']
    Output: ['qid', 'query', 'query_vec']

    Example::
    
            prf_pipe = model >> index >> index.vec_loader() >> pyterier_dr.vector_prf() >> index 

    Reference: Hang Li, Ahmed Mourad, Shengyao Zhuang, Bevan Koopman, Guido Zuccon. [Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers: Successes and Pitfalls](https://arxiv.org/pdf/2108.11044.pdf)
    """
    def _vector_prf(inp):
        pta.validate.result_frame(inp, extra_columns=['query', 'query_vec', 'doc_vec'])

        # get the docvectors for the top k docs
        doc_vecs = np.stack([ row.doc_vec for row in inp.head(k).itertuples() ])
        # combine their average and add to the query
        query_vec = alpha * inp.iloc[0]['query_vec'] + beta * np.mean(doc_vecs, axis=0)
        # generate new query dataframe with 'qid', 'query', 'query_vec'
        return pd.DataFrame([[inp.iloc[0]['qid'], inp.iloc[0]['query'], query_vec]], columns=['qid', 'query', 'query_vec'])

    return pt.apply.by_query(_vector_prf, add_ranks=False)

def average_prf(*, k : int = 3):
    """
    Performs Average PRF (as described by Li et al.) by averaging the query_vec column with 
    the doc_vec column of the top k documents.

    Arguments:
     - k: number of pseudo-relevant feedback documents

    Expected Input: ['qid', 'query_vec', 'doc_vec']
    Output: ['qid', 'query', 'query_vec']

    Example::
    
            prf_pipe = model >> index >> index.vec_loader() >> pyterier_dr.average_prf() >> index 

    Reference: Hang Li, Ahmed Mourad, Shengyao Zhuang, Bevan Koopman, Guido Zuccon. [Pseudo Relevance Feedback with Deep Language Models and Dense Retrievers: Successes and Pitfalls](https://arxiv.org/pdf/2108.11044.pdf)
    
    """
    def _average_prf(inp):
        pta.validate.result_frame(inp, extra_columns=['query_vec', 'doc_vec'])

        # get the docvectors for the top k docs and the query_vec
        all_vecs = np.stack([inp.iloc[0]['query_vec']] + [row.doc_vec for row in inp.head(k).itertuples()])
        # combine their average and add to the query
        query_vec = np.mean(all_vecs, axis=0)
        # generate new query dataframe with 'qid', 'query', 'query_vec'
        return pd.DataFrame([[inp.iloc[0]['qid'], inp.iloc[0]['query'], query_vec]], columns=['qid', 'query', 'query_vec'])

    return pt.apply.by_query(_average_prf, add_ranks=False)
