from typing import Optional, Iterable, Tuple
import numpy as np
import pandas as pd
import ir_measures
import pyterrier as pt
from pyterrier_dr import FlexIndex


def ILS(index: FlexIndex, *, name: Optional[str] = None, verbose: bool = False) -> ir_measures.Measure: # noqa: N802
    """Create an ILS (Intra-List Similarity) measure calculated using the vectors in the provided index.

    Higher scores indicate lower diversity in the results.

    This measure supports the ``@k`` convention for applying a top-k cutoff before scoring.

    Args:
        index (FlexIndex): The index to use for loading document vectors.
        name (str, optional): The name of the measure (default: "ILS").
        verbose (bool, optional): Whether to display a progress bar.

    Returns:
        ir_measures.Measure: An ILS measure object.

    .. cite.dblp:: conf/www/ZieglerMKL05
    """
    return ir_measures.define(lambda qrels, results: _ils(results, index, verbose=verbose), name=name or 'ILS')


def ils(results: pd.DataFrame, index: Optional[FlexIndex] = None, *, verbose: bool = False) -> Iterable[Tuple[str, float]]:
    """Calculate the ILS (Intra-List Similarity) of a set of results.

    Higher scores indicate lower diversity in the results.

    Args:
        results: The result frame to calculate ILS for.
        index: The index to use for loading document vectors. Required if `results` does not have a `doc_vec` column.
        verbose: Whether to display a progress bar.

    Returns:
        Iterable[Tuple[str,float]]: An iterable of (qid, ILS) pairs.

    .. cite.dblp:: conf/www/ZieglerMKL05
    """
    return _ils(results.rename(columns={'docno': 'doc_id', 'qid': 'query_id'}), index, verbose=verbose)


def _ils(results: pd.DataFrame, index: Optional[FlexIndex] = None, *, verbose: bool = False) -> Iterable[Tuple[str, float]]:
    res = {}

    if index is not None:
        results = index.vec_loader()(results.rename(columns={'doc_id': 'docno'}))

    if 'doc_vec' not in results:
        raise ValueError('You must provide index to ils() if results do not have a `doc_vec` column.')

    it = results.groupby('query_id')
    if verbose:
        it = pt.tqdm(it, unit='q', desc='ILS')

    for qid, frame in it:
        if len(frame) > 1:
            vec_matrix = np.stack(frame['doc_vec'])
            vec_matrix = vec_matrix / np.linalg.norm(vec_matrix, axis=1)[:, None] # normalize vectors
            vec_sims = vec_matrix @ vec_matrix.T
            upper_right = np.triu_indices(vec_sims.shape[0], k=1)
            res[qid] = np.mean(vec_sims[upper_right])
        else:
            res[qid] = 0.0 # ILS is ill-defined when there's only one item.

    return res.items()
