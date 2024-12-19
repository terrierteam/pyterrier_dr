import pyterrier as pt
import pyterrier_dr
from . import FlexIndex


def _mmr(self, *, Lambda: float = 0.5, norm_rel: bool = False, norm_sim: bool = False, drop_doc_vec: bool = True, verbose: bool = False) -> pt.Transformer:
    """Returns an MMR (Maximal Marginal Relevance) scorer (i.e., re-ranker) over this index.

    The method first loads vectors from the index and then applies :class:`MmrScorer` to re-rank the results. See
    :class:`MmrScorer` for more details on MMR.

    Args:
        Lambda: The balance parameter between relevance and diversity (default: 0.5)
        norm_rel: Whether to normalize relevance scores to [0, 1] (default: False)
        norm_sim: Whether to normalize similarity scores to [0, 1] (default: False)
        drop_doc_vec: Whether to drop the 'doc_vec' column after re-ranking (default: True)
        verbose: Whether to display verbose output (e.g., progress bars) (default: False)

    .. cite.dblp:: conf/sigir/CarbonellG98
    """
    return self.vec_loader() >> pyterrier_dr.MmrScorer(Lambda=Lambda, norm_rel=norm_rel, norm_sim=norm_sim, drop_doc_vec=drop_doc_vec, verbose=verbose)
FlexIndex.mmr = _mmr
