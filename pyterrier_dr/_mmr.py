import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


class MmrScorer(pt.Transformer):
    """An MMR (Maximal Marginal Relevance) scorer (i.e., re-ranker).

    The MMR scorer re-orders documents by balancing relevance (from the initial scores) and diversity (based on the
    similarity of the document vectors).

    .. cite.dblp:: conf/sigir/CarbonellG98
    """
    def __init__(self, *, Lambda: float = 0.5, norm_rel: bool = False, norm_sim: bool = False, verbose: bool = False):
        """
        Args:
            Lambda: The balance parameter between relevance and diversity (default: 0.5)
            norm_rel: Whether to normalize relevance scores to [0, 1] (default: False)
            norm_sim: Whether to normalize similarity scores to [0, 1] (default: False)
            verbose: Whether to display verbose output (e.g., progress bars) (default: False)
        """
        self.Lambda = Lambda
        self.norm_rel = norm_rel
        self.norm_sim = norm_sim
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.result_frame(inp, extra_columns=['doc_vec'])
        out = []

        it = inp.groupby('qid')
        if self.verbose:
            it = pt.tqdm(it, unit='q', desc=repr(self))

        for qid, frame in it:
            scores = frame['score'].values
            if self.norm_rel:
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            dvec_matrix = np.stack(frame['doc_vec'])
            dvec_matrix = dvec_matrix / np.linalg.norm(dvec_matrix, axis=1)[:, None]
            dvec_sims = dvec_matrix @ dvec_matrix.T
            if self.norm_sim:
                dvec_sims = (dvec_sims - dvec_sims.min()) / (dvec_sims.max() - dvec_sims.min())
            marg_rels = np.zeros_like(scores)
            new_idxs = []
            for _ in range(scores.shape[0]):
                mmr_scores = (self.Lambda * scores) - ((1 - self.Lambda) * marg_rels)
                idx = mmr_scores.argmax()
                new_idxs.append(idx)
                if marg_rels.shape[0] > 1:
                    marg_rels = np.max(np.stack([marg_rels, dvec_sims[idx]]), axis=0)
                    marg_rels[idx] = float('inf')
            new_frame = frame.iloc[new_idxs].reset_index(drop=True).copy()
            new_frame['score'] = -np.arange(len(new_idxs))
            new_frame['rank'] = np.arange(len(new_idxs))
            out.append(new_frame)

        return pd.concat(out, ignore_index=True)

    __repr__ = pta.transformer_repr
