import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta


class MmrScorer(pt.Transformer):
    schematic = {'label': 'MMR'}
    """An MMR (Maximal Marginal Relevance) scorer (i.e., re-ranker).

    The MMR scorer re-orders documents by balancing relevance (from the initial scores) and diversity (based on the
    similarity of the document vectors).

    .. cite.dblp:: conf/sigir/CarbonellG98
    """
    def __init__(self, *, Lambda: float = 0.5, norm_rel: bool = False, norm_sim: bool = False, drop_doc_vec: bool = True, verbose: bool = False):
        """
        Args:
            Lambda: The balance parameter between relevance and diversity (default: 0.5)
            norm_rel: Whether to normalize relevance scores to [0, 1] (default: False)
            norm_sim: Whether to normalize similarity scores to [0, 1] (default: False)
            drop_doc_vec: Whether to drop the 'doc_vec' column after re-ranking (default: True)
            verbose: Whether to display verbose output (e.g., progress bars) (default: False)
        """
        self.Lambda = Lambda
        self.norm_rel = norm_rel
        self.norm_sim = norm_sim
        self.drop_doc_vec = drop_doc_vec
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.result_frame(inp, extra_columns=['doc_vec'])
        out = []

        if len(inp) == 0:
            return inp.assign(score=[], rank=[])

        it = inp.groupby('qid')
        if self.verbose:
            it = pt.tqdm(it, unit='q', desc=repr(self))

        for qid, frame in it:
            scores = frame['score'].values
            dvec_matrix = np.stack(frame['doc_vec'])
            dvec_matrix = dvec_matrix / np.linalg.norm(dvec_matrix, axis=1)[:, None]
            dvec_sims = dvec_matrix @ dvec_matrix.T
            if self.norm_rel:
                scores = (scores - scores.min()) / (scores.max() - scores.min())
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
                    marg_rels[idx] = float('inf') # ignore this document from now on
            new_frame = frame.iloc[new_idxs].reset_index(drop=True).assign(
                score=-np.arange(len(new_idxs), dtype=float),
                rank=np.arange(len(new_idxs))
            )
            if self.drop_doc_vec:
                new_frame = new_frame.drop(columns='doc_vec')
            out.append(new_frame)

        return pd.concat(out, ignore_index=True)

    __repr__ = pta.transformer_repr
