import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
from . import FlexIndex


class MmrReRanker(pt.Transformer):
    def __init__(self, flex_index: FlexIndex, *, Lambda: float = 0.5, norm_rel: bool = False, norm_sim: bool = False, verbose: bool = False):
        self.flex_index = flex_index
        self.Lambda = Lambda
        self.norm_rel = norm_rel
        self.norm_sim = norm_sim
        self.verbose = verbose

    def transform(self, inp):
        out = []
        with pta.validate.any(inp) as v:
            v.result_frame(extra_columns=['doc_vec'], mode='rerank')
            v.result_frame(mode='lookup_rerank')

        if v.mode == 'lookup_rerank':
            inp = self.flex_index.vec_loader()(inp)

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

def _mmr(self, *, Lambda: float = 0.5, norm_rel: bool = False, norm_sim: bool = False, verbose: bool = False) -> MmrReRanker:
    return MmrReRanker(self, Lambda=Lambda, norm_rel=norm_rel, norm_sim=norm_sim, verbose=verbose)
FlexIndex.mmr = _mmr
