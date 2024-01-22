import itertools
import pyterrier as pt
import numpy as np
import pandas as pd
from .. import SimFn
from ..indexes import RankedLists
from . import FlexIndex
import ir_datasets

logger = ir_datasets.log.easy()

class NumpyRetriever(pt.Transformer):
    def __init__(self, flex_index, num_results=1000, batch_size=None, mask=None):
        self.flex_index = flex_index
        self.num_results = num_results
        self.batch_size = batch_size or 4096
        self.mask = mask

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        inp = inp.reset_index(drop=True)
        query_vecs = np.stack(inp['query_vec'])
        docnos, dvecs, config = self.flex_index.payload()
        if self.flex_index.sim_fn == SimFn.cos:
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        elif self.flex_index.sim_fn == SimFn.dot:
            pass # nothing to do
        else:
            raise ValueError(f'{self.flex_index.sim_fn} not supported')
        num_q = query_vecs.shape[0]
        ranked_lists = RankedLists(self.num_results, num_q)
        batch_it = range(0, dvecs.shape[0], self.batch_size)
        if self.flex_index.verbose:
            batch_it = pt.tqdm(batch_it)
        for idx_start in batch_it:
            doc_batch = dvecs[idx_start:idx_start+self.batch_size].T
            if self.flex_index.sim_fn == SimFn.cos:
                doc_batch = doc_batch / np.linalg.norm(doc_batch, axis=0, keepdims=True)
            scores = query_vecs @ doc_batch
            if self.mask is not None:
                scores = self.mask[idx_start:idx_start+self.batch_size].reshape(1, -1) * scores
            dids = np.arange(idx_start, idx_start+doc_batch.shape[1], dtype='i4').reshape(1, -1).repeat(num_q, axis=0)
            ranked_lists.update(scores, dids)
        result_scores, result_dids = ranked_lists.results()
        result_docnos = docnos.fwd[result_dids.flatten()]
        cols = {
            'score': np.concatenate(result_scores),
            'docno': result_docnos,
            'docid': np.concatenate(result_dids),
            'rank': np.concatenate([np.arange(len(scores)) for scores in result_scores]),
        }
        idxs = list(itertools.chain(*(itertools.repeat(i, len(scores)) for i, scores in enumerate(result_scores))))
        for col in inp.columns:
            if col != 'query_vec':
                cols[col] = inp[col][idxs].values
        return pd.DataFrame(cols)


class NumpyVectorLoader(pt.Transformer):
    def __init__(self, flex_index):
        self.flex_index = flex_index

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        docids = self.flex_index._load_docids(inp)
        dvecs, config = self.flex_index.payload(return_docnos=False)
        return inp.assign(doc_vec=list(dvecs[docids]))


class NumpyScorer(pt.Transformer):
    def __init__(self, flex_index, num_results=None):
        self.flex_index = flex_index
        self.num_results = num_results

    def score(self, query_vecs, docids):
        dvecs, config = self.flex_index.payload(return_docnos=False)
        doc_vecs = dvecs[docids]
        if self.flex_index.sim_fn == SimFn.cos:
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
            doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            return query_vecs @ doc_vecs.T
        elif self.flex_index.sim_fn == SimFn.dot:
            return query_vecs @ doc_vecs.T
        else:
            raise ValueError(f'{self.flex_index.sim_fn} not supported')

    def transform(self, inp):
        assert 'query_vec' in inp.columns
        assert 'docno' in inp.columns or 'docid' in inp.columns
        inp = inp.reset_index(drop=True)

        res_idxs = []
        res_scores = []
        res_ranks = []
        for qid, df in logger.pbar(inp.groupby('qid')):
            docids = self.flex_index._load_docids(df)
            query_vecs = df['query_vec'].iloc[0].reshape(1, -1)
            scores = self.score(query_vecs, docids).reshape(-1)
            index = df.index
            if self.num_results is not None and scores.shape[0] > self.num_results:
                idxs = np.argpartition(scores, -self.flex_index.num_results)[-self.num_results:]
                scores = scores[idxs]
                index = index[idxs]
            idxs = (-scores).argsort()
            res_scores.append(scores[idxs])
            res_idxs.append(index[idxs])
            res_ranks.append(np.arange(idxs.shape[0]))
        res_idxs = np.concatenate(res_idxs)
        res_scores = np.concatenate(res_scores)
        res_ranks = np.concatenate(res_ranks)
        res = pd.DataFrame({k: inp[k][res_idxs] for k in inp.columns if k not in ['score', 'rank']})
        res = res.assign(score=res_scores, rank=res_ranks)
        return res


def _np_retriever(self, num_results=1000, batch_size=None, mask=None):
        return NumpyRetriever(self, num_results=num_results, batch_size=batch_size, mask=mask)
FlexIndex.np_retriever = _np_retriever


def _np_vec_loader(self):
        return NumpyVectorLoader(self)
FlexIndex.np_vec_loader = _np_vec_loader
FlexIndex.vec_loader = _np_vec_loader # default vec_loader


def _np_scorer(self, num_results=None):
        return NumpyScorer(self, num_results)
FlexIndex.np_scorer = _np_scorer
FlexIndex.scorer = _np_scorer # default scorer
