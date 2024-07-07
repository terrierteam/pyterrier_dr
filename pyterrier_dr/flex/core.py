import shutil
import itertools
import json
from pathlib import Path
from warnings import warn
import numpy as np
import more_itertools
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
from npids import Lookup
from enum import Enum
from .. import SimFn
from ..indexes import RankedLists
import ir_datasets
import torch
from pyterrier_alpha import Artifact

logger = ir_datasets.log.easy()

class IndexingMode(Enum):
    create = "create"
    overwrite = "overwrite"
    # append???


class FlexIndex(Artifact, pt.Indexer):
    def __init__(self, index_path, num_results=1000, sim_fn=SimFn.dot, indexing_mode=IndexingMode.create, verbose=True):
        super().__init__(index_path)
        self.index_path = Path(index_path)
        self.num_results = num_results
        self.sim_fn = SimFn(sim_fn)
        self.indexing_mode = IndexingMode(indexing_mode)
        self.verbose = verbose
        self._meta = None
        self._docnos = None
        self._dvecs = None
        self._cache = {}

    def payload(self, return_dvecs=True, return_docnos=True):
        if self._meta is None:
            with open(self.index_path/'pt_meta.json', 'rt') as f_meta:
                meta = json.load(f_meta)
                assert meta.get('type') == 'dense_index' and meta['format'] == 'flex'
                self._meta = meta
        res = [self._meta]
        if return_dvecs:
            if self._dvecs is None:
                self._dvecs = np.memmap(self.index_path/'vecs.f4', mode='r', dtype=np.float32, shape=(self._meta['doc_count'], self._meta['vec_size']))
            res.insert(0, self._dvecs)
        if return_docnos:
            if self._docnos is None:
                self._docnos = Lookup(self.index_path/'docnos.npids')
            res.insert(0, self._docnos)
        return res

    def __len__(self):
        meta, = self.payload(return_dvecs=False, return_docnos=False)
        return meta['doc_count']

    def index(self, inp):
        if isinstance(inp, pd.DataFrame):
            inp = inp.to_dict(orient="records")
        inp = more_itertools.peekable(inp)
        path = Path(self.index_path)
        if path.exists():
            if self.indexing_mode == IndexingMode.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {self.index_path}. If you want to delete and re-create an existing index, you can pass indexing_mode=IndexingMode.overwrite')
        path.mkdir(parents=True, exist_ok=True)
        vec_size = None
        count = 0
        if self.verbose:
            inp = pt.tqdm(inp, desc='indexing', unit='dvec')
        with open(path/'vecs.f4', 'wb') as fout, Lookup.builder(path/'docnos.npids') as docnos:
            for d in inp:
                vec = d['doc_vec']
                if vec_size is None:
                    vec_size = vec.shape[0]
                elif vec_size != vec.shape[0]:
                    raise ValueError(f'Inconsistent vector shapes detected (expected {vec_size} but found {vec.shape[0]})')
                vec = vec.astype(np.float32)
                fout.write(vec.tobytes())
                docnos.add(d['docno'])
                count += 1
        with open(path/'pt_meta.json', 'wt') as f_meta:
            json.dump({"type": "dense_index", "format": "flex", "vec_size": vec_size, "doc_count": count}, f_meta)

    def transform(self, inp):
        columns = set(inp.columns)
        modes = [
            (['qid', 'query_vec'], self.np_retriever, "performing exhaustive saerch with FlexIndex.np_retriever -- note that other FlexIndex retrievers may be faster"),
        ]
        for fields, fn, note in modes:
            if all(f in columns for f in fields):
                warn(f'based on input columns {list(columns)}, {note}')
                return fn()(inp)
        message = f'Unexpected input with columns: {inp.columns}. Supports:'
        for fields, fn in modes:
            message += f'\n - {fn.__doc__.strip()}: {fields}'
        raise RuntimeError(message)

    def get_corpus_iter(self, start_idx=None, stop_idx=None, verbose=True):
        docnos, dvecs, meta = self.payload()
        docno_iter = iter(docnos)
        if start_idx is not None or stop_idx is not None:
            docno_iter = itertools.islice(docno_iter, start_idx, stop_idx)
            dvecs = dvecs[start_idx:stop_idx]
        it = zip(docno_iter, range(dvecs.shape[0]))
        if self.verbose:
            it = pt.tqdm(it, total=dvecs.shape[0], desc=f'{str(self.index_path)} dvecs')
        for docno, i in it:
            yield {'docno': docno, 'doc_vec': dvecs[i]}

    def np_retriever(self, batch_size=None, num_results=None):
        return FlexIndexNumpyRetriever(self, batch_size, num_results=num_results or self.num_results)

    def torch_retriever(self, batch_size=None):
        return FlexIndexTorchRetriever(self, batch_size)

    def vec_loader(self):
        return FlexIndexVectorLoader(self)

    def scorer(self):
        return FlexIndexScorer(self)

    def _load_docids(self, inp):
        assert 'docid' in inp.columns or 'docno' in inp.columns
        if 'docid' in inp.columns:
            return inp['docid'].values
        docnos, config = self.payload(return_dvecs=False)
        return docnos.inv[inp['docno'].values] # look up docids from docnos

    def built(self):
        return self.index_path.exists()


class FlexIndexNumpyRetriever(pt.Transformer):
    def __init__(self, flex_index, batch_size=None, num_results=None):
        self.flex_index = flex_index
        self.batch_size = batch_size or 4096
        self.num_results = num_results or self.flex_index.num_results

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
        res = []
        ranked_lists = RankedLists(self.num_results, num_q)
        batch_it = range(0, dvecs.shape[0], self.batch_size)
        if self.flex_index.verbose:
            batch_it = pt.tqdm(batch_it)
        for idx_start in batch_it:
            doc_batch = dvecs[idx_start:idx_start+self.batch_size].T
            if self.flex_index.sim_fn == SimFn.cos:
                doc_batch = doc_batch / np.linalg.norm(doc_batch, axis=0, keepdims=True)
            scores = query_vecs @ doc_batch
            dids = np.arange(idx_start, idx_start+doc_batch.shape[1], dtype='i4').reshape(1, -1).repeat(num_q, axis=0)
            ranked_lists.update(scores, dids)
        result_scores, result_dids = ranked_lists.results()
        result_docnos = [docnos.fwd[d] for d in result_dids]
        cols = {
            'score': np.concatenate(result_scores),
            'docno': np.concatenate(result_docnos),
            'docid': np.concatenate(result_dids),
            'rank': np.concatenate([np.arange(len(scores)) for scores in result_scores]),
        }
        idxs = list(itertools.chain(*(itertools.repeat(i, len(scores)) for i, scores in enumerate(result_scores))))
        for col in inp.columns:
            if col != 'query_vec':
                cols[col] = inp[col][idxs].values
        return pd.DataFrame(cols)


class FlexIndexTorchRetriever(pt.Transformer):
    def __init__(self, flex_index, batch_size=None):
        self.flex_index = flex_index
        self.batch_size = batch_size or 4096
        docnos, meta, = flex_index.payload(return_dvecs=False)
        SType, TType, CTType, SIZE = torch.FloatStorage, torch.FloatTensor, torch.cuda.FloatTensor, 4
        self._cpu_data = TType(SType.from_file(str(self.flex_index.index_path/'vecs.f4'), size=meta['doc_count'] * meta['vec_size'])).reshape(meta['doc_count'], meta['vec_size'])
        self._cuda_data = CTType(size=(self.batch_size, meta['vec_size']), device='cuda')
        self._docnos = docnos

    def transform(self, inp):
        columns = set(inp.columns)
        assert all(f in columns for f in ['qid', 'query_vec']), "TorchIndex expects columns ['qid', 'query_vec'] when used in a pipeline"
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = torch.from_numpy(query_vecs).cuda() # TODO: can this go directly to CUDA? device='cuda' doesn't work

        step = self._cuda_data.shape[0]
        it = range(0, self._cpu_data.shape[0], step)
        if self.flex_index.verbose:
            it = pt.tqdm(it, desc='TorchIndex scoring', unit='docbatch')

        ranked_lists = RankedLists(self.flex_index.num_results, query_vecs.shape[0])
        for start_idx in it:
            end_idx = start_idx + step
            batch = self._cpu_data[start_idx:end_idx]
            bsize = batch.shape[0]
            self._cuda_data[:bsize] = batch

            scores = query_vecs @ self._cuda_data[:bsize].T
            if scores.shape[0] > self.flex_index.num_results:
                scores, dids = torch.topk(scores, k=self.flex_index.num_results, dim=1)
            else:
                scores, dids = torch.sort(scores, dim=1, descending=False)
            scores = scores.cpu().float().numpy()

            dids = (dids + start_idx).cpu().numpy()
            ranked_lists.update(scores, dids)

        result_scores, result_dids = ranked_lists.results()
        result_docnos = self._docnos.fwd[result_dids]
        res = []
        for query, scores, docnos in zip(inp.itertuples(index=False), result_scores, result_docnos):
            for score, docno in zip(scores, docnos):
                res.append((*query, docno, score))
        res = pd.DataFrame(res, columns=list(query._fields) + ['docno', 'score'])
        res = res[~res.score.isna()]
        res = add_ranks(res)
        return res


def _load_dvecs(flex_index, inp):
    assert 'docid' in inp.columns or 'docno' in inp.columns
    docnos, dvecs, config = flex_index.payload()
    if 'docid' in inp.columns:
        docids = inp['docid'].values
    else:
        docids = docnos.inv[inp['docno'].values]
    return dvecs[docids]


class FlexIndexVectorLoader(pt.Transformer):
    def __init__(self, flex_index):
        self.flex_index = flex_index

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        return inp.assign(doc_vec=list(_load_dvecs(self.flex_index, inp)))


class FlexIndexScorer(pt.Transformer):
    def __init__(self, flex_index):
        self.flex_index = flex_index

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        assert 'query_vec' in inp.columns
        doc_vecs = _load_dvecs(self.flex_index, inp)
        query_vecs = np.stack(inp['query_vec'])
        if self.flex_index.sim_fn == SimFn.cos:
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
            doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            scores = (query_vecs * doc_vecs).sum(axis=1)
        elif self.flex_index.sim_fn == SimFn.dot:
            scores = (query_vecs * doc_vecs).sum(axis=1)
        else:
            raise ValueError(f'{self.flex_index.sim_fn} not supported')
        res = inp.assign(score=scores)
        res = add_ranks(res)
        return res
