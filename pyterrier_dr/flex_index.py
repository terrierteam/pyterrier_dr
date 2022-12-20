import math
import os
import shutil
import threading
import struct
import tempfile
import itertools
import json
from pathlib import Path
import numpy as np
import shutil
import more_itertools
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
from npids import Lookup
from enum import Enum
from . import SimFn
from .indexes import RankedLists
import ir_datasets
import torch

logger = ir_datasets.log.easy()

class IndexingMode(Enum):
    create = "create"
    overwrite = "overwrite"
    # append???


class FlexIndex(pt.Indexer):
    def __init__(self, index_path, num_results=1000, sim_fn=SimFn.dot, indexing_mode=IndexingMode.create, verbose=True):
        self.index_path = Path(index_path)
        self.num_results = num_results
        self.sim_fn = SimFn(sim_fn)
        self.indexing_mode = IndexingMode(indexing_mode)
        self.verbose = verbose
        self._meta = None
        self._docnos = None
        self._dvecs = None
        self._faiss_flat = None
        self._faiss_flat_gpu = None
        self._faiss_ivf = {}
        self._faiss_hnsw = {}
        self._inmem = False

    def payload(self, return_dvecs=True, return_docnos=True):
        inmem = False
        path = self.index_path
        if self._meta is None:
            with open(path/'pt_meta.json', 'rt') as f_meta:
                meta = json.load(f_meta)
                assert meta.get('type') == 'dense_index' and meta['format'] == 'flex'
                self._meta = meta
        res = [self._meta]
        if return_dvecs:
            if self._dvecs is None or (inmem and not self._inmem):
                if inmem:
                    if self._dvecs is not None:
                        self._dvecs.close()
                    with (path/'vecs.f4').open('rb') as fin:
                        self._dvecs = np.fromfile(fin, dtype=np.float32).reshape(self._meta['doc_count'], self._meta['vec_size'])
                    self._inmem = True
                else:
                    self._dvecs = np.memmap(path/'vecs.f4', mode='r', dtype=np.float32, shape=(self._meta['doc_count'], self._meta['vec_size']))
            res.insert(0, self._dvecs)
        if return_docnos:
            if self._docnos is None:
                self._docnos = Lookup(path/'docnos.npids')
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

    def np_retriever(self, batch_size=None):
        return FlexIndexNumpyRetriever(self, batch_size)

    def torch_retriever(self, batch_size=None):
        return FlexIndexTorchRetriever(self, batch_size)

    def faiss_flat_retriever(self, gpu=False):
        import faiss
        if not self._faiss_flat:
            meta, = self.payload(return_dvecs=False, return_docnos=False)
            with tempfile.TemporaryDirectory() as tmp, logger.duration('reading faiss flat'):
                DUMMY = b'\x00\x00\x10\x00\x00\x00\x00\x00'
                #       [ type ]                                                                             [ train ]  [ metric (dot)     ]
                header = b'IxFI'  + struct.pack('<IQ', meta['vec_size'], meta['doc_count']) + DUMMY + DUMMY  + b'\x00' + b'\x00\x00\x00\x00' + struct.pack('<Q', meta['doc_count'] * meta['vec_size'])
                #os.mkfifo(os.path.join(tmp, 'tmp'))
                #def write():
                #    with open(os.path.join(tmp, 'tmp'), 'wb') as fout, (self.index_path/'vecs.f4').open('rb') as fin:
                #        fout.write(header)
                #        shutil.copyfileobj(fin, fout)
                #thread = threading.Thread(target=write).run()
                #reader = faiss.FileIOReader(os.path.join(tmp, 'tmp'))
                with open(os.path.join(tmp, 'tmp'), 'wb') as fout:
                    fout.write(b' ' + header)
                reader = faiss.BufferedIOReader(faiss.FileIOReader(os.path.join(tmp, 'tmp')))
                reader.read_bytes(1)
                reader.reader = faiss.FileIOReader(str(self.index_path/'vecs.f4'))
                self._faiss_flat = faiss.read_index(reader)
        if gpu:
            if not self._faiss_flat_gpu:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                self._faiss_flat_gpu = faiss.index_cpu_to_all_gpus(self._faiss_flat, co=co)
            return FaissRetriever(self, self._faiss_flat_gpu)
        return FaissRetriever(self, self._faiss_flat)
    """
    def faiss_lsh_retriever(self, n_bits=None):
        import faiss
        meta, = self.payload(return_dvecs=False, return_docnos=False)
        n_bits = meta['vec_size'] * 2
        key = (n_bits,)
        index_name = f'lsh_n-{n_bits}.faiss'
        if key not in self._faiss_lsh:
            dvecs, meta = self.payload(return_docnos=False)
            if not os.path.exists(self.index_path/index_name):
                import pdb; pdb.set_trace()
                idx = faiss.IndexLSH.new_with_options(meta['vec_size'], n_bits, train_thresholds=True)
                train = self._sample_train()
                print(f'sampled {train.shape}')
                idx.train(train)
                for start_idx in pt.tqdm(range(0, dvecs.shape[0], 4096), desc='indexing', unit='batch'):
                    idx.add(np.array(dvecs[start_idx:start_idx+4096]))
                faiss.write_index(idx, str(self.index_path/index_name))
                self._faiss_lsh[key] = idx
            else:
                self._faiss_lsh[key] = faiss.read_index(str(self.index_path/index_name))
        return FaissRetriever(self, self._faiss_lsh[key])
    """

    def faiss_ivf_retriever(self, train_sample=None, n_list=None, cache=True):
        import faiss
        meta, = self.payload(return_dvecs=False, return_docnos=False)

        if n_list is None:
            if train_sample is None:
                n_list = int(1 << math.ceil(math.log2(meta['doc_count'] * 0.001 / 39)))
            else:
                n_list = math.floor(train_sample / 39)
            n_list = max(n_list, 4)

        if train_sample is None:
            train_sample = n_list * 39
        elif 0 < train_sample < 1:
            train_sample = math.ceil(train_sample * meta['doc_count'])

        key = (train_sample, n_list)
        index_name = f'ivf_train-{train_sample}_nlist-{n_list}.faiss'
        if key not in self._faiss_ivf:
            dvecs, meta = self.payload(return_docnos=False)
            if not os.path.exists(self.index_path/index_name):
                quantizer = faiss.IndexFlatIP(meta['vec_size'])
                quantizer = faiss.index_cpu_to_all_gpus(quantizer)
                idx = faiss.IndexIVFFlat(quantizer, meta['vec_size'], n_list, faiss.METRIC_INNER_PRODUCT)
                with logger.duration(f'loading {train_sample} train samples'):
                    train = self._sample_train(train_sample)
                with logger.duration(f'training ivf with {n_list} posting lists'):
                    idx.train(train)
                for start_idx in pt.tqdm(range(0, dvecs.shape[0], 4096), desc='indexing', unit='batch'):
                    idx.add(np.array(dvecs[start_idx:start_idx+4096]))
                if cache:
                    with logger.duration('caching index'):
                        idx.quantizer = faiss.index_gpu_to_cpu(idx.quantizer)
                        faiss.write_index(idx, str(self.index_path/index_name))
                self._faiss_ivf[key] = idx
            else:
                with logger.duration('reading index'):
                    self._faiss_ivf[key] = faiss.read_index(str(self.index_path/index_name))
        return FaissRetriever(self, self._faiss_ivf[key], n_probe=n_probe)

    def faiss_hnsw_retriever(self, cache=True, ef_construction=40, ef_search=16, neighbours=32):
        import faiss
        meta, = self.payload(return_dvecs=False, return_docnos=False)

        key = (ef_construction, neighbours)
        index_name = f'hnsw_constr-{ef_construction}_neigh-{neighbours}.faiss'
        if key not in self._faiss_hnsw:
            dvecs, meta = self.payload(return_docnos=False)
            if not os.path.exists(self.index_path/index_name):
                idx = faiss.IndexHNSWFlat(meta['vec_size'], neighbours, faiss.METRIC_INNER_PRODUCT)
                for start_idx in pt.tqdm(range(0, dvecs.shape[0], 4096), desc='indexing', unit='batch'):
                    idx.add(np.array(dvecs[start_idx:start_idx+4096]))
                if cache:
                    with logger.duration('caching index'):
                        import pdb;pdb.set_trace()
                        faiss.write_index(idx, str(self.index_path/index_name))
                self._faiss_hnsw[key] = idx
            else:
                with logger.duration('reading hnsw table'):
                    self._faiss_hnsw[key] = faiss.read_index(str(self.index_path/index_name))
                self._faiss_hnsw[key].storage = self.faiss_flat_retriever().faiss_index
        return FaissRetriever(self, self._faiss_hnsw[key], ef_search=ef_search)

    def vec_loader(self):
        return FlexIndexVectorLoader(self)

    def scorer(self):
        return FlexIndexScorer(self)

    def _sample_train(self, count=None):
        dvecs, meta = self.payload(return_docnos=False)
        #count = math.ceil(0.001 * dvecs.shape[0])
        count = count or min(10_000, dvecs.shape[0])
        idxs = np.random.RandomState(0).choice(dvecs.shape[0], size=count, replace=False)
        return dvecs[idxs]


class FlexIndexNumpyRetriever(pt.Transformer):
    def __init__(self, flex_index, batch_size=None):
        self.flex_index = flex_index
        self.batch_size = batch_size or 4096

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
        ranked_lists = RankedLists(self.flex_index.num_results, num_q)
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
        result_docnos = docnos.fwd[result_dids]
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


def _load_dvecs(flex_index, dvec):
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


class FaissRetriever(pt.Indexer):
    def __init__(self, flex_index, faiss_index, n_probe=None, ef_search=None):
        self.flex_index = flex_index
        self.faiss_index = faiss_index
        self.n_probe = n_probe
        self.ef_search = ef_search

    def transform(self, inp):
        inp = inp.reset_index(drop=True)
        assert all(f in inp.columns for f in ['qid', 'query_vec'])
        docnos, config = self.flex_index.payload(return_dvecs=False)
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = query_vecs.copy()
        idxs = []
        res = {'docid': [], 'score': [], 'rank': []}
        num_q = query_vecs.shape[0]
        QBATCH = 64
        if self.n_probe is not None:
            self.faiss_index.nprobe = self.n_probe
        if self.ef_search is not None:
            self.faiss_index.hnsw.efSearch = self.ef_search
        for qidx in range(0, num_q, QBATCH):
            scores, dids = self.faiss_index.search(query_vecs[qidx:qidx+QBATCH], self.flex_index.num_results)
            for i, (s, d) in enumerate(zip(scores, dids)):
                mask = d != -1
                d = d[mask]
                s = s[mask]
                res['docid'].append(d)
                res['score'].append(s)
                res['rank'].append(np.arange(d.shape[0]))
                idxs.extend(itertools.repeat(qidx+i, d.shape[0]))
        res = {k: np.concatenate(v) for k, v in res.items()}
        res['docno'] = docnos.fwd[res['docid']]
        for col in inp.columns:
            if col != 'query_vec':
                res[col] = inp[col][idxs].values
        return pd.DataFrame(res)
