# Deprecated module
import torch
import itertools
import math
import json
from pathlib import Path
from os.path import commonprefix
import numpy as np
import shutil
import more_itertools
import pandas as pd
import pyterrier as pt
from pyterrier.model import add_ranks
import ir_datasets

_logger = ir_datasets.log.easy()


class DocnoFile:
    def __init__(self, path):
        with open(path, 'rb') as f:
            metadata_size = int.from_bytes(f.read(2), byteorder='little')
            metadata = json.loads(f.read(metadata_size))
        self.prefix = metadata['prefix']
        self.max_length = metadata['max_length']
        self.count = metadata['count']
        if 'start' in metadata:
            self.start = metadata['start']
            self.step = metadata['step']
            self.fmt_string = metadata['fmt_string']
            self.mmap = None
        else:
            self.mmap = np.memmap(path, f'S{self.max_length}', offset=metadata_size+2)

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.int64, np.int32)):
            if self.mmap is not None:
                return self.prefix + self.mmap[idx].decode()
            else:
                assert 0 <= idx < self.count
                return self._fmt_idx(idx)
        elif isinstance(idx, str):
            if self.mmap is not None:
                raise RuntimeError("reverse lookup not supported")
            else:
                if not idx.startswith(self.prefix):
                    return None
                idx = idx[len(self.prefix):]
                if not idx.isnumeric():
                    return None
                idx = int(idx)
                if idx % self.step != 0:
                    return None
                idx = (idx // self.step) - self.start
                if idx < 0 or idx >= self.count:
                    return None
                return idx
        else:
            if self.mmap is not None:
                docnos = self.mmap[idx]
                return np.array([self.prefix + d.decode() for d in docnos], dtype=object)
            else:
                assert ((0 <= idx) & (idx < self.count)).all()
                docnos = idx * self.step + self.start
                return np.array([self.prefix + format(d, self.fmt_string) for d in docnos], dtype=object)

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.mmap is not None:
            for item in self.mmap:
                yield self.prefix + item.decode()
        else:
            for idx in range(self.count):
                yield self._fmt_idx(idx)

    def __del__(self):
        del self.mmap
        self.mmap = None

    def _fmt_idx(self, idx):
        return self.prefix + format(idx * self.step + self.start, self.fmt_string)

    @staticmethod
    def build(docnos, path):
        assert len(docnos) > 0
        prefix = commonprefix(docnos) # reduce the size by removing common docno prefixes
        docnos = [d[len(prefix):].encode() for d in docnos] # remove prefix
        max_length = max(len(d) for d in docnos)
        metadata = {'count': len(docnos), 'prefix': prefix, 'max_length': max_length}
        if all(d.decode().isnumeric() for d in docnos) and len(docnos) >= 2:
            # Special case: are the docnos (1) all numeric, (2) incrementing at a constant step?
            start = int(docnos[0])
            step = int(docnos[1]) - start
            fmt_string = f'0{max_length}d' if any(d!=b'0' and d.startswith(b'0') for d in docnos) else 'd'
            for d, expected in zip(docnos, itertools.count(start, step)):
                if d.decode() != format(expected, fmt_string):
                    break # pattern broken
            else:
                # everything was numeric and incremental
                metadata['start'] = start
                metadata['step'] = step
                metadata['fmt_string'] = fmt_string
        with open(path, 'wb') as f:
            b_metadata = json.dumps(metadata).encode()
            f.write(len(b_metadata).to_bytes(2, byteorder='little'))
            f.write(b_metadata)
            if 'start' not in metadata:
                f.write(np.array(docnos, dtype=f'S{max_length}').tobytes())


class NilIndex(pt.Indexer):
    # simulates an indexer; just used for testing
    def transform(self, res):
        return res

    def index(self, it):
        for _ in it:
            pass


class NumpyIndex(pt.Indexer):
    def __init__(self, index_path=None, num_results=1000, score_fn='dot', overwrite=False, batch_size=4096, verbose=False, dtype='f4', drop_query_vec=True, inmem=False, cuda=False):
        self.index_path = Path(index_path)
        self.num_results = num_results
        self.score_fn = score_fn
        self.overwrite = overwrite
        self.verbose = verbose
        self.batch_size = batch_size
        self.dtype = dtype
        self.drop_query_vec = drop_query_vec
        self._docnos = None
        self._data = None
        self.inmem = inmem
        self.cuda = cuda

    def docnos_and_data(self):
        if self._data is None:
            path = self.index_path
            with open(path/'meta.json', 'rt') as f_meta:
                meta = json.load(f_meta)
            if self.inmem:
                with open(path/'index.npy', 'rb') as fin:
                    self._data = np.frombuffer(fin.read(), dtype=self.dtype).reshape(meta['count'], meta['vec_size'])
                if self.cuda:
                    self._data = torch.tensor(self._data, dtype=torch.float16, device='cuda')
            else:
                self._data = np.memmap(path/'index.npy', mode='r', dtype=self.dtype, shape=(meta['count'], meta['vec_size']))
            self._docnos = DocnoFile(path/'docnos.npy')
        return self._docnos, self._data

    def __len__(self):
        with open(self.index_path/'meta.json', 'rt') as f_meta:
            meta = json.load(f_meta)
        return meta['count']

    def transform(self, inp):
        columns = set(inp.columns)
        if all(c in columns for c in ('qid', 'query_vec', 'docno')):
            return self.transform_R(inp)
        if all(c in columns for c in ('qid', 'query_vec')):
            return self.transform_Q(inp)
        raise RuntimeError('NumpyIndex expects input columns of either:\n'
            ' - ["qid", "query_vec", "docno", ...]: Re-ranking\n'
            ' - ["qid", "query_vec", ...]: Retrieval\n'
            '(Are you encoding the query?)')

    def transform_Q(self, inp):
        query_vecs = np.stack(inp['query_vec'])
        if self.score_fn == 'cos':
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        if self.cuda:
            query_vecs = torch.from_numpy(query_vecs).cuda()
        num_q = query_vecs.shape[0]
        res = []
        docnos, data = self.docnos_and_data()
        ranked_lists = RankedLists(self.num_results, num_q)
        if self.cuda and self.inmem:
            scores = query_vecs.half() @ data.T
            scores, dids = torch.topk(scores, k=self.num_results, dim=1)
            scores = scores.cpu().float().numpy()
            dids = dids.cpu().numpy()
            ranked_lists.update(scores, dids)
        else:
            batch_it = range(0, data.shape[0], self.batch_size)
            if self.verbose:
                batch_it = pt.tqdm(batch_it)
            for idx_start in batch_it:
                doc_batch = data[idx_start:idx_start+self.batch_size].T
                if self.score_fn == 'cos':
                    doc_batch = doc_batch / np.linalg.norm(doc_batch, axis=0, keepdims=True)
                if self.cuda:
                    doc_batch = torch.from_numpy(doc_batch).cuda()
                scores = query_vecs @ doc_batch
                if self.cuda:
                    scores = scores.cpu().numpy()
                dids = np.arange(idx_start, idx_start+doc_batch.shape[1], dtype='i4').reshape(1, -1).repeat(num_q, axis=0)
                ranked_lists.update(scores, dids)
        result_scores, result_dids = ranked_lists.results()
        unique_dids, unique_inverse = np.unique(result_dids, return_inverse=True)
        unique_docnos = docnos[unique_dids]
        result_docnos = unique_docnos[unique_inverse].reshape(num_q, -1)
        for query, scores, docnos in zip(inp.itertuples(index=False), result_scores, result_docnos):
            if self.drop_query_vec:
                query = query._replace(query_vec=None)
            for score, docno in zip(scores, docnos):
                res.append((*query, docno, score))
        res = pd.DataFrame(res, columns=list(query._fields) + ['docno', 'score'])
        if self.drop_query_vec:
            res = res.drop(columns=['query_vec'])
        res = res[~res.score.isna()]
        res = add_ranks(res)
        return res

    def transform_R(self, inp):
        assert not self.cuda, "cuda not yet supported for re-ranking"
        docnos, data = self.docnos_and_data()
        all_scores = []
        it = range(0, len(inp), self.batch_size)
        if self.verbose:
            it = pt.tqdm(it, desc='re-ranking from index', unit='batch')
        for start_idx in it:
            end_idx = start_idx + self.batch_size
            query_vecs = np.stack(inp.iloc[start_idx:end_idx]['query_vec'])
            dids = np.array([docnos[str(d)] for d in inp.iloc[start_idx:end_idx]['docno']]) # reverse lookup
            doc_vecs = data[dids]
            if self.score_fn == 'cos':
                query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
                doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            all_scores.append((query_vecs * doc_vecs).sum(axis=1))
        all_scores = np.concatenate(all_scores, axis=0)
        res = inp.assign(score=all_scores)
        if self.drop_query_vec:
            res = res.drop(columns=['query_vec'])
        res = res[~res.score.isna()]
        res = add_ranks(res)
        return res

    def index(self, inp):
        if isinstance(inp, pd.DataFrame):
            inp = inp.to_dict(orient="records")
        inp = more_itertools.peekable(inp)
        path = Path(self.index_path)
        if path.exists():
            if self.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {self.index_path}. If you want to delete and re-create an existing index, you can pass overwrite=True')
        path.mkdir(parents=True, exist_ok=True)
        docnos = []
        vec_size = None
        count = 0
        with open(path/'index.npy', 'wb') as fout:
            for docs in more_itertools.chunked(inp, self.batch_size):
                doc_vecs = np.stack([d['doc_vec'] for d in docs])
                if vec_size is None:
                    vec_size = doc_vecs.shape[1]
                elif vec_size != doc_vecs.shape[1]:
                    raise ValueError('Inconsistent vector shapes detected')
                doc_vecs = doc_vecs.astype(self.dtype)
                fout.write(doc_vecs.tobytes())
                docnos.extend([d['docno'] for d in docs])
                count += len(docs)
        DocnoFile.build(docnos, path/'docnos.npy')
        with open(path/'meta.json', 'wt') as f_meta:
            json.dump({'dtype': self.dtype, 'vec_size': vec_size, 'count': count}, f_meta)

    def get_corpus_iter(self, start_idx=None, stop_idx=None, verbose=True):
        docnos, data = self.docnos_and_data()
        docno_iter = iter(docnos)
        if start_idx is not None or stop_idx is not None:
            docno_iter = itertools.islice(docno_iter, start_idx, stop_idx)
            data = data[start_idx:stop_idx]
        with pt.tqdm(total=data.shape[0], desc=f'{str(self.index_path)} document vectors') as pbar:
            for start_idx in range(0, data.shape[0], self.batch_size):
                vec_batch = data[start_idx:start_idx+self.batch_size]
                for vec_idx in range(vec_batch.shape[0]):
                    pbar.update()
                    yield {'docno': next(docno_iter), 'doc_vec': vec_batch[vec_idx]}


class MemIndex(pt.Indexer):
    def __init__(self, num_results=1000, score_fn='dot', batch_size=4096, verbose=True, dtype='f4', drop_query_vec=True):
        # check we havent been passed a destination index, as per disk-based indexers
        assert isinstance(num_results, int)
        self.num_results = num_results
        self.score_fn = score_fn
        self.verbose = verbose
        self.batch_size = batch_size
        self.dtype = dtype
        self.drop_query_vec = drop_query_vec
        self._docnos = None
        self._data = None

    def docnos_and_data(self):
        return self._docnos, self._data

    def clear(self):
        self._docnos = None
        self._data = None

    def __len__(self):
        return self._data.shape[0]

    def transform(self, inp):
        columns = set(inp.columns)
        assert all(f in columns for f in ['qid', 'query_vec']), "NumpyIndex expects columns: ['qid', 'query_vec'] when used in a pipeline"
        query_vecs = np.stack(inp['query_vec'])
        if self.score_fn == 'cos':
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        num_q = query_vecs.shape[0]
        res = []
        docnos, data = self.docnos_and_data()
        data = data.T
        assert docnos is not None or data is not None, "You must first call index()"
        ranked_lists = RankedLists(self.num_results, num_q)
        if self.score_fn == 'cos':
            data = data / np.linalg.norm(data, axis=0, keepdims=True)
        scores = query_vecs @ data
        dids = np.arange(data.shape[1], dtype='i4').reshape(1, -1).repeat(num_q, axis=0)
        ranked_lists.update(scores, dids)
        result_scores, result_dids = ranked_lists.results()
        for query, scores, dids in zip(inp.itertuples(index=False), result_scores, result_dids):
            if self.drop_query_vec:
                query = query._replace(query_vec=None)
            for score, did in zip(scores, dids):
                res.append((*query, docnos[int(did)], score))
        res = pd.DataFrame(res, columns=list(query._fields) + ['docno', 'score'])
        if self.drop_query_vec:
            res = res.drop(columns=['query_vec'])
        res = res[~res.score.isna()]
        res = add_ranks(res)
        return res

    def index(self, inp):
        if isinstance(inp, pd.DataFrame):
            inp = inp.to_dict(orient="records")
        inp = more_itertools.peekable(inp)
        if self.verbose:
            inp = pt.tqdm(inp, desc='indexing', unit='doc')
        docnos = []
        vec_size = None
        count = 0
        data = []
        for docs in more_itertools.chunked(inp, self.batch_size):
            doc_vecs = np.stack([d['doc_vec'] for d in docs])
            if vec_size is None:
                vec_size = doc_vecs.shape[1]
            doc_vecs = doc_vecs.astype(self.dtype)
            data.append(doc_vecs)
            docnos.extend([d['docno'] for d in docs])
            count += len(docs)
        self._data = np.concatenate(data, axis=0)
        self._docnos = docnos

    def get_corpus_iter(self, start_idx=None, stop_idx=None, verbose=True):
        docnos, data = self.docnos_and_data()
        docno_iter = iter(docnos)
        if start_idx is not None or stop_idx is not None:
            docno_iter = iter(docnos[start_idx:stop_idx])
            data = data[start_idx:stop_idx]
        with pt.tqdm(total=data.shape[0], desc=f'{str(self.index_path)} document vectors') as pbar:
            for start_idx in range(0, data.shape[0], self.batch_size):
                vec_batch = data[start_idx:start_idx+self.batch_size]
                for vec_idx in range(vec_batch.shape[0]):
                    pbar.update()
                    yield {'docno': next(docno_iter), 'doc_vec': vec_batch[vec_idx]}


class RankedLists:
    def __init__(self, num_results : int, num_queries : int):
        self.num_results = num_results
        self.num_queries = num_queries
        self.scores = np.empty((num_queries, 0), dtype='f4')
        self.docids = np.empty((num_queries, 0), dtype='i4')

    def update(self, scores, docids):
        assert self.num_queries == scores.shape[0]
        self.scores = np.concatenate([self.scores, -scores], axis=1)
        self.docids = np.concatenate([self.docids,  docids], axis=1)
        if self.scores.shape[1] > self.num_results:
            partition_idxs = np.argpartition(self.scores, self.num_results, axis=1)[:, :self.num_results]
            self.scores = np.take_along_axis(self.scores, partition_idxs, axis=1)
            self.docids = np.take_along_axis(self.docids, partition_idxs, axis=1)

    def results(self):
        sort_idxs = np.argsort(self.scores, axis=1)
        return (
            np.take_along_axis(-self.scores, sort_idxs, axis=1),
            np.take_along_axis( self.docids, sort_idxs, axis=1),
        )


class TorchRankedLists:
    def __init__(self, num_results, num_queries):
        self.num_results = num_results
        self.num_queries = num_queries
        self.scores = None
        self.docids = None

    def update(self, scores, docids):
        if self.scores is not None:
            self.scores, idxs = torch.cat([self.scores, scores], dim=1).topk(self.num_results, dim=1)
            self.docids = torch.cat([self.docids, docids], dim=1).take_along_dim(idxs, dim=1)
        else:
            self.scores = scores
            self.docids = docids
        #self.scores.append(scores)
        #self.docids.append(docids)

    def results(self):
        return (self.scores.cpu().numpy(), self.docids.cpu().numpy())
        #scores = torch.cat(self.scores, dim=0)
        #docids = torch.cat(self.docids, dim=0)
        #top_scores, top_idxs = scores.topk(self.num_results, dim=1)
        #return (
        #    top_scores.cpu().numpy(),
        #    docids[top_idxs].cpu().numpy()
        #)


class FaissFlat(pt.Indexer):
    def __init__(self, index_path=None, num_results=1000, shard_size=500_000, score_fn='cos', overwrite=False, batch_size=4096, verbose=False, drop_query_vec=True, inmem=False, cuda=False):
        self.index_path = Path(index_path)
        self.num_results = num_results
        self.shard_size = shard_size
        self.score_fn = score_fn
        self.overwrite = overwrite
        self.verbose = verbose
        self.batch_size = batch_size
        self.drop_query_vec = drop_query_vec
        self.cuda = cuda
        self._shards = None
        self._shards = list(self.iter_shards())

    def iter_shards(self):
        import faiss
        if self.cuda:
            res = faiss.StandardGpuResources()
        if self._shards is None:
            for shardid in itertools.count():
                if not (self.index_path/f'{shardid}.faiss').exists():
                    break
                index = faiss.read_index(str(self.index_path/f'{shardid}.faiss'))
                if self.cuda:
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                yield index
        else:
            yield from self._shards

        # if self._shards is None:
        #     shards = []
        #     for shardid in itertools.count():
        #         if not (self.index_path/f'{shardid}.faiss').exists():
        #             break
        #         index = faiss.read_index(str(self.index_path/f'{shardid}.faiss'))
        #         shards.append(index)
        #     self._shards = shards
        #     if shardid == 0:
        #         raise RuntimeError(f'Index not found at {self.index_path}')
        # return iter(self._shards)

    def transform(self, inp):
        columns = set(inp.columns)
        assert all(f in columns for f in ['qid', 'query_vec']), "Faiss expects columns: ['qid', 'query_vec'] when used in a pipeline"
        query_vecs = np.stack(inp['query_vec'])
        if self.score_fn == 'cos':
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        query_vecs = query_vecs.copy()
        res = []
        docnos = DocnoFile(self.index_path/'docnos.npy')
        num_q = query_vecs.shape[0]
        ranked_lists = RankedLists(self.num_results, num_q)
        dids_offset = 0
        it = enumerate(self.iter_shards())
        if self.verbose:
            it = pt.tqdm(it, unit='shard', desc='scanning shards')
        for shardidx, index in it:
            scores, dids = [], []
            QBATCH = 16
            for qidx in range(0, num_q, QBATCH):
                s, d = index.search(query_vecs[qidx:qidx+QBATCH], self.num_results)
                scores.append(s)
                dids.append(d)
            ranked_lists.update(np.concatenate(scores, axis=0), np.concatenate(dids, axis=0)+dids_offset)
            dids_offset += index.ntotal
        result_scores, result_dids = ranked_lists.results()
        unique_dids, unique_inverse = np.unique(result_dids, return_inverse=True)
        unique_docnos = docnos[unique_dids]
        result_docnos = unique_docnos[unique_inverse].reshape(num_q, -1)
        for query, scores, docnos in zip(inp.itertuples(), result_scores, result_docnos):
            if self.drop_query_vec:
                query = query._replace(query_vec=None)
            for score, docno in zip(scores, docnos):
                res.append((*query, docno, score))
        res = pd.DataFrame(res, columns=list(query._fields) + ['docno', 'score'])
        if self.drop_query_vec:
            res = res.drop(columns=['query_vec'])
        res = add_ranks(res)
        return res

    def index(self, inp):
        import faiss
        if isinstance(inp, pd.DataFrame):
            inp = inp.to_dict(orient="records")
        path = Path(self.index_path)
        if path.exists():
            if self.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {self.index_path}. If you want to delete and re-create an existing index, you can pass overwrite=True')
        path.mkdir(parents=True, exist_ok=True)
        docnos = []
        for shardid, shard in enumerate(more_itertools.ichunked(inp, self.shard_size)):
            shard = more_itertools.peekable(shard)
            d = shard.peek()['doc_vec'].shape[0]
            index = faiss.index_factory(d, 'Flat', faiss.METRIC_INNER_PRODUCT)
            shard_it = more_itertools.chunked(shard, self.batch_size)
            if self.verbose:
                shard_it = pt.tqdm(shard_it, total=math.ceil(self.shard_size/self.batch_size), unit='batch', desc=f'building shard {shardid}')
            for batch in shard_it:
                doc_vecs = np.stack(d['doc_vec'] for d in batch)
                if self.score_fn == 'cos':
                    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
                index.add(doc_vecs)
                docnos.extend(d['docno'] for d in batch)
            faiss.write_index(index, str(path/f'{shardid}.faiss'))
        DocnoFile.build(docnos, path/'docnos.npy')


class FaissHnsw(pt.Indexer):
    def __init__(self, index_path, num_links=32, num_results=1000, shard_size=500_000, score_fn='cos', overwrite=False, batch_size=4096, verbose=False, drop_query_vec=True, qbatch=16):
        self.index_path = Path(index_path)
        self.num_links = num_links
        self.num_results = num_results
        self.shard_size = shard_size
        self.score_fn = score_fn
        self.overwrite = overwrite
        self.verbose = verbose
        self.batch_size = batch_size
        self.drop_query_vec = drop_query_vec
        self.qbatch = qbatch
        self._shards = None

    def iter_shards(self, cache=True):
        if not cache:
            return self._iter_shards()
        else:
            if self._shards is None:
                self._shards = list(self._iter_shards())
            return iter(self._shards)

    def _iter_shards(self):
        import faiss
        for shardid in itertools.count():
            if not (self.index_path/f'{shardid}.faiss').exists():
                break
            yield faiss.read_index(str(self.index_path/f'{shardid}.faiss'))
        if shardid == 0:
            raise RuntimeError(f'Index not found at {self.index_path}')

    def transform(self, inp):
        columns = set(inp.columns)
        assert all(f in columns for f in ['qid', 'query_vec']), "Faiss expects columns: ['qid', 'query_vec'] when used in a pipeline"
        query_vecs = np.stack(inp['query_vec'])
        if self.score_fn == 'cos':
            query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
        query_vecs = query_vecs.copy()
        res = []
        docnos = DocnoFile(self.index_path/'docnos.npy')
        num_q = query_vecs.shape[0]
        ranked_lists = RankedLists(self.num_results, num_q)
        dids_offset = 0
        it = enumerate(self.iter_shards())
        if self.verbose:
            it = pt.tqdm(it, unit='shard', desc='scanning shards')
        for shardidx, index in it:
            scores, dids = [], []
            for qidx in range(0, num_q, self.qbatch):
                s, d = index.search(query_vecs[qidx:qidx+self.qbatch], self.num_results)
                scores.append(s)
                dids.append(d)
            ranked_lists.update(np.concatenate(scores, axis=0), np.concatenate(dids, axis=0)+dids_offset)
            dids_offset += index.ntotal
        result_scores, result_dids = ranked_lists.results()
        unique_dids, unique_inverse = np.unique(result_dids, return_inverse=True)
        unique_docnos = docnos[unique_dids]
        result_docnos = unique_docnos[unique_inverse].reshape(num_q, -1)
        for query, scores, docnos in zip(inp.itertuples(), result_scores, result_docnos):
            if self.drop_query_vec:
                query = query._replace(query_vec=None)
            for score, docno in zip(scores, docnos):
                res.append((*query, docno, score))
        res = pd.DataFrame(res, columns=list(query._fields) + ['docno', 'score'])
        if self.drop_query_vec:
            res = res.drop(columns=['query_vec'])
        res = add_ranks(res)
        return res

    def index(self, inp):
        import faiss
        if isinstance(inp, pd.DataFrame):
            inp = inp.to_dict(orient="records")
        path = Path(self.index_path)
        if path.exists():
            if self.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {self.index_path}. If you want to delete and re-create an existing index, you can pass overwrite=True')
        path.mkdir(parents=True, exist_ok=True)
        docnos = []
        if self.shard_size == 0:
            it = [(0, inp)]
        else:
            it = enumerate(more_itertools.ichunked(inp, self.shard_size))
        for shardid, shard in it:
            shard = more_itertools.peekable(shard)
            d = shard.peek()['doc_vec'].shape[0]
            index = faiss.index_factory(d, f'HNSW{self.num_links}', faiss.METRIC_INNER_PRODUCT)
            shard_it = more_itertools.chunked(shard, self.batch_size)
            if self.verbose:
                shard_it = pt.tqdm(shard_it, total=math.ceil(self.shard_size/self.batch_size), unit='batch', desc=f'building shard {shardid}')
            for batch in shard_it:
                doc_vecs = np.stack(d['doc_vec'] for d in batch)
                if self.score_fn == 'cos':
                    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
                index.add(doc_vecs)
                docnos.extend(d['docno'] for d in batch)
            faiss.write_index(index, str(path/f'{shardid}.faiss'))
        DocnoFile.build(docnos, path/'docnos.npy')













class TorchIndex(NumpyIndex):
    def __init__(self,
                 index_path=None,
                 num_results=1000,
                 score_fn='dot',
                 overwrite=False,
                 batch_size=4096,
                 verbose=False,
                 dtype='f4',
                 drop_query_vec=True,
                 half=False,
                 idx_mem=500000000):
        super().__init__(index_path, num_results, score_fn, overwrite, batch_size, verbose, dtype, drop_query_vec, inmem=False)
        self.half = half
        self.idx_mem = idx_mem
        self._meta = None
        self._cpu_data = None
        self._cuda_data = None
        self._docnos = None
        self._did_start = None
        self._cuda_slice = None

    def docnos_and_data(self):
        if self._meta is None:
            with (self.index_path/'meta.json').open('rt') as f_meta:
                self._meta = json.load(f_meta)
        meta = self._meta
        if self._cpu_data is None:
            SType, TType, CTType, SIZE = {
                'f4': (torch.FloatStorage, torch.FloatTensor, torch.cuda.FloatTensor, 4),
                'f2': (torch.HalfStorage,  torch.HalfTensor,  torch.cuda.HalfTensor,  2),
            }[self.dtype]
            self._cpu_data = TType(SType.from_file(str(self.index_path/'index.npy'), size=meta['count'] * meta['vec_size'])).reshape(meta['count'], meta['vec_size'])
            doc_batch_size = self.idx_mem//SIZE//meta['vec_size']
            if self.half:
                self._cuda_data = torch.cuda.HalfTensor(size=(doc_batch_size, meta['vec_size']), device='cuda')
            else:
                self._cuda_data = CTType(size=(doc_batch_size, meta['vec_size']), device='cuda')
            if self.verbose and doc_batch_size > self.num_results:
                _logger.info(f'TorchIndex using document batches of size {doc_batch_size}')
            if self.num_results >= doc_batch_size:
                _logger.warn(f'TorchIndex using document batches of size {doc_batch_size} but this is less than num_rsults={self.num_results}. Consider using a larger idx_mem (currently {self.idx_mem})')
            self._docnos = DocnoFile(self.index_path/'docnos.npy')

    def transform(self, inp):
        columns = set(inp.columns)
        assert all(f in columns for f in ['qid', 'query_vec']), "TrochIndex expects columns ['qid', 'query_vec'] when used in a pipeline"
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = torch.from_numpy(query_vecs).cuda() # TODO: can this go directly to CUDA? device='cuda' doesn't work
        if self.half:
            query_vecs = query_vecs.half()

        self.docnos_and_data()

        step = self._cuda_data.shape[0]
        it = range(0, self._cpu_data.shape[0], step)
        if self.verbose:
            it = pt.tqdm(it, desc='TorchIndex scoring', unit='docbatch')

        ranked_lists = RankedLists(self.num_results, query_vecs.shape[0])
        for start_idx in it:
            end_idx = start_idx + step
            batch = self._cpu_data[start_idx:end_idx]
            bsize = batch.shape[0]
            if self.half:
                self._cuda_data[:bsize] = batch.half()
            else:
                self._cuda_data[:bsize] = batch

            scores = query_vecs @ self._cuda_data[:bsize].T
            if self.half:
                scores = scores.float()
            if scores.shape[0] > self.num_results:
                scores, dids = torch.topk(scores, k=self.num_results, dim=1)
            else:
                scores, dids = torch.sort(scores, dim=1, descending=False)
            scores = scores.cpu().float().numpy()

            dids = (dids + start_idx).cpu().numpy()
            ranked_lists.update(scores, dids)

        result_scores, result_dids = ranked_lists.results()
        unique_dids, unique_inverse = np.unique(result_dids, return_inverse=True)
        unique_docnos = self._docnos[unique_dids]
        result_docnos = unique_docnos[unique_inverse].reshape(query_vecs.shape[0], -1)
        res = []
        for query, scores, docnos in zip(inp.itertuples(index=False), result_scores, result_docnos):
            if self.drop_query_vec:
                query = query._replace(query_vec=None)
            for score, docno in zip(scores, docnos):
                res.append((*query, docno, score))
        res = pd.DataFrame(res, columns=list(query._fields) + ['docno', 'score'])
        if self.drop_query_vec:
            res = res.drop(columns=['query_vec'])
        res = res[~res.score.isna()]
        res = add_ranks(res)
        return res
