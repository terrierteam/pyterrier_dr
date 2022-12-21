import pandas as pd
import math
import struct
import os
import pyterrier as pt
import itertools
import numpy as np
import tempfile
import ir_datasets
from . import FlexIndex

logger = ir_datasets.log.easy()


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


def _faiss_flat_retriever(self, gpu=False):
        import faiss
        if 'faiss_flat' not in self._cache:
            meta, = self.payload(return_dvecs=False, return_docnos=False)
            with tempfile.TemporaryDirectory() as tmp, logger.duration('reading faiss flat'):
                DUMMY = b'\x00\x00\x10\x00\x00\x00\x00\x00'
                #       [ type ]                                                                             [ train ]  [ metric (dot)     ]
                header = b'IxFI'  + struct.pack('<IQ', meta['vec_size'], meta['doc_count']) + DUMMY + DUMMY  + b'\x00' + b'\x00\x00\x00\x00' + struct.pack('<Q', meta['doc_count'] * meta['vec_size'])
                with open(os.path.join(tmp, 'tmp'), 'wb') as fout:
                    fout.write(b' ' + header)
                reader = faiss.BufferedIOReader(faiss.FileIOReader(os.path.join(tmp, 'tmp')))
                reader.read_bytes(1)
                reader.reader = faiss.FileIOReader(str(self.index_path/'vecs.f4'))
                self._cache['faiss_flat'] = faiss.read_index(reader)
        if gpu:
            if 'faiss_flat_gpu' not in self._cache:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                self._cache['faiss_flat_gpu'] = faiss.index_cpu_to_all_gpus(self._faiss_flat, co=co)
            return FaissRetriever(self, self._cache['faiss_flat_gpu'])
        return FaissRetriever(self, self._cache['faiss_flat'])
FlexIndex.faiss_flat_retriever = _faiss_flat_retriever


def _faiss_hnsw_retriever(self, neighbours=32, ef_construction=40, ef_search=16, cache=True):
        import faiss
        meta, = self.payload(return_dvecs=False, return_docnos=False)

        key = ('faiss_hnsw', neighbours, ef_construction)
        index_name = f'hnsw_n-{neighbours}_ef-{ef_construction}.faiss'
        if key not in self._cache:
            dvecs, meta = self.payload(return_docnos=False)
            if not os.path.exists(self.index_path/index_name):
                idx = faiss.IndexHNSWFlat(meta['vec_size'], neighbours, faiss.METRIC_INNER_PRODUCT)
                for start_idx in logger.pbar(range(0, dvecs.shape[0], 4096), desc='indexing', unit='batch'):
                    idx.add(np.array(dvecs[start_idx:start_idx+4096]))
                idx.storage = faiss.IndexFlatIP(meta['vec_size']) # clear storage ; we can use faiss_flat here instead so we don't keep an extra copy
                if cache:
                    with logger.duration('caching index'):
                        faiss.write_index(idx, str(self.index_path/index_name))
                self._cache[key] = idx
            else:
                with logger.duration('reading hnsw table'):
                    self._cache[key] = faiss.read_index(str(self.index_path/index_name))
            self._cache[key].storage = self.faiss_flat_retriever().faiss_index
        return FaissRetriever(self, self._cache[key], ef_search=ef_search)
FlexIndex.faiss_hnsw_retriever = _faiss_hnsw_retriever


def _faiss_ivf_retriever(self, train_sample=None, n_list=None, cache=True, n_probe=1):
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

        key = ('faiss_ivf', train_sample, n_list)
        index_name = f'ivf_train-{train_sample}_nlist-{n_list}.faiss'
        if key not in self._cache:
            dvecs, meta = self.payload(return_docnos=False)
            if not os.path.exists(self.index_path/index_name):
                quantizer = faiss.IndexFlatIP(meta['vec_size'])
                quantizer = faiss.index_cpu_to_all_gpus(quantizer)
                idx = faiss.IndexIVFFlat(quantizer, meta['vec_size'], n_list, faiss.METRIC_INNER_PRODUCT)
                with logger.duration(f'loading {train_sample} train samples'):
                    train = self._sample_train(train_sample)
                with logger.duration(f'training ivf with {n_list} posting lists'):
                    idx.train(train)
                for start_idx in logger.pbar(range(0, dvecs.shape[0], 4096), desc='indexing', unit='batch'):
                    idx.add(np.array(dvecs[start_idx:start_idx+4096]))
                if cache:
                    with logger.duration('caching index'):
                        idx.quantizer = faiss.index_gpu_to_cpu(idx.quantizer)
                        faiss.write_index(idx, str(self.index_path/index_name))
                self._cache[key] = idx
            else:
                with logger.duration('reading index'):
                    self._cache[key] = faiss.read_index(str(self.index_path/index_name))
        return FaissRetriever(self, self._cache[key], n_probe=n_probe)
FlexIndex.faiss_ivf_retriever = _faiss_ivf_retriever
