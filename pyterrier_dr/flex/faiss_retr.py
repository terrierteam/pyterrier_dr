import json
import math
import struct
import os
import pyterrier as pt
import numpy as np
import tempfile
import ir_datasets
import pyterrier_dr
import pyterrier_alpha as pta
from . import FlexIndex

logger = ir_datasets.log.easy()


class FaissRetriever(pt.Indexer):
    def __init__(self, flex_index, faiss_index, n_probe=None, ef_search=None, search_bounded_queue=None, qbatch=64, drop_query_vec=False):
        self.flex_index = flex_index
        self.faiss_index = faiss_index
        self.n_probe = n_probe
        self.ef_search = ef_search
        self.search_bounded_queue = search_bounded_queue
        self.qbatch = qbatch
        self.drop_query_vec = drop_query_vec

    def transform(self, inp):
        pta.validate.query_frame(inp, extra_columns=['query_vec'])
        inp = inp.reset_index(drop=True)
        docnos, config = self.flex_index.payload(return_dvecs=False)
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = query_vecs.copy()
        num_q = query_vecs.shape[0]
        QBATCH = self.qbatch
        if self.n_probe is not None:
            self.faiss_index.nprobe = self.n_probe
        if self.ef_search is not None:
            self.faiss_index.hnsw.efSearch = self.ef_search
        if self.search_bounded_queue is not None:
            self.faiss_index.hnsw.search_bounded_queue = self.search_bounded_queue
        it = range(0, num_q, QBATCH)
        if self.flex_index.verbose:
            it = pt.tqdm(it, unit='qbatch')

        result = pta.DataFrameBuilder(['docno', 'docid', 'score', 'rank'])
        for qidx in it:
            scores, dids = self.faiss_index.search(query_vecs[qidx:qidx+QBATCH], self.flex_index.num_results)
            for s, d in zip(scores, dids):
                mask = d != -1
                d = d[mask]
                s = s[mask]
                result.extend({
                    'docno': docnos.fwd[d],
                    'docid': d,
                    'score': s,
                    'rank': np.arange(d.shape[0]),
                })

        if self.drop_query_vec:
            inp = inp.drop(columns='query_vec')
        return result.to_df(inp)


def _faiss_flat_retriever(self, gpu=False, qbatch=64, drop_query_vec=False):
    pyterrier_dr.util.assert_faiss()
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
        return FaissRetriever(self, self._cache['faiss_flat_gpu'], drop_query_vec=drop_query_vec)
    return FaissRetriever(self, self._cache['faiss_flat'], qbatch=qbatch, drop_query_vec=drop_query_vec)
FlexIndex.faiss_flat_retriever = _faiss_flat_retriever


def _faiss_hnsw_retriever(self, neighbours=32, ef_construction=40, ef_search=16, cache=True, search_bounded_queue=True, qbatch=64, drop_query_vec=False):
    pyterrier_dr.util.assert_faiss()
    import faiss
    meta, = self.payload(return_dvecs=False, return_docnos=False)

    key = ('faiss_hnsw', neighbours, ef_construction)
    index_name = f'hnsw_n-{neighbours}_ef-{ef_construction}.faiss'
    if key not in self._cache:
        dvecs, meta = self.payload(return_docnos=False)
        if not os.path.exists(self.index_path/index_name):
            idx = faiss.IndexHNSWFlat(meta['vec_size'], neighbours, faiss.METRIC_INNER_PRODUCT)
            for start_idx in pt.tqdm(range(0, dvecs.shape[0], 4096), desc='indexing', unit='batch'):
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
    return FaissRetriever(self, self._cache[key], ef_search=ef_search, search_bounded_queue=search_bounded_queue, qbatch=qbatch, drop_query_vec=drop_query_vec)
FlexIndex.faiss_hnsw_retriever = _faiss_hnsw_retriever


def _faiss_hnsw_graph(self, neighbours=32, ef_construction=40):
    key = ('faiss_hnsw', neighbours//2, ef_construction)
    graph_name = f'hnsw_n-{neighbours}_ef-{ef_construction}.graph'
    if key not in self._cache:
        if not (self.index_path/graph_name/'pt_meta.json').exists():
            retr = self.faiss_hnsw_retriever(neighbours=neighbours//2, ef_construction=ef_construction)
            _build_hnsw_graph(retr.faiss_index.hnsw, self.index_path/graph_name)
        from pyterrier_adaptive import CorpusGraph
        self._cache[key] = CorpusGraph.load(self.index_path/graph_name)
    return self._cache[key]
FlexIndex.faiss_hnsw_graph = _faiss_hnsw_graph


def _build_hnsw_graph(hnsw, out_dir):
    lvl_0_size = hnsw.nb_neighbors(0)
    num_docs = hnsw.offsets.size() - 1
    scores = np.zeros(lvl_0_size, dtype=np.float16)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir/'edges.u32.np'
    weights_path = out_dir/'weights.f16.np'
    with ir_datasets.util.finialized_file(str(edges_path), 'wb') as fe, \
         ir_datasets.util.finialized_file(str(weights_path), 'wb') as fw:
        for did in pt.tqdm(range(num_docs), unit='doc', smoothing=1):
            start = hnsw.offsets.at(did)
            dids = [hnsw.neighbors.at(i) for i in range(start, start+lvl_0_size)]
            dids = [(d if d != -1 else did) for d in dids] # replace with self if missing value
            fe.write(np.array(dids, dtype=np.uint32).tobytes())
            fw.write(scores.tobytes())
    (out_dir/'docnos.npids').symlink_to('../docnos.npids')
    with (out_dir/'pt_meta.json').open('wt') as fout:
        json.dump({
            'type': 'corpus_graph',
            'format': 'np_topk',
            'doc_count': num_docs,
            'k': lvl_0_size,
        }, fout)

def _sample_train(index, count=None):
    dvecs, meta = index.payload(return_docnos=False)
    count = min(count or 10_000, dvecs.shape[0])
    idxs = np.random.RandomState(0).choice(dvecs.shape[0], size=count, replace=False)
    return dvecs[idxs]

def _faiss_ivf_retriever(self, train_sample=None, n_list=None, cache=True, n_probe=1, drop_query_vec=False):
    pyterrier_dr.util.assert_faiss()
    import faiss
    meta, = self.payload(return_dvecs=False, return_docnos=False)

    if n_list is None:
        if train_sample is None:
            n_list = math.ceil(math.sqrt(meta['doc_count']))
            # we'll shift it to the nearest power of 2
            n_list = int(1 << math.ceil(math.log2(n_list)))
        else:
            n_list = math.floor(train_sample / 39)
        n_list = max(n_list, 4)

    if train_sample is None:
        train_sample = n_list * 39
    elif 0 < train_sample < 1:
        train_sample = math.ceil(train_sample * meta['doc_count'])

    key = ('faiss_ivf', n_list, train_sample)
    index_name = f'ivf_nlist-{n_list}_train-{train_sample}.faiss'
    if key not in self._cache:
        dvecs, meta = self.payload(return_docnos=False)
        if not os.path.exists(self.index_path/index_name):
            quantizer = faiss.IndexFlatIP(meta['vec_size'])
            if pyterrier_dr.util.infer_device().type == 'cuda':
                quantizer = faiss.index_cpu_to_all_gpus(quantizer)
            idx = faiss.IndexIVFFlat(quantizer, meta['vec_size'], n_list, faiss.METRIC_INNER_PRODUCT)
            with logger.duration(f'loading {train_sample} train samples'):
                train = _sample_train(self, train_sample)
            with logger.duration(f'training ivf with {n_list} posting lists'):
                idx.train(train)
            for start_idx in pt.tqdm(range(0, dvecs.shape[0], 4096), desc='indexing', unit='batch'):
                idx.add(np.array(dvecs[start_idx:start_idx+4096]))
            if cache:
                with logger.duration('caching index'):
                    if pyterrier_dr.util.infer_device().type == 'cuda':
                        idx.quantizer = faiss.index_gpu_to_cpu(idx.quantizer)
                    faiss.write_index(idx, str(self.index_path/index_name))
            self._cache[key] = idx
        else:
            with logger.duration('reading index'):
                self._cache[key] = faiss.read_index(str(self.index_path/index_name))
    return FaissRetriever(self, self._cache[key], n_probe=n_probe, drop_query_vec=drop_query_vec)
FlexIndex.faiss_ivf_retriever = _faiss_ivf_retriever
