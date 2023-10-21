import pandas as pd
import os
import pyterrier as pt
import itertools
import numpy as np
import ir_datasets
import pyterrier_dr
from . import FlexIndex

logger = ir_datasets.log.easy()


class VoyagerRetriever(pt.Indexer):
    def __init__(self, flex_index, voyager_index, query_ef=None, qbatch=64):
        self.flex_index = flex_index
        self.voyager_index = voyager_index
        self.query_ef = query_ef
        self.qbatch = qbatch

    def transform(self, inp):
        inp = inp.reset_index(drop=True)
        assert all(f in inp.columns for f in ['qid', 'query_vec'])
        docnos, config = self.flex_index.payload(return_dvecs=False)
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = query_vecs.copy()
        idxs = []
        res = {'docid': [], 'score': [], 'rank': []}
        num_q = query_vecs.shape[0]
        QBATCH = self.qbatch
        it = range(0, num_q, QBATCH)
        if self.flex_index.verbose:
            it = logger.pbar(it, unit='qbatch')
        for qidx in it:
            qvec_batch = query_vecs[qidx:qidx+QBATCH]
            neighbor_ids, distances = self.voyager_index.query(qvec_batch, self.flex_index.num_results, self.query_ef)
            for i, (s, d) in enumerate(zip(distances, neighbor_ids)):
                mask = d != -1
                d = d[mask]
                s = s[mask]
                res['docid'].append(d)
                res['score'].append(-s)
                res['rank'].append(np.arange(d.shape[0]))
                idxs.extend(itertools.repeat(qidx+i, d.shape[0]))
        res = {k: np.concatenate(v) for k, v in res.items()}
        res['docno'] = docnos.fwd[res['docid']]
        for col in inp.columns:
            if col != 'query_vec':
                res[col] = inp[col][idxs].values
        return pd.DataFrame(res)


def _voyager_retriever(self, neighbours=12, ef_construction=200, random_seed=1, storage_data_type='float32', query_ef=10):
    pyterrier_dr.util.assert_voyager()
    import voyager
    meta, = self.payload(return_dvecs=False, return_docnos=False)

    space = {
        pyterrier_dr.SimFn.dot: voyager.Space.InnerProduct,
        pyterrier_dr.SimFn.cos: voyager.Space.Cosine,
    }[self.sim_fn]
    storage_data_type = {
        'float32': voyager.StorageDataType.Float32,
        'float8': voyager.StorageDataType.Float8,
        'e4m3': voyager.StorageDataType.E4M3,
    }[storage_data_type.lower()]

    key = ('voyager', space, neighbours, ef_construction, random_seed, storage_data_type)
    index_name = f'voyager-{space.name}-{neighbours}-{ef_construction}-{random_seed}-{storage_data_type.name}.voyager'
    if key not in self._cache:
        if not os.path.exists(self.index_path/index_name):
            BATCH_SIZE = 10_000
            dvecs, meta = self.payload(return_docnos=False)
            index = voyager.Index(space, meta['vec_size'], neighbours, ef_construction, random_seed, meta['doc_count'], storage_data_type)
            print(index.ef)
            it = range(0, meta['doc_count'], BATCH_SIZE)
            if self.verbose:
                it = logger.pbar(it, desc='building index', unit='dbatch')
            for idx in it:
                index.add_items(dvecs[idx:idx+BATCH_SIZE])
            with logger.duration('saving index'):
                index.save(str(self.index_path/index_name))
            self._cache[key] = index
        else:
            with logger.duration('reading index'):
                self._cache[key] = voyager.Index.load(str(self.index_path/index_name))
    return VoyagerRetriever(self, self._cache[key], query_ef=query_ef)
FlexIndex.voyager_retriever = _voyager_retriever
