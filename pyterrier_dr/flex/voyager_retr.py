import os
import pyterrier as pt
import numpy as np
import ir_datasets
import pyterrier_alpha as pta
import pyterrier_dr
from . import FlexIndex

logger = ir_datasets.log.easy()


class VoyagerRetriever(pt.Indexer):
    def __init__(self, flex_index, voyager_index, query_ef=None, num_results=1000, qbatch=64, drop_query_vec=False):
        self.flex_index = flex_index
        self.voyager_index = voyager_index
        self.query_ef = query_ef
        self.num_results = num_results
        self.qbatch = qbatch
        self.drop_query_vec = drop_query_vec

    def transform(self, inp):
        pta.validate.query_frame(inp, extra_columns=['query_vec'])
        inp = inp.reset_index(drop=True)
        docnos, config = self.flex_index.payload(return_dvecs=False)
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = query_vecs.copy()
        
        result = pta.DataFrameBuilder(['docno', 'docid', 'score', 'rank'])
        num_q = query_vecs.shape[0]
        QBATCH = self.qbatch
        it = range(0, num_q, QBATCH)
        if self.flex_index.verbose:
            it = pt.tqdm(it, unit='qbatch')
        for qidx in it:
            qvec_batch = query_vecs[qidx:qidx+QBATCH]
            neighbor_ids, distances = self.voyager_index.query(qvec_batch, self.num_results, self.query_ef)
            for s, d in zip(distances, neighbor_ids):
                mask = d != -1
                d = d[mask]
                s = s[mask]
                result.extend({
                    'docno': docnos.fwd[d],
                    'docid': d,
                    'score': -s,
                    'rank': np.arange(d.shape[0]),
                })

        if self.drop_query_vec:
            inp = inp.drop(columns='query_vec')
        return result.to_df(inp)


def _voyager_retriever(self,
    neighbours: int = 12,
    *,
    num_results: int = 1000,
    ef_construction: int = 200,
    random_seed: int = 1,
    storage_data_type: str = 'float32',
    query_ef: int = 10,
    drop_query_vec: bool = False
) -> pt.Transformer:
    """Returns a retriever that uses HNSW to search over a Voyager index.

    Args:
        neighbours (int, optional): Number of neighbours to search. Defaults to 12.
        num_results (int, optional): Number of results to return per query. Defaults to 1000.
        ef_construction (int, optional): Expansion factor for graph construction. Defaults to 200.
        random_seed (int, optional): Random seed. Defaults to 1.
        storage_data_type (str, optional): Storage data type. One of 'float32', 'float8', 'e4m3'. Defaults to 'float32'.
        query_ef (int, optional): Expansion factor during querying. Defaults to 10.
        drop_query_vec (bool, optional): Drop the query vector from the output. Defaults to False.

    Returns:
        :class:`~pyterrier.Transformer`: A retriever that uses HNSW to search over a Voyager index.

    .. note::
        This method requires the ``voyager`` package. Install it via ``pip install voyager``.
    """
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
                it = pt.tqdm(it, desc='building index', unit='dbatch')
            for idx in it:
                index.add_items(dvecs[idx:idx+BATCH_SIZE])
            with logger.duration('saving index'):
                index.save(str(self.index_path/index_name))
            self._cache[key] = index
        else:
            with logger.duration('reading index'):
                self._cache[key] = voyager.Index.load(str(self.index_path/index_name))
    return VoyagerRetriever(self, self._cache[key], query_ef=query_ef, drop_query_vec=drop_query_vec)
FlexIndex.voyager_retriever = _voyager_retriever
