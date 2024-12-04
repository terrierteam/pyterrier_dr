import os
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import pyterrier_dr
from pyterrier_dr import SimFn
from . import FlexIndex


class FlatNavRetriever(pt.Transformer):
    def __init__(self, flex_index, flatnav_index, *, threads=16, ef_search=100, num_initializations=100, num_results=1000, qbatch=64, drop_query_vec=False, verbose=False):
        self.flex_index = flex_index
        self.flatnav_index = flatnav_index
        self.threads = threads
        self.ef_search = ef_search
        self.num_results = num_results
        self.qbatch = qbatch
        self.drop_query_vec = drop_query_vec
        self.num_initializations = num_initializations
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        pta.validate.query_frame(inp, extra_columns=['query_vec'])
        inp = inp.reset_index(drop=True)
        docnos, config = self.flex_index.payload(return_dvecs=False)
        query_vecs = np.stack(inp['query_vec'])
        query_vecs = query_vecs.copy()
        num_q = query_vecs.shape[0]
        QBATCH = self.qbatch
        it = range(0, num_q, QBATCH)
        if self.flex_index.verbose:
            it = pt.tqdm(it, unit='qbatch')
        self.flatnav_index.set_num_threads(self.threads)

        result = pta.DataFrameBuilder(['docno', 'docid', 'score', 'rank'])
        for qidx in it:
            scores, dids = self.flatnav_index.search(
                queries=query_vecs[qidx:qidx+QBATCH],
                ef_search=self.ef_search,
                K=min(self.num_results, len(self.flex_index)),
                num_initializations=self.num_initializations,
            )
            scores = -scores # distances -> scores
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


def _flatnav_retriever(self,
    k: int = 32,
    *,
    ef_search: int = 100,
    num_initializations: int = 100,
    ef_construction: int = 100,
    threads: int = 16,
    num_results: int = 1000,
    cache: bool = True,
    qbatch: int = 64,
    drop_query_vec: bool = False,
    verbose: bool = False,
) -> pt.Transformer:
    """Returns a retriever that searchers over a flatnav index.

    Args:
        k (int): the maximum number of edges per document in the index
        ef_search (int): the size of the list during searches. Higher values are slower but more accurate.
        num_initializations (int): the number of random initializations to use during search.
        ef_construction (int): the size of the list during graph construction. Higher values are slower but more accurate.
        threads (int): the number of threads to use
        num_results (int): the number of results to return per query
        cache (bool): whether to cache the index to disk
        qbatch (int): the number of queries to search at once
        drop_query_vec (bool): whether to drop the query_vec column after retrieval
        verbose (bool): whether to show progress bars

    .. versionadded:: 0.4.0
    .. versionchanged:: 0.4.1 fixed bug with `num_initializations`

    .. note::
        This transformer requires the ``flatnav`` package to be installed. Instructions are available
        in the `flatnav repository <https://github.com/BlaiseMuhirwa/flatnav>`__.

    .. cite:: arxiv:2412.01940
        :citation: Munyampirwa et al. Down with the Hierarchy: The 'H' in HNSW Stands for "Hubs". arXiv 2024.
        :link: https://arxiv.org/abs/2412.01940
    """
    pyterrier_dr.util.assert_flatnav()
    import flatnav

    key = ('flatnav', k, ef_construction)
    index_name = f'{k}_ef-{ef_construction}-{str(self.sim_fn)}.flatnav'
    if key not in self._cache:
        dvecs, meta = self.payload(return_docnos=False)
        if not os.path.exists(self.index_path/index_name):
            distance_type = {
                SimFn.dot: 'angular',
            }[self.sim_fn]
            idx = flatnav.index.create(
                distance_type=distance_type,
                index_data_type=flatnav.data_type.DataType.float32,
                dim=dvecs.shape[1],
                dataset_size=dvecs.shape[0],
                max_edges_per_node=k,
                verbose=True,
                collect_stats=True,
            )
            idx.set_num_threads(threads)
            idx.add(data=np.array(dvecs), ef_construction=ef_construction)
            # for start_idx in pt.tqdm(range(0, dvecs.shape[0], 4096), desc='indexing flatnav', unit='batch'):
            #     idx.add(data=np.array(dvecs[start_idx:start_idx+4096]), ef_construction=ef_construction)
            if cache:
                idx.save(str(self.index_path/index_name))
            self._cache[key] = idx
        else:
            self._cache[key] = flatnav.index.IndexIPFloat.load_index(str(self.index_path/index_name))
            self._cache[key].set_data_type(flatnav.data_type.DataType.float32)
    return FlatNavRetriever(self, self._cache[key], threads=threads, ef_search=ef_search, num_initializations=num_initializations, num_results=num_results, qbatch=qbatch, drop_query_vec=drop_query_vec, verbose=verbose)
FlexIndex.flatnav_retriever = _flatnav_retriever
