import shutil
import itertools
import json
from pathlib import Path
from warnings import warn
import numpy as np
import more_itertools
import pandas as pd
import pyterrier as pt
from npids import Lookup
from enum import Enum
from .. import SimFn
import pyterrier_alpha as pta


class IndexingMode(Enum):
    create = "create"
    overwrite = "overwrite"
    # append???


class FlexIndex(pta.Artifact, pt.Indexer):
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
        with pta.validate.any(inp) as v:
            v.query_frame(extra_columns=['query_vec'], mode='np_retriever')

        if v.mode == 'np_retriever':
            warn("performing exhaustive search with FlexIndex.np_retriever -- note that other FlexIndex retrievers may be faster")
            return self.np_retriever()(inp)

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

    def _load_docids(self, inp):
        with pta.validate.any(inp) as v:
            v.columns(includes=['docid'], mode='docid')
            v.columns(includes=['docno'], mode='docno')
        if v.mode == 'docid':
            return inp['docid'].values
        docnos, config = self.payload(return_dvecs=False)
        return docnos.inv[inp['docno'].values] # look up docids from docnos

    def built(self):
        return self.index_path.exists()


def _load_dvecs(flex_index, inp):
    dvecs, config = flex_index.payload(return_docnos=False)
    return dvecs[flex_index._load_docids(inp)]
