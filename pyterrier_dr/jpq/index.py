from pathlib import Path
import shutil
import pyterrier as pt
from typing import List
import numpy as np
import json
from npids import Lookup
from pyterrier_dr.flex.core import IndexingMode
from .retriever import JPQRetrieverFlat, JPQRetrieverPrune

class JPQIndex(pt.Artifact):

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'jpq'

    def __init__(self, path: str):
        super().__init__(path)
        self._meta = None
        self._dvecs = None
        self._codes = None
        self._docnos = None
        self.index_path = path
    
    def payload(self, return_dvecs=True, return_docnos=True, return_codes=True):
        if self._meta is None:
            with open(self.index_path/'pt_meta.json', 'rt') as f_meta:
                meta = json.load(f_meta)
                assert meta.get('type') == 'dense_index' and meta['format'] == 'jpq'
                self._meta = meta
        res = [self._meta]
        if return_dvecs:
            if self._dvecs is None:
                self._dvecs = np.memmap(self.index_path/'subvecs.f4', mode='r', dtype=np.float32, shape=(self._meta['doc_count'], self._meta['code_size'], self._meta['subvec_size']))
            res.insert(0, self._dvecs)
        if return_codes:
            if self._codes is None:
                self._codes = np.memmap(self.index_path/'codes.f4', mode='r', dtype=np.uint8, shape=(self._meta['doc_count'], self._meta['code_size']))
            res.insert(0, self._codes)
        if return_docnos:
            if self._docnos is None:
                self._docnos = Lookup(self.index_path/'docnos.npids')
            res.insert(0, self._docnos)
        return res
    
    @staticmethod
    def build(path : str, 
              docnos : List[str], 
              codes : np.ndarray, # N x M
              embs : np.ndarray, # sub-item embeddings: M x 2^nbits x dsub (aka the centroids)
              mode = IndexingMode.create) -> "JPQIndex":
        index = JPQIndex(path)
        index.docnos = docnos
        index.codes = codes
        index.embs = embs
        path = Path(path)
        if path.exists():
            if mode == IndexingMode.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {path}. If you want to delete and re-create an existing index, you can pass mode="overwrite"')
        path.mkdir(parents=True, exist_ok=True)
        count = len(docnos)
        Lookup.build(docnos, path/'docnos.npids')

        with open(path/'subvecs.f4', 'wb') as fout:
            for split in range(embs.shape[0]):
                for code in range(embs.shape[1]):
                    vec = embs[split, code, :] # dim: dsub
                    vec = vec.astype(np.float32)
                    fout.write(vec.tobytes())
        with open(path/'pt_meta.json', 'wt') as f_meta:
            json.dump({
                "type": JPQIndex.ARTIFACT_TYPE,
                "format": JPQIndex.ARTIFACT_FORMAT,
                "code_size": embs.shape[1],
                "subvec_size" : embs.shape[2],
                "doc_count": count
            }, f_meta)
        return JPQIndex(str(path))

    def retriever_flat(self, topk: int = 1000) -> "JPQRetrieverFlat":
        _, subembs, codes, docnos = self.payload(return_dvecs=True, return_docnos=True, return_codes=True)
        return JPQRetrieverFlat(docnos, codes, subembs, topk=1000, name="JPQ-Full")
    
    def retriever_prune(self, topk: int = 1000, ub_inflation : float =1.) -> "JPQRetrieverPrune":
        _, subembs, codes, docnos = self.payload(return_dvecs=True, return_docnos=True, return_codes=True)
        return JPQRetrieverFlat(docnos, codes, subembs, topk=1000, name="JPQ-Full", ub_inflation=ub_inflation)
        