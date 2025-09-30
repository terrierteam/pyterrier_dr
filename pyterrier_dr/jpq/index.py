import pyterrier as pt
from typing import List
import numpy as np
import json
from npids import Lookup
from .retriever import JPQRetrieverFlat

class JPQIndex(pt.Artifact):

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'jpq'

    def __init__(self, path: str):
        super().__init__(path)
    
    def payload(self, return_dvecs=True, return_docnos=True, return_codes=True):
        if self._meta is None:
            with open(self.index_path/'pt_meta.json', 'rt') as f_meta:
                meta = json.load(f_meta)
                assert meta.get('type') == 'dense_index' and meta['format'] == 'jpq'
                self._meta = meta
        res = [self._meta]
        if return_dvecs:
            if self._dvecs is None:
                self._dvecs = np.memmap(self.index_path/'vecs.f4', mode='r', dtype=np.float32, shape=(self._meta['doc_count'], self._meta['vec_size']))
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
    def build(path : str, docnos : List[str], codes, embs) -> "JPQIndex":
        index = JPQIndex(path)
        index.docnos = docnos
        index.codes = codes
        index.embs = embs

    def retriever_flat(self, topk: int = 1000) -> "JPQRetrieverFlat":
        _, embs, codes, docnos = self.payload(return_dvecs=True, return_docnos=True, return_codes=True)
        return JPQRetrieverFlat(docnos,codes, topk=1000, name="JPQ-Full")
    
