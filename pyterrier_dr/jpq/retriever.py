from typing import List, Optional
import pandas as pd
import numpy as np
import pyterrier as pt
import torch
from .utils import timer
from pyterrier import tqdm
import faiss

# class JPQRetrieverFlatEncoder(pt.Transformer):
#     """Subset-mode retriever (Flat-IP reconstructed from codes)."""
#     def __init__(self, biencoder: BiEncoder, docnos: List[str],
#                  codes: np.array,
#                  topk: int = 1000,
#                  name: Optional[str] = None):
#         super().__init__()
#         self.biencoder = biencoder
#         self.docnos = np.array(docnos)
#         self.codes = torch.from_numpy(codes).long()
#         self.topk = topk
#         self._index = None
#         self._name = name or "JPQRetriever"
    
#     def __str__(self): return self._name

#     def _ensure(self, bs: int = 20000):
#         if self._index is not None: return
#         dim = self.biencoder.passage.sub_embeddings[0].embedding_dim * self.biencoder.passage.M
#         index = faiss.IndexFlatIP(dim)
#         pm = self.biencoder.passage.to("cpu").eval()
#         with torch.no_grad():
#             for i in tqdm(range(0, self.codes.size(0), bs), desc=f"{self._name} / build flat", leave=False):
#                 chunk = self.codes[i:i+bs]
#                 embs = pm(chunk)
#                 embs = (embs / (embs.norm(dim=1, keepdim=True) + 1e-12)).detach().cpu().numpy().astype('float32')
#                 index.add(embs)
#         self._index = index
    
#     def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
#         pt.validate.query_frame(topics)
#         qids = topics['qid'].astype(str).tolist()
#         texts = topics['query'].astype(str).tolist()
#         with timer(f"{self._name} / encode_queries"):
#             Q_t = self.biencoder.query.to("cpu").encode_texts(texts, batch_size=64)
#             Q = Q_t.detach().cpu().numpy().astype('float32')
#             Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
#         self._ensure()
#         with timer(f"{self._name} / flat search"):
#             D, I = self._index.search(Q, min(self.topk, len(self.docnos)))
#         rows = []
#         for i, qid in enumerate(qids):
#             for r, (did, s) in enumerate(zip(I[i], D[i]), start=1):
#                 rows.append((qid, self.docnos[int(did)], float(s), r))
#         return pd.DataFrame(rows, columns=['qid','docno','score','rank'])
    

class JPQRetrieverFlat(pt.Transformer):
    """Subset-mode retriever (Flat-IP reconstructed from codes)."""
    def __init__(self, docnos: List[str],
                 codes: np.array,
                 topk: int = 1000,
                 name: Optional[str] = None):
        super().__init__()
        self.docnos = np.array(docnos)
        self.codes = torch.from_numpy(codes).long()
        self.topk = topk
        self._index = None
        self._name = name or "JPQRetriever"
    
    def __str__(self): return self._name

    def _ensure(self, bs: int = 20000):
        if self._index is not None: return
        dim = self.biencoder.passage.sub_embeddings[0].embedding_dim * self.biencoder.passage.M
        index = faiss.IndexFlatIP(dim)
        pm = self.biencoder.passage.to("cpu").eval()
        with torch.no_grad():
            for i in tqdm(range(0, self.codes.size(0), bs), desc=f"{self._name} / build flat", leave=False):
                chunk = self.codes[i:i+bs]
                embs = pm(chunk)
                embs = (embs / (embs.norm(dim=1, keepdim=True) + 1e-12)).detach().cpu().numpy().astype('float32')
                index.add(embs)
        self._index = index
    
    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        pt.validate.query_frame(topics, extra_cols=['query_vec'])

        # with timer(f"{self._name} / encode_queries"):
        #     Q_t = self.biencoder.query.to("cpu").encode_texts(texts, batch_size=64)
        #     Q = Q_t.detach().cpu().numpy().astype('float32')
        #     Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        Q = topics["query_vec"].tolist()
        qids = topics['qid'].astype(str).tolist()
        self._ensure()
        with timer(f"{self._name} / flat search"):
            D, I = self._index.search(Q, min(self.topk, len(self.docnos)))
        rows = []
        for i, qid in enumerate(qids):
            for r, (did, s) in enumerate(zip(I[i], D[i]), start=1):
                rows.append((qid, self.docnos[int(did)], float(s), r))
        return pd.DataFrame(rows, columns=['qid','docno','score','rank'])