from typing import List, Optional
import pandas as pd
import numpy as np
import pyterrier as pt
from pyterrier_dr import FlexIndex
from pyterrier_dr.jpq.pq import ProductQuantizer
import torch
from .utils import timer
from pyterrier import tqdm

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


def build_from_flex(existing_index : FlexIndex, pq : ProductQuantizer, biencoder, dest_folder : str, bs : int = 20_000) -> FlexIndex:

        docnos, original_vec, _ = existing_index.payload()
        pm = biencoder.passage.to("cpu").eval()
        
        def _gen():
            running_se = 0.
            with torch.no_grad():
                for i in tqdm(range(0, len(existing_index), bs), desc=f"build flat", leave=False):
                    codes_batch = pq.encode(original_vec[i:i+bs])
                    embs = pm(torch.Tensor(codes_batch).int())
                    embs = (embs / (embs.norm(dim=1, keepdim=True) + 1e-12)).detach().cpu().numpy().astype('float32')
                    for j in range(codes_batch.shape[0]):
                        running_se += np.sum((embs[j] - original_vec[i+j])**2)
                        yield {'docno' : docnos[i+j], 'doc_vec' : embs[j]}
                print("Emb MSE", running_se/len(existing_index))
        
        new_index = FlexIndex(dest_folder).indexer(mode='overwrite').index(_gen())
        
        print("Old index size in docs", len(existing_index))
        print("New index size in docs", len(new_index))
        
        return new_index

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
        import faiss
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