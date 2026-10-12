from abc import abstractmethod
import math
import pandas as pd
import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_dr import FlexIndex
from pyterrier_dr.jpq.pq import ProductQuantizer
import torch
from .utils import timer
from pyterrier import tqdm

import faiss

def build_from_flex(existing_index : FlexIndex, pq : ProductQuantizer, biencoder, dest_folder : str, bs : int = 20_000) -> FlexIndex:

        docnos, original_vec, _ = existing_index.payload()
        pm = biencoder.passage.to("cpu").eval()
        
        def _gen():
            running_se = 0.
            with torch.no_grad():
                for i in tqdm(range(0, len(existing_index), bs), desc="build flat", leave=False):
                    codes_batch = pq.encode(original_vec[i:i+bs])
                    embs = pm(torch.Tensor(codes_batch).int()).detach().cpu().numpy().astype('float32')
                    #embs = (embs / (embs.norm(dim=1, keepdim=True) + 1e-12)).detach().cpu().numpy().astype('float32')
                    for j in range(codes_batch.shape[0]): # type: ignore
                        running_se += np.sum((embs[j] - original_vec[i+j])**2)
                        yield {'docno' : docnos[i+j], 'doc_vec' : embs[j]}
                print("Emb MSE", running_se/len(existing_index))
        
        new_index = FlexIndex(dest_folder).indexer(mode='overwrite').index(_gen())
        
        print("Old index size in docs", len(existing_index))
        print("New index size in docs", len(new_index))
        
        return new_index

# FAISS stores PQ codes bit-packed when nbits < 8 (and also packs to 2 bytes when nbits > 8).
# Copying a plain [N, M] uint8 matrix into index.codes only works when nbits == 8. 
def _pack_pq_codes(codes: np.ndarray, nbits: int) -> np.ndarray:
    """
    Pack [N, M] uint8 codes (values in [0, 2^nbits)) into FAISS bit-packed layout.
    Returns a flat uint8 array of length N * code_size, where code_size = ceil(M*nbits/8).
    Packing is LSB-first per code value, contiguous over sub-quantizers.
    """
    assert codes.dtype in [np.uint8, np.uint16], str(codes.dtype)
    assert codes.ndim == 2, str(codes.ndim)
    N, M = codes.shape
    code_size = (M * nbits + 7) // 8  # bytes per vector
    out = np.zeros((N, code_size), dtype=np.uint8)

    bit_cursor = 0
    for m in range(M):
        v = codes[:, m].astype(np.uint32)  # [N]
        # write nbits LSB-first into the bitstream
        for b in range(nbits):
            byte_idx = (bit_cursor + b) // 8
            bit_in_byte = (bit_cursor + b) % 8
            out[:, byte_idx] |= (((v >> b) & 1).astype(np.uint8)) << bit_in_byte
        bit_cursor += nbits

    return out.reshape(-1)  # flat vector as FAISS expects


class JPQRetriever(pt.Transformer):

    def __init__(
        self, 
        docnos: list[str], # list of [N] docids
        codes: np.ndarray, # PQ codes of shape [N, M] (uint8 or uint16)
        sub_embeddings: np.ndarray, #  PQ centroids of shape [M, Ks, dsub] where Ks = 2^nbits and d = M * dsub (aka the centroids, float32)
        topk: int = 1000, # number of results to retrieve
        opq : np.ndarray | None = None,
    ):
        super().__init__()
        
        self.docnos = list(map(str, docnos))
        self.codes = np.ascontiguousarray(codes)
        self.sub_embeddings = np.ascontiguousarray(sub_embeddings)
        self.topk = int(topk)
        self.opq = opq

        self._validate_data()

        self._index = None
        self._name = self.__class__.__name__        

    def __str__(self) -> str:
        return self._name        

    def _validate_data(self):
        if self.codes.ndim != 2:
            raise ValueError(f"codes must be [N, M], got {self.codes.shape}")
        if self.sub_embeddings.ndim != 3:
            raise ValueError(f"sub_embeddings must be [M, Ks, dsub], got {self.sub_embeddings.shape}")

        N, M = self.codes.shape
        M2, Ks, dsub = self.sub_embeddings.shape
        if len(self.docnos) != N:
            raise ValueError(f"len(docnos)={len(self.docnos)} != N={N}")
        if M != M2:
            raise ValueError(f"M mismatch: codes has {M}, sub_embeddings has {M2}")

        nbits_f = math.log2(Ks)
        if abs(nbits_f - round(nbits_f)) > 1e-9:
            raise ValueError(f"Ks must be a power of 2, got {Ks}")
        
        self.N = N
        self.M = M
        self.Ks = Ks
        self.dsub = dsub
        self.d = M * dsub
        self.nbits = int(round(nbits_f))

    def _validate_queries(self, topics: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        pta.validate.query_frame(topics, extra_columns=["query_vec"])
        qvecs = topics["query_vec"].to_list()
        if not len(topics):
            return np.empty((0, self.d), dtype=np.float32), []
        Q = np.stack(qvecs).astype(np.float32, copy=False)
        Q = np.ascontiguousarray(Q)
        if self.opq is not None:
            Q = Q @ self.opq
        qids = topics["qid"].astype(str).tolist()
        return Q, qids


class JPQRetrieverFaissBase(JPQRetriever):

    def __init__(self, *args, name: str | None = None, gpu=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = None
        self._name = name or self.__class__.__name__
        self._gpu = gpu

    def __del__(self):
        if self._index is not None:
            del self._index
            self._index = None

    @abstractmethod
    def _ensure(self, bs: int = 20000) -> None:
        ...

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        Q, qids = self._validate_queries(topics)
        self._ensure()
        k = min(self.topk, len(self.docnos))
        D, doc_ids = self._index.search(Q, k) # type: ignore

        rows = []
        for i, qid in enumerate(qids):
            dids = doc_ids[i].astype(int)
            scores = D[i].astype(float)

            # stable, deterministic ordering: by score desc, then by doc index asc
            order = np.lexsort((dids, -scores))  # primary: -scores, secondary: dids
            dids = dids[order]
            scores = scores[order]

            for rank, (did, score) in enumerate(zip(dids, scores), start=1):
                row = [qid, did, self.docnos[did], score, rank]
                if "query" in topics:
                    row.append(topics.iloc[i]["query"])

                rows.append(row)
        
        cols = ['qid', 'docid', 'docno', 'score', 'rank']
        if "query" in topics:
            cols.append("query")
            
        return pd.DataFrame(rows, columns=cols)


class JPQRetrieverFlat(JPQRetrieverFaissBase):
    """Subset-mode retriever (Flat-IP reconstructed from codes)."""

    def _ensure(self, bs: int = 20_000):
        if self._index is not None: 
            return
        
        index = faiss.IndexFlatIP(self.d)
        if self._gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        for start in tqdm(range(0, self.N, bs), desc=f"{self._name} / build flat", leave=False):
            stop = min(start + bs, self.N)
            chunk_codes = self.codes[start:stop, :]           # [b, M] uint8 or uint16
            parts = [self.sub_embeddings[m, chunk_codes[:, m], :] for m in range(self.M)]
            embs = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)  # [b, M*dsub]
            embs = np.ascontiguousarray(embs)
            index.add(embs) # type: ignore
            
        self._index = index
        self._name = "JPQRetrieverFlat"


class JPQRetrieverPQ(JPQRetrieverFaissBase):
    """Subset-mode retriever (IndexPQ constructed directly from codes and centroids)."""
    
    def _ensure(self, bs: int = 20_000) -> None:
        if self._index is not None:
            return
        
        if self._gpu:
            raise ValueError("IndexPQ cannot be moved to gpu")
        
        index = faiss.IndexPQ(self.d, self.M, self.nbits, faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True

        # Copy centroids: flat float array length M*Ks*dsub
        faiss.copy_array_to_vector(self.sub_embeddings.reshape(-1), index.pq.centroids)

        # Copy codes: flat uint8 or unit16 array length N*M
        # pack codes if needed
        if self.nbits == 8:
            packed = self.codes.reshape(-1)
        else:
            packed = _pack_pq_codes(self.codes, self.nbits)

        faiss.copy_array_to_vector(packed, index.codes)

        index.ntotal = len(self.docnos)

        self._index = index
        self._name = "JPQRetrieverPQ"
   
