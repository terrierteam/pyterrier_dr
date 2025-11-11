from abc import abstractmethod
import math
import pandas as pd
import numpy as np
import pyterrier as pt, pyterrier_alpha as pta
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
                for i in tqdm(range(0, len(existing_index), bs), desc=f"build flat", leave=False):
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

def build_inverted_index(item_codes, pq_type_name, dataset_models_config, current_dir, k : int = 256) -> np.ndarray:
    #dir = current_dir / "inverted_indexes" / pq_type_name / dataset_models_config.config_name
    #dir.mkdir(parents=True, exist_ok=True)
    #cache_filename = dir / "inverted_index.npy"

    if False:# os.path.exists(cache_filename):
        pass #code_items = np.load(cache_filename, allow_pickle=True)
    else:
        # Convert to NumPy if it's a tensor-like object
        target_codes = np.array(item_codes)
        num_items, num_splits = target_codes.shape
        code_items = [[] for _ in range(num_splits * k)]

        for split in tqdm(range(num_splits), unit='split'):
            for item in tqdm(range(num_items)):
                code = target_codes[item, split]
                code_items[split * k + code].append(item)

        # Compute padding
        items_per_code = [len(code_items[i]) for i in range(len(code_items))]
        max_items_per_code = max(items_per_code)

        # Pad each list with -1 so all arrays are the same length
        for code in range(len(code_items)):
            padding = max_items_per_code - len(code_items[code])
            if padding > 0:
                code_items[code] = [-1] * padding + code_items[code]

        # Convert to NumPy array
        code_items = np.array(code_items, dtype=np.int32)
        #np.save(cache_filename, code_items)

    index_int32 = code_items.astype(np.int32)
    return index_int32

# THIS DOES NOT YET WORK
def build_inverted_index_fast(item_codes, pq_type_name, dataset_models_config, current_dir, k: int = 256) -> np.ndarray:
    target_codes = np.asarray(item_codes, dtype=np.int32)
    num_items, num_splits = target_codes.shape
    num_codes = num_splits * k

    # Compute global code IDs (combine split + local code)
    global_codes = np.arange(num_splits) * k + target_codes  # shape = (num_items, num_splits)

    # Flatten for grouping
    flat_codes = global_codes.ravel()
    flat_items = np.repeat(np.arange(num_items, dtype=np.int32), num_splits)

    # Sort by code to make grouping fast
    order = np.argsort(flat_codes)
    flat_codes = flat_codes[order]
    flat_items = flat_items[order]

    # Find group boundaries
    unique_codes, start_idx = np.unique(flat_codes, return_index=True)
    end_idx = np.append(start_idx[1:], len(flat_codes))

    # Prepare result container
    code_items = [[] for _ in range(num_codes)]

    # Fill groups (only for existing codes)
    for code, s, e in zip(unique_codes, start_idx, end_idx):
        code_items[code] = flat_items[s:e]

    # Pad for uniform shape
    max_items_per_code = max(len(v) for v in code_items)
    padded = np.full((num_codes, max_items_per_code), -1, dtype=np.int32)

    for i, arr in enumerate(code_items):
        if len(arr) > 0:
            padded[i, -len(arr):] = arr  # right-align padding like before

    return padded


# FAISS stores PQ codes bit-packed when nbits < 8 (and also packs to 2 bytes when nbits > 8).
# Copying a plain [N, M] uint8 matrix into index.codes only works when nbits == 8. 
def _pack_pq_codes(codes: np.ndarray, nbits: int) -> np.ndarray:
    """
    Pack [N, M] uint8 codes (values in [0, 2^nbits)) into FAISS bit-packed layout.
    Returns a flat uint8 array of length N * code_size, where code_size = ceil(M*nbits/8).
    Packing is LSB-first per code value, contiguous over sub-quantizers.
    """
    assert codes.dtype == np.uint8 and codes.ndim == 2
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
        codes: np.ndarray, # PQ codes of shape [N, M] (uint8)
        sub_embeddings: np.ndarray, #  PQ centroids of shape [M, Ks, dsub] where Ks = 2^nbits and d = M * dsub (aka the centroids, float32)
        topk: int = 1000, # number of results to retrieve
    ):
        super().__init__()
        
        self.docnos = list(map(str, docnos))
        self.codes = np.ascontiguousarray(codes)
        self.sub_embeddings = np.ascontiguousarray(sub_embeddings)
        self.topk = int(topk)

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

        self._index = None
        self._name = self.__class__.__name__        

    def __str__(self) -> str:
        return self._name        


class JPQRetrieverFaissBase(JPQRetriever):

    def __init__(self, *args, name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = None
        self._name = name or self.__class__.__name__

    @abstractmethod
    def _ensure(self, bs: int = 20000) -> None:
        ...

    def _validate_queries(self, topics: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        pta.validate.query_frame(topics, extra_columns=["query_vec"])
        qvecs = topics["query_vec"].to_list()
        Q = np.stack(qvecs).astype(np.float32, copy=False)
        Q = np.ascontiguousarray(Q)
        qids = topics["qid"].astype(str).tolist()
        return Q, qids

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        pta.validate.query_frame(topics, extra_columns=['query_vec'])
        Q, qids = self._validate_queries(topics)
        self._ensure()
        with timer(f"{self._name} / search"):
            k = min(self.topk, len(self.docnos))
            D, I = self._index.search(Q, k) # type: ignore

        rows = []
        for i, qid in enumerate(qids):
            dids = I[i].astype(int)
            scores = D[i].astype(float)

            # stable, deterministic ordering: by score desc, then by doc index asc
            order = np.lexsort((dids, -scores))  # primary: -scores, secondary: dids
            dids = dids[order]
            scores = scores[order]

            for rank, (did, score) in enumerate(zip(dids, scores), start=1):
                rows.append((qid, self.docnos[did], score, rank))
        return pd.DataFrame(rows, columns=['qid','docno','score','rank'])


class JPQRetrieverFlat(JPQRetrieverFaissBase):
    """Subset-mode retriever (Flat-IP reconstructed from codes)."""

    def _ensure(self, bs: int = 20_000):
        if self._index is not None: 
            return
        
        index = faiss.IndexFlatIP(self.d)
        for start in tqdm(range(0, self.N, bs), desc=f"{self._name} / build flat", leave=False):
            stop = min(start + bs, self.N)
            chunk_codes = self.codes[start:stop, :]           # [b, M] uint8
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
        
        index = faiss.IndexPQ(self.d, self.M, self.nbits, faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True

        # Copy centroids: flat float array length M*Ks*dsub
        faiss.copy_array_to_vector(self.sub_embeddings.reshape(-1), index.pq.centroids)

        # Copy codes: flat uint8 array length N*M
        # pack codes if needed
        if self.nbits == 8:
            packed = self.codes.reshape(-1)
        else:
            packed = _pack_pq_codes(self.codes, self.nbits)

        faiss.copy_array_to_vector(packed, index.codes)

        index.ntotal = len(self.docnos)

        self._index = index
        self._name = "JPQRetrieverPQ"

        self._index = index
    

class JPQRetrieverPrune(JPQRetriever):

    """Subset-mode retriever - uses dynamic pruning."""
    def __init__(self,
                 *args,
                 name: str | None = None,
                 ub_inflation : float = 1.,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self._index = None
        self._name = name or "JPQRetrieverPrune"
        ks = self.sub_embeddings.shape[1] #2**nbits
        self.d = self.sub_embeddings.shape[2] * self.sub_embeddings.shape[0]
        self.dsub = self.sub_embeddings.shape[2]
        self.scorer = _PrunedScorer(
            ks, 
            build_inverted_index(self.codes, ..., ..., ..., k=ks),
            self.codes,
            self.codes.shape[-1], 
            top_k=self.topk,
            ub_inflation=ub_inflation
        )

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        pta.validate.query_frame(topics, extra_columns=['query_vec'])
        num_q = len(topics)
        Q = np.stack(topics["query_vec"].to_list())
        qids = topics['qid'].astype(str).tolist()
        with timer(f"{self._name} / prune search"):
            # split query_vec into M sub-vecs
            Q = Q.reshape((num_q, self.M, self.dsub))
            assert Q.shape == (num_q, self.M, self.dsub), Q.shape
            # TODO check this works query as the first dimension...?
            centroid_scores = np.einsum("mbd,md->mb", self.sub_embeddings, Q)
            assert centroid_scores.shape == (num_q, self.M, self.Ks), centroid_scores.shape
            Is = []
            Ds = []
            for qoffset in range(centroid_scores.shape[0]):
                I_q, D_q = self.scorer(centroid_scores[qoffset])
                Is.append(I_q)
                Ds.append(D_q)
            I, D = self.scorer(centroid_scores)
        rows = []
        for i, qid in enumerate(qids):
            for r, (did, s) in enumerate(zip(I[i], D[i]), start=1):
                rows.append((qid, self.docnos[int(did)], float(s), r))
        return pd.DataFrame(rows, columns=['qid','docno','score','rank'])

def merge_top_k(item_score_ids_1: np.ndarray,
                item_scores_values_1: np.ndarray,
                item_score_ids_2: np.ndarray,
                item_scores_values_2: np.ndarray,
                k: int,
                need_items: int):
    # Fast path: if the smallest in top-k is already larger than the largest new score, skip expensive merge
    if item_scores_values_1[-1] < item_scores_values_2[0]:
        # Create tiled versions to find duplicates
        tiled_1 = np.tile(item_score_ids_1[np.newaxis, :], (need_items, 1))
        tiled_2 = np.tile(item_score_ids_2[:, np.newaxis], (1, k))
        non_unique_items = np.any(tiled_1 == tiled_2, axis=1)

        # Penalize duplicate scores to avoid reselecting the same items
        item_scores_values_2_updated = item_scores_values_2 - non_unique_items.astype(np.float32) * 1e9

        # Combine old and new
        combined_ids = np.concatenate([item_score_ids_1, item_score_ids_2])
        combined_scores = np.concatenate([item_scores_values_1, item_scores_values_2_updated])

        # Select top-k
        sorted_idx = np.argpartition(-combined_scores, k)[:k]
        sorted_idx = sorted_idx[np.argsort(-combined_scores[sorted_idx])]

        top_k_ids = combined_ids[sorted_idx]
        top_k_scores = combined_scores[sorted_idx]

        # Update original arrays in place
        item_score_ids_1[:] = top_k_ids
        item_scores_values_1[:] = top_k_scores

# port of https://github.com/asash/recjpq_dp_pruning/blob/main/pruned_score_batch.py to PyTorch
class _PrunedScorer:
    def __init__(self, 
                 centroids_per_split : int, 
                 inverted_index : np.ndarray, 
                 item_codes : np.ndarray, 
                 item_code_bytes : int,
                 top_k : int = 10, 
                 max_iterations : int = 1000, 
                 ub_inflation : float = 1.0, 
                 sub_ids_per_iteration : int = 8):
        self.centroids_per_split = centroids_per_split
        self.item_codes = item_codes
        self.item_code_bytes = item_code_bytes
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.ub_inflation = ub_inflation
        self.inverted_index = inverted_index
        self.pointers = np.zeros(item_code_bytes, dtype=np.int32)
        self.pointer_increments = np.eye(item_code_bytes, dtype=np.int32) * sub_ids_per_iteration
        self.sub_ids_per_iteration = sub_ids_per_iteration
        self.iteration = 0
        self.upper_bound = float("inf")
        self.bound_tensor = np.full((item_code_bytes, 1), float("-inf"), dtype=np.float32)
        self.num_scored_items = 0
        self.last_lower_bound_update = 0
        self.threshold = float("-inf")
        self.old_lower_bound = float("-inf")
        self.best_items = np.zeros(top_k, dtype=np.int32)
        self.best_values = np.zeros(top_k, dtype=np.float32)
        self.scores_with_lower_bound = np.zeros((item_code_bytes, centroids_per_split + 1), dtype=np.float32)
        self.centroid_scores = np.zeros(item_code_bytes * centroids_per_split, dtype=np.float32)
        self.items_per_sub_id = inverted_index.shape[1]
        self.sorted_scores_indices = np.zeros((item_code_bytes, centroids_per_split), dtype=np.int32)

    def score_top_k(self, items, top_k):
        # filter out negative (padding) items
        valid_mask = items >= 0
        valid_items = items[valid_mask]

        # gather scores
        item_codes = self.item_codes[np.maximum(valid_items, 0)]
        item_score_full = np.sum(self.centroid_scores[item_codes], axis=1)

        # assign large negative value to masked (invalid) items
        full_scores = np.full(items.shape, -1e9, dtype=np.float32)
        full_scores[valid_mask] = item_score_full

        # get top-k
        top_k_idx = np.argpartition(-full_scores, top_k)[:top_k]
        top_k_idx = top_k_idx[np.argsort(-full_scores[top_k_idx])]

        return items[top_k_idx].astype(np.int32), full_scores[top_k_idx]

    def __call__(self, centroid_scores):
        self.centroid_scores[:] = centroid_scores

        # reshape and sort
        centroid_scores_reshaped = centroid_scores.reshape(self.item_code_bytes, self.centroids_per_split)
        sorted_scores_indices = np.argsort(-centroid_scores_reshaped, axis=1)
        sorted_scores_values = np.take_along_axis(centroid_scores_reshaped, sorted_scores_indices, axis=1)

        self.sorted_scores_indices[:] = sorted_scores_indices
        self.pointers[:] = 0
        self.upper_bound = float("inf")
        self.threshold = float("-inf")
        self.iteration = 0
        self.num_scored_items = 0
        self.last_lower_bound_update = 0

        self.scores_with_lower_bound = np.concatenate([sorted_scores_values, self.bound_tensor], axis=1)
        top_scores = self.scores_with_lower_bound[np.arange(self.item_code_bytes), self.pointers]

        need_items = self.top_k

        while (1 / (1 + np.exp(-self.upper_bound)) >
               (1 / (1 + np.exp(-self.threshold))) * self.ub_inflation) and (self.iteration < self.max_iterations):

            top_scored_centroid_split_num = np.argmax(top_scores)
            top_scored_centroid_num = self.pointers[top_scored_centroid_split_num]

            self.pointers[top_scored_centroid_split_num] += self.sub_ids_per_iteration

            # get top centroids for this split
            top_scored_centroids = (
                self.sorted_scores_indices[top_scored_centroid_split_num,
                                           top_scored_centroid_num:top_scored_centroid_num + self.sub_ids_per_iteration]
                + top_scored_centroid_split_num * self.centroids_per_split
            )

            items_with_centroids = self.inverted_index[top_scored_centroids].reshape(-1)

            if self.iteration == 0:
                items, vals = self.score_top_k(items_with_centroids, need_items)
                self.best_items[:] = items
                self.best_values[:] = vals
            else:
                items, vals = self.score_top_k(items_with_centroids, need_items)
                merge_top_k(self.best_items, self.best_values, items, vals, self.top_k, need_items) # type: ignore

            self.old_lower_bound = self.threshold
            self.threshold = self.best_values[-1]
            self.num_scored_items += len(items_with_centroids)
            self.iteration += 1

            top_scores = self.scores_with_lower_bound[np.arange(self.item_code_bytes), self.pointers]
            self.upper_bound = np.sum(top_scores)
            n_safe_items = np.sum(self.best_values > self.upper_bound)
            need_items = self.top_k - n_safe_items

            if self.old_lower_bound != self.threshold:
                self.last_lower_bound_update = self.iteration

        return self.best_items, self.best_values