from abc import abstractmethod
import hashlib
import logging
import numpy as np
import torch
from tqdm import tqdm
import math
import faiss
from sklearn.cluster import KMeans

from pyterrier_dr.jpq.utils import code_type_from_Ks

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

def fingerprint_tensor_bits_torch(t: torch.Tensor) -> str:
    t_cpu = t.detach().cpu().contiguous()
    return hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()

def fingerprint_tensor_bits_np(t: np.ndarray) -> str:
    return hashlib.sha256(t.tobytes()).hexdigest()

# def _unpack_pq_codes_batch(packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
#     """Unpack FAISS PQ codes to shape (B, M) uint8.

#     FAISS packs product-quantiser codes into bytes when `nbits != 8`. This
#     function restores the (B, M) table of subquantiser indices in [0, 2^nbits).

#     Args:
#         packed: np.ndarray of shape (B, code_size_bytes) and dtype uint8,
#                 as returned by `pq.compute_codes(...)`.
#         M:      Number of subquantisers per vector (PQ's `M`).
#         nbits:  Bits per subquantiser code (e.g., 4, 5, 6, 7, or 8).

#     Returns:
#         np.ndarray of shape (B, M) and dtype uint8 with the unpacked codes.

#     Notes:
#         - For `nbits == 8`, FAISS already returns (B, M) uint8; we just return a view/copy.
#         - For `nbits < 8`, FAISS bit-packs the M codes into `ceil(M * nbits / 8)` bytes.
#         - We use little-endian bit order (FAISS packs least-significant bit first).
#     """
#     if packed.dtype != np.uint8:
#         packed = packed.astype(np.uint8, copy=False)

#     B = packed.shape[0]
#     if nbits == 8:
#         # Fast path: already (B, M) uint8 in most FAISS builds
#         if packed.shape[1] != M:
#             raise ValueError(f"Expected packed shape (B, {M}) for nbits=8, got {packed.shape}.")
#         return packed

#     if not (1 <= nbits < 8):
#         raise ValueError(f"nbits must be in [1,7] when packed; got {nbits}.")

#     # Total bits per vector and bytes actually provided
#     total_bits = M * nbits
#     code_size_bytes = packed.shape[1]
#     provided_bits = code_size_bytes * 8
#     if provided_bits < total_bits:
#         raise ValueError(
#             f"Packed buffer too small: need >= {total_bits} bits, have {provided_bits} bits."
#         )

#     # Unpack all bits (little-endian) → shape (B, code_size_bytes*8)
#     bits = np.unpackbits(packed, axis=1, bitorder="little")  # uint8 {0,1}

#     # Take exactly the bits we need per row
#     bits = bits[:, :total_bits]  # (B, M*nbits)

#     # Reshape into (B, M, nbits), then compute little-endian integer per subquantiser
#     bits_3d = bits.reshape(B, M, nbits)  # last axis: [b0, b1, ..., b(nbits-1)]
#     # weights for little-endian: [1, 2, 4, ...]
#     weights = (1 << np.arange(nbits, dtype=np.uint16)).astype(np.uint16)  # safe up to nbits<=16
#     codes = (bits_3d * weights).sum(axis=2).astype(np.uint8)  # (B, M)

#     return codes


class ProductQuantizer:
    def __init__(self, M: int=8, Ks: int=256):
        """
        M: number of subquantizers (splits of the vector)
        Ks: number of centroids per subquantizer
        """
        self._M = M
        self._Ks = Ks
        self._d = None # will be set after fit
        self._centroids = None # will be set after fit
    
    @abstractmethod
    def fit(self, X : np.ndarray):
        pass

    @abstractmethod
    def encode(self, X : np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def decode(self, codes):
        pass

    @property
    def centroids(self) -> np.ndarray:
        if self._centroids is None:
            raise AttributeError("Must call fit() first.")
        return self._centroids
    
    @centroids.setter
    def centroids(self, centroids: np.ndarray):
        self._centroids = centroids

    @property
    def d(self) -> int:
        if self._d is None:
            raise AttributeError("Must call fit() first.")
        return self._d
    
    @d.setter
    def d(self, d: int):
        if d % self._M != 0:
            raise ValueError(f"Dimensionality {d} must be divisible by M = {self._M}.")
        self._d = d

    @property
    def dsub(self) -> int:
        return self.d // self._M
    
    def encode_batch(
            self, 
            X: np.ndarray, # vector db
            selected: np.ndarray, # vector indexes to encode
            bs: int=10_000, 
            verbose: bool=True, 
            error: bool=True,
            gpu : None|torch.device = None
    ) -> np.ndarray:
        """Take some embeddings from X according to selected and encodes them in batches"""
        N = len(selected)
        codes = np.empty((N, self._M), dtype=code_type_from_Ks(self._Ks)) # [N, M]
        total_error = 0.0
        encode_method = self.encode
        encode_args = {}
        gpu_msg = "cpu"
        if gpu is not None:
            logger.info("Centroid fingerprint " + str( fingerprint_tensor_bits_np(self.centroids)))
            self.centroids_t = torch.from_numpy(self.centroids).to(gpu)
            encode_method = self.encode_gpu
            if gpu is not None:
                encode_args['device'] = gpu
            gpu_msg = str(gpu)

        iter = range(0, N, bs)
        for start in tqdm(iter, desc=f"[PQ] Encoding PQ batches ({gpu_msg})", total = math.ceil(N / bs)) if verbose else iter:
            end = min(start + bs, N) 
            X_sel = X[selected[start:end]] # [B, D]
            batch_codes = encode_method(X_sel, **encode_args) # [B, M]
            codes[start:end] = batch_codes # [B, M]
            
            if error:
                X_sel_recon = self.decode(batch_codes)  # [B, D]
                err = np.square(X_sel - X_sel_recon).sum(axis=1)  # [B]
                total_error += err.sum()
        
        logger.info("Codes fingerprint " + str( fingerprint_tensor_bits_np(codes)))
        root_mean_recon_error = np.sqrt(total_error / N)
        if error:
            logger.info(f"[PQ] Reconstruction RMSE: {root_mean_recon_error:.6f}")

        return codes


class ProductQuantizerSKLearn(ProductQuantizer):
    def __init__(self, M=4, Ks=256, random_state=42):
        super().__init__(M, Ks)
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> None: # [N, D]
        """Train PQ on data X [N, D]."""
        _, self.d = X.shape
        centroids = []
        for X_m in np.hsplit(X, self._M): # X_sub has shape [N, dsub]
            kmeans = KMeans(n_clusters=self._Ks, random_state=self.random_state)
            kmeans.fit(X_m)
            centroids.append(kmeans.cluster_centers_) # [Ks, dsub]
        self.centroids = np.array(centroids)  # shape (M, Ks, dsub)
        
    def encode(self, X: np.ndarray) -> np.ndarray: # [N, D] -> [N, M]
        """Encode each vector in X [N,D] into M integer codes."""
        N = len(X)
        codes = np.empty((N, self._M), dtype=code_type_from_Ks(self._Ks)) # [N, M]
        for m, X_m in enumerate(np.hsplit(X, self._M)): # X_m has shape [N, dsub]
            centers = self.centroids[m] # [Ks, dsub]
            distances = np.linalg.norm(X_m[:, None, :] - centers[None, :, :], axis=2)
            codes[:, m] = np.argmin(distances, axis=1)
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray: # [N, M] -> [N, D]
        """Reconstruct vectors from PQ codes."""
        N = len(codes)
        X_recon = np.empty((N, self.d), dtype=np.float32) # [N, D]
        for m, centers in enumerate(self.centroids):
            X_recon[:, m * self.dsub:(m + 1) * self.dsub] = centers[codes[:, m]]
        return X_recon


class ProductQuantizerFAISS(ProductQuantizer):
    def __init__(self, M=4, Ks=256):
        super().__init__(M=M, Ks=Ks)
        self._pq = None

    def fit(self, X: np.ndarray) -> None:
        """Train FAISS PQ on data X (n_samples, d)."""
        _, self.d = X.shape

        self.pq = faiss.ProductQuantizer(self.d, self._M, int(np.log2(self._Ks)))#, faiss.METRIC_INNER_PRODUCT )
        self.pq.train(X.astype(np.float32)) # type: ignore
        self._centroids = faiss.vector_to_array(self.pq.centroids).reshape(self._M, self._Ks, self.dsub).astype('float32')
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Assign PQ codes for X using dot product"""
        N = len(X)
        codes = np.empty((N, self._M), dtype=np.int64) # [N, M]
        for m, X_m in enumerate(np.hsplit(X, self._M)): # X_sub has shape [N, dsub]            
            C_m = self.centroids[m]                # [K, dsub]
            # Calculate the dot product using numpy.dot or the @ operator.
            similarity = X_m @ C_m.T
            # Find the index with the *maximum* dot product (highest similarity).
            idx = np.argmax(similarity, axis=1)    # [N]
            codes[:, m] = idx
        return codes
    
    def encode_gpu(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        X_t = torch.from_numpy(X).to(device)  # [N, D]
        N, D = X_t.shape
        X_view = X_t.view(N, self._M, self.dsub)              # [N, M, dsub]
        assert hasattr(self, "centroids_t") 
        assert self.centroids_t is not None
        #centroids_t = torch.from_numpy(self.centroids).cuda() # [M, Ks, dsub]
        # similarity[n, m, k] = dot(X_view[n, m, :], centroids[m, k, :])
        similarity = torch.einsum('nmd,mkd->nmk', X_view, self.centroids_t)
        codes_t = similarity.argmax(dim=-1)  # [N, M], int64 on GPU
        return codes_t.cpu().numpy().astype(np.int64)

    # def encode_(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Encode vectors into PQ codes (n_samples, M).
    #     Uses FAISS native API (no custom unpacking).
    #     """
    #     assert self.pq is not None, "Must call fit() first."
    #     X = X.astype(np.float32)
    #     n_samples = len(X)
    
    #     # Prepare an empty array for the codes
    #     codes = np.zeros((n_samples, self._M), dtype=code_type_from_Ks(self._Ks))
    
    #     # Compute PQ codes directly into the array
    #     faiss_pq = self.pq
    #     faiss_pq.compute_codes(
    #         faiss.swig_ptr(X),
    #         faiss.swig_ptr(codes),
    #         n_samples
    #     )
    
    #     return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray: # [N, M] -> [N, D]
        N = len(codes)
        X_recon = np.zeros((N, self.d), dtype=np.float32)  # [N, D]

        for m, centers in enumerate(self.centroids):
            X_recon[:, m * self.dsub:(m + 1) * self.dsub] = centers[codes[:, m]]
        return X_recon
    

class ProductQuantizerFAISSIndexPQ(ProductQuantizerFAISS):

    def fit(self, X: np.ndarray) -> None: # [N, D]
        """Train FAISS PQ on data X (n_samples, d)."""
        _, self.d = X.shape

        pqi = faiss.IndexPQ(self.d, self._M, int(np.log2(self._Ks)), faiss.METRIC_INNER_PRODUCT)

        # Initialize FAISS PQ
        pqi.train(X.astype(np.float32)) # type: ignore
        self.pq = pqi.pq
        self.centroids = faiss.vector_to_array(self.pq.centroids).reshape(self._M, self._Ks, self.dsub).astype('float32')

    
class ProductQuantizerFAISSIndexPQOPQ(ProductQuantizerFAISSIndexPQ):
    def fit(self, X):
        _, self.d = X.shape
        logger.info("X fingerprint " + str(fingerprint_tensor_bits_np(X)))

        opq = faiss.OPQMatrix(self.d, self._M)
        opq.train(X) # type: ignore
        
        x_rotated = opq.apply_py(X) # type: ignore
        logger.info("X rotated fingerprint "  + str(fingerprint_tensor_bits_np(x_rotated)))
        super().fit(x_rotated)

        # I've checked, T is what we want.
        self.opq = faiss.vector_to_array(opq.A).reshape(self.d, self.d).T.astype('float32')
        logger.info("OPQ fingerprint "  + str(fingerprint_tensor_bits_np(self.opq)))
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        # rotate before PQ encoding
        X = X @ self.opq
        return super().encode(X)
    
    def encode_gpu(self, X: np.ndarray, device: torch.device) -> np.ndarray:
        # rotate before PQ encoding
        X_rot = X @ self.opq
        return super().encode_gpu(X_rot, device)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        # decode with PQ, then apply inverse rotate
        X_recon = super().decode(codes)
        X_recon = X_recon @ self.opq.T
        return X_recon
