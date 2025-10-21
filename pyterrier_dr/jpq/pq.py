from abc import abstractmethod
import numpy as np


def _unpack_pq_codes_batch(packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
    """Unpack FAISS PQ codes to shape (B, M) uint8.

    FAISS packs product-quantiser codes into bytes when `nbits != 8`. This
    function restores the (B, M) table of subquantiser indices in [0, 2^nbits).

    Args:
        packed: np.ndarray of shape (B, code_size_bytes) and dtype uint8,
                as returned by `pq.compute_codes(...)`.
        M:      Number of subquantisers per vector (PQ's `M`).
        nbits:  Bits per subquantiser code (e.g., 4, 5, 6, 7, or 8).

    Returns:
        np.ndarray of shape (B, M) and dtype uint8 with the unpacked codes.

    Notes:
        - For `nbits == 8`, FAISS already returns (B, M) uint8; we just return a view/copy.
        - For `nbits < 8`, FAISS bit-packs the M codes into `ceil(M * nbits / 8)` bytes.
        - We use little-endian bit order (FAISS packs least-significant bit first).
    """
    if packed.dtype != np.uint8:
        packed = packed.astype(np.uint8, copy=False)

    B = packed.shape[0]
    if nbits == 8:
        # Fast path: already (B, M) uint8 in most FAISS builds
        if packed.shape[1] != M:
            raise ValueError(f"Expected packed shape (B, {M}) for nbits=8, got {packed.shape}.")
        return packed

    if not (1 <= nbits < 8):
        raise ValueError(f"nbits must be in [1,7] when packed; got {nbits}.")

    # Total bits per vector and bytes actually provided
    total_bits = M * nbits
    code_size_bytes = packed.shape[1]
    provided_bits = code_size_bytes * 8
    if provided_bits < total_bits:
        raise ValueError(
            f"Packed buffer too small: need >= {total_bits} bits, have {provided_bits} bits."
        )

    # Unpack all bits (little-endian) → shape (B, code_size_bytes*8)
    bits = np.unpackbits(packed, axis=1, bitorder="little")  # uint8 {0,1}

    # Take exactly the bits we need per row
    bits = bits[:, :total_bits]  # (B, M*nbits)

    # Reshape into (B, M, nbits), then compute little-endian integer per subquantiser
    bits_3d = bits.reshape(B, M, nbits)  # last axis: [b0, b1, ..., b(nbits-1)]
    # weights for little-endian: [1, 2, 4, ...]
    weights = (1 << np.arange(nbits, dtype=np.uint16)).astype(np.uint16)  # safe up to nbits<=16
    codes = (bits_3d * weights).sum(axis=2).astype(np.uint8)  # (B, M)

    return codes


class ProductQuantizer:
    def __init__(self, M=4, Ks=256):
        """
        M: number of subquantizers (splits of the vector)
        Ks: number of centroids per subquantizer
        """
        self.M = M
        self.Ks = Ks
        self.centroids = []
    
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def encode(self, X):
        pass

    def get_centroids(self) -> np.ndarray | list:
        return self.centroids

    def encode_batch(self, veclookup, indices, batch_size=10_000, verbose=True, error=True) -> np.ndarray:
        n = len(indices)
        codes = np.empty((n, self.M), dtype=np.uint8)
        iter = range(0, n, batch_size)
        total_error = 0.0

        if verbose:
            from tqdm import tqdm
            iter = tqdm(iter, desc="[PQ] Encoding PQ batches")
        for start in iter:
            end = min(start + batch_size, n)
            vecs = veclookup[indices[start:end]]
            batch_codes = self.encode(vecs)      # [B, M]
            codes[start:end] = batch_codes
            
            if error:
                # --- Reconstruction ---
                # Look up centroids to reconstruct vector
                # self.centroids shape: [M, Ks, D_sub]
                _, _, D_sub = self.centroids.shape

                # Reconstruct each subvector from its centroid
                reconstructed = np.zeros_like(vecs, dtype=np.float32)  # [B, D]
                for m in range(self.M):
                    reconstructed[:, m * D_sub:(m + 1) * D_sub] = self.centroids[m][batch_codes[:, m]]

                # Compute reconstruction error (e.g. mean squared error)
                errors = np.square(vecs - reconstructed).sum(axis=1)  # [B]
                total_error += errors.sum()
        
        root_mean_recon_error = np.sqrt(total_error / n)
        if error:
            print(f"[PQ] Reconstruction RMSE: {root_mean_recon_error:.6f}")
        return codes

    @abstractmethod
    def decode(self, codes):
        pass

class ProductQuantizerSKLearn(ProductQuantizer):
    def __init__(self, M=4, Ks=256, random_state=42):
        """
        M: number of subquantizers (splits of the vector)
        Ks: number of centroids per subquantizer
        """
        super().__init__(M, Ks)
        self.random_state = random_state
        self.dsub = 0 # dimensionality of each subvector

    def fit(self, X):
        from sklearn.cluster import KMeans
        """Train PQ on data X (n_samples, d)."""
        n_samples, d = X.shape
        assert d % self.M == 0, "Dimensionality must be divisible by M."
        self.dsub = d // self.M
        centroids = []
        
        for m in range(self.M):
            X_sub = X[:, m * self.dsub:(m + 1) * self.dsub]
            kmeans = KMeans(n_clusters=self.Ks, random_state=self.random_state)
            kmeans.fit(X_sub)
            centroids.append(kmeans.cluster_centers_)
        self.centroids = np.array(centroids)  # shape (M, Ks, dsub)
        
    def encode(self, X) -> np.ndarray:
        """Encode each vector into M integer codes."""
        n_samples = X.shape[0]
        codes = np.empty((n_samples, self.M), dtype=np.uint8)
        for m in range(self.M):
            X_sub = X[:, m * self.dsub:(m + 1) * self.dsub]
            centers = self.centroids[m]
            distances = np.linalg.norm(X_sub[:, None, :] - centers[None, :, :], axis=2)
            codes[:, m] = np.argmin(distances, axis=1)
        return codes
    
    def decode(self, codes) -> np.ndarray:
        """Reconstruct vectors from PQ codes."""
        n_samples = codes.shape[0]
        X_recon = np.empty((n_samples, self.M * self.dsub), dtype=np.float32)
        for m in range(self.M):
            centers = self.centroids[m]
            X_recon[:, m * self.dsub:(m + 1) * self.dsub] = centers[codes[:, m]]
        return X_recon


class ProductQuantizerFAISS(ProductQuantizer):
    def __init__(self, M=4, Ks=256):
        """
        M: number of subquantizers
        Ks: number of clusters per subquantizer
        """
        super().__init__(M, Ks)
        self.dsub = 0
        self.pq = None

    def fit(self, X):
        """Train FAISS PQ on data X (n_samples, d)."""
        n_samples, d = X.shape
        assert d % self.M == 0, "Dimensionality must be divisible by M."
        self.dsub = d // self.M
        import faiss

        # Initialize FAISS PQ
        self.pq = faiss.ProductQuantizer(d, self.M, int(np.log2(self.Ks)))#, faiss.METRIC_INNER_PRODUCT )
        self.pq.train(X.astype(np.float32)) # type: ignore
        self.centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.M, self.Ks, self.dsub).astype('float32')
        return self
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Assign PQ codes for X using dot product, return (codes [N, M], recon [N, D])."""
        # X: (N, D)
        N, D = X.shape
        dsub = D // self.M
        codes = np.empty((N, self.M), dtype=np.int64)
        #recon_parts = []
        
        for m in range(self.M):
            x_m = X[:, m * dsub : (m + 1) * dsub]  # (N, dsub)
            C_m = self.centroids[m]                # (K, dsub)
    
            # Calculate the dot product using numpy.dot or the @ operator.
            # This performs a batched matrix multiplication.
            # (N, K) = (N, dsub) @ (dsub, K)
            # Note: In NumPy, the '@' operator is preferred for matrix multiplication over the older `np.dot` function for 2D arrays.
            similarity = x_m @ C_m.T
    
            # Find the index with the *maximum* dot product (highest similarity).
            idx = np.argmax(similarity, axis=1)    # (N,)
            codes[:, m] = idx
            #recon_parts.append(C_m[idx])           # (N, dsub)
            
        #recon = np.concatenate(recon_parts, axis=-1)  # (N, D)
        return codes #, recon
    
    def encode_(self, X) -> np.ndarray:
        """
        Encode vectors into PQ codes (n_samples, M).
        Uses FAISS native API (no custom unpacking).
        """
        import faiss
        assert self.pq is not None, "Must call fit() first."
        X = X.astype(np.float32)
        n_samples = X.shape[0]
    
        # Prepare an empty array for the codes
        codes = np.zeros((n_samples, self.M), dtype=np.uint8)
    
        # Compute PQ codes directly into the array
        faiss_pq = self.pq
        faiss_pq.compute_codes(
            faiss.swig_ptr(X),
            faiss.swig_ptr(codes),
            n_samples
        )
    
        return codes

    # def encode(self, X) -> np.ndarray:
    #     """Encode vectors into PQ codes (n_samples, n_splits)."""
    #     assert self.pq is not None, "Must call fit() first."
    #     n_samples = X.shape[0]
    #     packed = self.pq.compute_codes(X.astype(np.float32)) # type: ignore
    #     codes = _unpack_pq_codes_batch(packed, M=self.M, nbits=int(np.log2(self.Ks)))
    #     assert codes.shape == (n_samples, self.M)
    #     return codes

    def decode(self, codes) -> np.ndarray:
        """Decode PQ codes back to approximate vectors."""
        n_samples = codes.shape[0]
        assert self.pq is not None, "Must call fit() first."
        X_recon = np.zeros((n_samples, self.M * self.dsub), dtype=np.float32)
        self.pq.decode(codes, X_recon)
        return X_recon
    

class ProductQuantizerFAISSIndexPQ(ProductQuantizerFAISS):
    def fit(self, X):
        """Train FAISS PQ on data X (n_samples, d)."""
        n_samples, d = X.shape
        assert d % self.M == 0, "Dimensionality must be divisible by M."
        self.dsub = d // self.M
        import faiss
        pqi = faiss.IndexPQ(d, self.M, int(np.log2(self.Ks)), faiss.METRIC_INNER_PRODUCT)

        # Initialize FAISS PQ
        pqi.train(X.astype(np.float32)) # type: ignore
        self.pq = pqi.pq
        self.centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.M, self.Ks, self.dsub).astype('float32')
        return self