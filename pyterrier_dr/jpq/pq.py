from abc import abstractmethod
import numpy as np

class ProductQuantizer:
    def __init__(self, M=4, Ks=256, backend='faiss', random_state=42):
        """
        M: number of subquantizers (splits of the vector)
        Ks: number of centroids per subquantizer
        backend: 'faiss' or 'sklearn'
        """
        self.M = M
        self.Ks = Ks
    
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def encode(self, X):
        pass

    def get_centroids(self):
        return self.centroids

    def encode_batch(self, X, batch_size=10_000, verbose=True) -> np.array:
        n = X.shape[0]
        codes = np.empty((n, self.M), dtype=np.uint8)
        iter = range(0, n, batch_size)
        if verbose:
            from tqdm import tqdm
            iter = tqdm(iter, desc="Encoding PQ batches")
        for start in iter:
            end = min(start + batch_size, n)
            codes[start:end] = self.encode(X[start:end])
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
        self.centroids = None
        self.dsub = None # dimensionality of each subvector

    def fit(self, X):
        from sklearn.cluster import KMeans
        """Train PQ on data X (n_samples, d)."""
        n, d = X.shape
        assert d % self.M == 0, "Dimensionality must be divisible by M."
        self.dsub = d // self.M
        centroids = []
        
        for m in range(self.M):
            X_sub = X[:, m * self.dsub:(m + 1) * self.dsub]
            kmeans = KMeans(n_clusters=self.Ks, random_state=self.random_state)
            kmeans.fit(X_sub)
            centroids.append(kmeans.cluster_centers_)
        self.centroids = np.array(centroids)  # shape (M, Ks, dsub)
        
    def encode(self, X):
        """Encode each vector into M integer codes."""
        n = X.shape[0]
        codes = np.empty((n, self.M), dtype=np.uint8)
        for m in range(self.M):
            X_sub = X[:, m * self.dsub:(m + 1) * self.dsub]
            centers = self.centroids[m]
            distances = np.linalg.norm(X_sub[:, None, :] - centers[None, :, :], axis=2)
            codes[:, m] = np.argmin(distances, axis=1)
        return codes
    
    def decode(self, codes):
        """Reconstruct vectors from PQ codes."""
        n = codes.shape[0]
        X_recon = np.empty((n, self.M * self.dsub), dtype=np.float32)
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
        self.dsub = None
        self.pq = None
        import faiss

    def fit(self, X):
        """Train FAISS PQ on data X (n_samples, d)."""
        n, d = X.shape
        assert d % self.M == 0, "Dimensionality must be divisible by M."
        self.dsub = d // self.M
        import faiss

        # Initialize FAISS PQ
        self.pq = faiss.ProductQuantizer(d, self.M, int(np.log2(self.Ks)))
        self.pq.train(X.astype(np.float32))
        self.centroids = faiss.vector_to_array(self.pq.centroids).reshape(self.M, self.Ks, self.dsub).astype('float32')
        return self

    def encode(self, X):
        """Encode vectors into PQ codes (n, M)."""
        assert self.pq is not None, "Must call fit() first."
        #codes = np.zeros((X.shape[0], self.M), dtype=np.uint8)
        # TODO check packing when nbits != 8
        # if packed.shape[1] != pq_M:  # packed bytes when nbits != 8
        codes = self.pq.compute_codes(X.astype(np.float32))
        assert codes.shape == (X.shape[0], self.M)
        return codes

    def decode(self, codes):
        """Decode PQ codes back to approximate vectors."""
        assert self.pq is not None, "Must call fit() first."
        X_recon = np.zeros((codes.shape[0], self.M * self.dsub), dtype=np.float32)
        self.pq.decode(codes, X_recon)
        return X_recon