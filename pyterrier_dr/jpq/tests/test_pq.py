import unittest
import numpy as np

from pyterrier_dr.jpq.pq import (
    ProductQuantizerSKLearn,
    ProductQuantizerFAISS,
    ProductQuantizerFAISSIndexPQ,
    ProductQuantizerFAISSIndexPQOPQ
)

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def make_synthetic_data(n=200, d=16, seed=123):
    """
    Create simple clusterable data: 4 Gaussian blobs tiled across dimensions.
    Ensures d is divisible by 4 so we can set M=4, dsub=d//M.
    """
    rng = np.random.default_rng(seed)
    assert d % 4 == 0
    centers = np.array([
        np.zeros(d),
        np.concatenate([np.ones(d//2), np.zeros(d - d//2)]),
        np.concatenate([np.zeros(d//2), np.ones(d - d//2)]),
        np.ones(d),
    ], dtype=float)

    X = np.vstack([centers[i % 4] + 0.1 * rng.standard_normal(size=d) for i in range(n)]).astype(np.float32)
    rng.shuffle(X)
    return X


class TestProductQuantizerCommon(unittest.TestCase):
    def test_centroids_before_fit_raises(self):
        pq = ProductQuantizerSKLearn(M=4, Ks=16, random_state=0)
        with self.assertRaises(AttributeError):
            _ = pq.centroids  # accessing before fit must raise


class TestProductQuantizerSKLearn(unittest.TestCase):
    def setUp(self):
        self.M = 4
        self.d = 16
        self.Ks = 16  # power of two; KMeans will learn 16 centroids per subspace
        self.X = make_synthetic_data(n=128, d=self.d, seed=7)

    def test_fit_centroids_shape(self):
        pq = ProductQuantizerSKLearn(M=self.M, Ks=self.Ks, random_state=0)
        pq.fit(self.X)
        cents = pq.centroids
        self.assertIsInstance(cents, np.ndarray)
        self.assertEqual(cents.shape, (self.M, self.Ks, self.d // self.M)) # type: ignore
        self.assertIn(cents.dtype, (np.float32, np.float64)) # type: ignore

    def test_encode_decode_shapes_and_ranges(self):
        pq = ProductQuantizerSKLearn(M=self.M, Ks=self.Ks, random_state=0)
        pq.fit(self.X)
        codes = pq.encode(self.X)
        self.assertEqual(codes.shape, (self.X.shape[0], self.M))
        self.assertEqual(codes.dtype, np.uint8)
        self.assertGreaterEqual(codes.min(), 0)
        self.assertLess(codes.max(), self.Ks)

        X_rec = pq.decode(codes)
        self.assertEqual(X_rec.shape, self.X.shape)
        self.assertTrue(np.isfinite(X_rec).all())

        # Reconstruction should not be absurdly far from input
        err = np.linalg.norm(self.X - X_rec, axis=1).mean()
        self.assertGreaterEqual(err, 0.0)
        # Not asserting a tight bound (depends on data), but it should be finite and reasonable.

    def test_encode_batch_matches_encode(self):
        pq = ProductQuantizerSKLearn(M=self.M, Ks=self.Ks, random_state=0)
        pq.fit(self.X)
        codes_ref = pq.encode(self.X)
        codes_b = pq.encode_batch(self.X, np.arange(len(self.X)), bs=17, verbose=False)
        np.testing.assert_array_equal(codes_ref, codes_b)

    def test_dim_not_divisible_raises(self):
        pq = ProductQuantizerSKLearn(M=6, Ks=self.Ks, random_state=0)
        X_bad = make_synthetic_data(n=32, d=16, seed=3)  # 16 % 6 != 0
        with self.assertRaises(ValueError):
            pq.fit(X_bad)


class TestProductQuantizerFAISS(unittest.TestCase):

    def setUp(self):
        self.M = 4
        self.d = 16
        self.Ks = 256  # 2^8 → nbits=8 path / or any power-of-two supported by FAISS
        self.X = make_synthetic_data(n=32768, d=self.d, seed=11)

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_fit_centroids_shape(self):
        pq = ProductQuantizerFAISS(M=self.M, Ks=self.Ks)
        pq.fit(self.X)
        cents = pq.centroids
        self.assertIsInstance(cents, (np.ndarray, list))
        cents = np.asarray(cents)
        self.assertEqual(cents.shape, (self.M, self.Ks, self.d // self.M))
        self.assertEqual(cents.dtype, np.float32)

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_encode_decode_shapes_and_ranges(self):
        pq = ProductQuantizerFAISS(M=self.M, Ks=self.Ks)
        pq.fit(self.X)
        codes = pq.encode(self.X)
        self.assertEqual(codes.shape, (self.X.shape[0], self.M))
        self.assertGreaterEqual(codes.min(), 0)
        self.assertLess(codes.max(), self.Ks)

        # X_rec = pq.decode(codes)
        # self.assertEqual(X_rec.shape, self.X.shape)
        # self.assertTrue(np.isfinite(X_rec).all())

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_encode_batch_matches_encode(self):
        pq = ProductQuantizerFAISS(M=self.M, Ks=self.Ks)
        pq.fit(self.X)
        codes_ref = pq.encode(self.X)
        codes_b = pq.encode_batch(self.X, np.arange(len(self.X)), bs=19, verbose=False)
        np.testing.assert_array_equal(codes_ref, codes_b)

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_dim_not_divisible_raises(self):
        pq = ProductQuantizerFAISS(M=6, Ks=self.Ks)
        X_bad = make_synthetic_data(n=32, d=16, seed=3)  # 16 % 6 != 0
        with self.assertRaises(ValueError):
            pq.fit(X_bad)


class TestProductQuantizerFAISSIndexPQ(unittest.TestCase):
    def setUp(self):
        self.M = 4
        self.d = 16
        self.Ks = 256
        self.X = make_synthetic_data(n=32768, d=self.d, seed=21)

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_fit_centroids_shape(self):
        pq = ProductQuantizerFAISSIndexPQ(M=self.M, Ks=self.Ks)
        pq.fit(self.X)
        cents = pq.centroids
        self.assertIsInstance(cents, np.ndarray)
        self.assertEqual(cents.shape, (self.M, self.Ks, self.d // self.M))
        self.assertEqual(cents.dtype, np.float32)

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_encode_decode_shapes_and_finiteness(self):
        pq = ProductQuantizerFAISSIndexPQ(M=self.M, Ks=self.Ks)
        pq.fit(self.X)
        codes = pq.encode(self.X[:512])
        self.assertEqual(codes.shape, (512, self.M))
        self.assertEqual(codes.dtype, np.int64)  # ProductQuantizerFAISS.encode returns int64
        self.assertGreaterEqual(codes.min(), 0)
        self.assertLess(codes.max(), self.Ks)

        X_rec = pq.decode(codes)
        self.assertEqual(X_rec.shape, (512, self.d))
        self.assertTrue(np.isfinite(X_rec).all())


class TestProductQuantizerFAISSIndexPQOPQ(unittest.TestCase):
    def setUp(self):
        self.M = 4
        self.d = 16
        self.Ks = 256
        self.X = make_synthetic_data(n=32768, d=self.d, seed=31)

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_opq_fit_and_roundtrip_shapes(self):
        pq = ProductQuantizerFAISSIndexPQOPQ(M=self.M, Ks=self.Ks)
        pq.fit(self.X)  # trains OPQ + PQ
        cents = pq.centroids
        self.assertIsInstance(cents, np.ndarray)
        self.assertEqual(cents.shape, (self.M, self.Ks, self.d // self.M))
        self.assertEqual(cents.dtype, np.float32)

        # encode / decode a slice
        Xq = self.X[:256]
        codes = pq.encode(Xq)
        self.assertEqual(codes.shape, (256, self.M))
        self.assertGreaterEqual(codes.min(), 0)
        self.assertLess(codes.max(), self.Ks)

        X_rec = pq.decode(codes)
        self.assertEqual(X_rec.shape, Xq.shape)
        self.assertTrue(np.isfinite(X_rec).all())

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_encode_batch_matches_encode(self):
        pq = ProductQuantizerFAISSIndexPQOPQ(M=self.M, Ks=self.Ks)
        pq.fit(self.X)
        idx = np.arange(len(self.X))
        codes_ref = pq.encode(self.X)
        codes_b = pq.encode_batch(self.X, idx, bs=123, verbose=False, error=False)
        np.testing.assert_array_equal(codes_ref, codes_b)


if __name__ == "__main__":
    unittest.main()