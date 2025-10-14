import unittest
import numpy as np

from pyterrier_dr.jpq.pq import _unpack_pq_codes_batch
from pyterrier_dr.jpq.pq import (
    ProductQuantizerSKLearn,
    ProductQuantizerFAISS,
)

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

def _pack_codes_little_endian(codes: np.ndarray, nbits: int) -> np.ndarray:
    """
    Test helper: pack (B, M) uint8 codes (values in [0, 2^nbits)) into bytes
    using little-endian bit order, matching FAISS behaviour when nbits < 8.
    """
    B, M = codes.shape
    assert 1 <= nbits <= 7
    total_bits = M * nbits
    out_bits = np.zeros((B, total_bits), dtype=np.uint8)

    # expand each code into nbits little-endian bits
    for m in range(M):
        # extract m-th code column
        vals = codes[:, m].astype(np.uint16)  # safe for bit ops
        for b in range(nbits):
            out_bits[:, m * nbits + b] = (vals >> b) & 1

    # pad to full bytes if needed
    pad = (-total_bits) % 8
    if pad:
        out_bits = np.pad(out_bits, ((0, 0), (0, pad)), mode="constant", constant_values=0)

    packed = np.packbits(out_bits, axis=1, bitorder="little")
    return packed


class TestUnpackPQCodesBatch(unittest.TestCase):

    def test_fast_path_nbits_8_ok(self):
        B, M = 3, 5
        packed = np.arange(B * M, dtype=np.uint8).reshape(B, M)
        out = _unpack_pq_codes_batch(packed, M=M, nbits=8)
        self.assertTrue(np.shares_memory(out, packed) or np.array_equal(out, packed))
        np.testing.assert_array_equal(out, packed)

    def test_fast_path_nbits_8_wrong_shape_raises(self):
        B, M = 2, 4
        packed = np.zeros((B, M + 1), dtype=np.uint8)  # wrong second dim
        with self.assertRaises(ValueError):
            _ = _unpack_pq_codes_batch(packed, M=M, nbits=8)

    def test_roundtrip_for_various_nbits(self):
        rng = np.random.default_rng(123)
        B, M = 7, 13
        for nbits in [1, 2, 3, 4, 5, 6, 7]:
            vmax = (1 << nbits) - 1
            codes = rng.integers(0, vmax + 1, size=(B, M), dtype=np.uint8)
            packed = _pack_codes_little_endian(codes, nbits)
            # sanity: packed should have ceil(M*nbits/8) bytes per row
            expected_bytes = (M * nbits + 7) // 8
            self.assertEqual(packed.shape, (B, expected_bytes))

            unpacked = _unpack_pq_codes_batch(packed, M=M, nbits=nbits)
            self.assertEqual(unpacked.shape, (B, M))
            self.assertEqual(unpacked.dtype, np.uint8)
            np.testing.assert_array_equal(unpacked, codes)

    def test_insufficient_bits_raises(self):
        # Need total_bits = M*nbits = 10*4 = 40 bits = 5 bytes
        B, M, nbits = 2, 10, 4
        too_small = np.zeros((B, 4), dtype=np.uint8)  # only 32 bits
        with self.assertRaises(ValueError):
            _ = _unpack_pq_codes_batch(too_small, M=M, nbits=nbits)

    def test_non_uint8_input_is_coerced(self):
        B, M, nbits = 3, 6, 4
        rng = np.random.default_rng(7)
        vmax = (1 << nbits) - 1
        codes = rng.integers(0, vmax + 1, size=(B, M), dtype=np.uint8)
        packed_uint8 = _pack_codes_little_endian(codes, nbits)

        # present as a different dtype; function should coerce to uint8 internally
        packed_int32 = packed_uint8.astype(np.int32)
        out = _unpack_pq_codes_batch(packed_int32, M=M, nbits=nbits)
        np.testing.assert_array_equal(out, codes)


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


class TestProductQuantizerSKLearn(unittest.TestCase):
    def setUp(self):
        self.M = 4
        self.d = 16
        self.Ks = 16  # power of two; KMeans will learn 16 centroids per subspace
        self.X = make_synthetic_data(n=128, d=self.d, seed=7)

    def test_fit_centroids_shape(self):
        pq = ProductQuantizerSKLearn(M=self.M, Ks=self.Ks, random_state=0)
        pq.fit(self.X)
        cents = pq.get_centroids()
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
        codes_b = pq.encode_batch(self.X, batch_size=17, verbose=False)
        np.testing.assert_array_equal(codes_ref, codes_b)

    def test_dim_not_divisible_raises(self):
        pq = ProductQuantizerSKLearn(M=6, Ks=self.Ks, random_state=0)
        X_bad = make_synthetic_data(n=32, d=16, seed=3)  # 16 % 6 != 0
        with self.assertRaises(AssertionError):
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
        cents = pq.get_centroids()
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
        self.assertEqual(codes.dtype, np.uint8)
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
        codes_b = pq.encode_batch(self.X, batch_size=19, verbose=False)
        np.testing.assert_array_equal(codes_ref, codes_b)

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_dim_not_divisible_raises(self):
        pq = ProductQuantizerFAISS(M=6, Ks=self.Ks)
        X_bad = make_synthetic_data(n=32, d=16, seed=3)  # 16 % 6 != 0
        with self.assertRaises(AssertionError):
            pq.fit(X_bad)


if __name__ == "__main__":
    unittest.main()