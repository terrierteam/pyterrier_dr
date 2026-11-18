"""Tests for pyterrier_dr.repconc.ivf (FAISS ADC index construction).

NB: the pyterrier_dr import (which loads torch) must come before ``import
faiss``: on macOS the two bundled OpenMP runtimes (torch + faiss-wheels)
cannot be mixed if faiss's is loaded first (see
:func:`pyterrier_dr.util.ensure_faiss_omp_compat`).
"""
import unittest

import numpy as np

from pyterrier_dr.repconc.ivf import build_faiss_index, pack_pq_codes
from pyterrier_dr.util import ensure_faiss_omp_compat

import faiss  # noqa: E402  (after torch, see module docstring)

# some raw faiss searches in this file bypass build_faiss_index; make sure
# the macOS OpenMP workaround is in place before the first parallel op
ensure_faiss_omp_compat()


def _exact_centroid_corpus(M=4, nbits=2, dsub=3, seed=0):
    """A corpus whose documents are quantised with zero error: the codes are
    the *argmin* assignments of ``x_rot`` to the codebook (so building an
    index with them reproduces the same codes, as ``faiss`` does on
    ``add``), and the decoded vectors equal the ADC reconstructions, so ADC
    scores equal brute-force dot products.

    The random corpus is redrawn until all code tuples are distinct, so the
    reconstructions (and hence scores) have no ties and the expected top-k
    ordering is unambiguous.
    """
    Ks = 2 ** nbits
    d = M * dsub
    N = M * Ks
    for s in range(seed, seed + 100):
        rng = np.random.default_rng(s)
        centroids = rng.standard_normal((M, Ks, dsub)).astype(np.float32)
        x_rot = rng.standard_normal((N, d)).astype(np.float32)
        x_m = x_rot.reshape(N, M, dsub)
        codes = np.stack([
            ((x_m[:, m:m+1] - centroids[m][None]) ** 2).sum(-1).argmin(-1)
            for m in range(M)
        ], axis=1).astype(np.uint8)
        if len({tuple(c) for c in codes}) == N:
            break
    else:
        raise RuntimeError("could not draw a tie-free test corpus")
    recon = np.concatenate([centroids[m, codes[:, m], :] for m in range(M)], axis=-1)
    return d, M, nbits, centroids, codes, recon, x_rot


class TestPackPQCodes(unittest.TestCase):

    def test_roundtrip_via_adc_scores(self):
        # nbits=2: bit-packed codes; the ADC scores must equal the dot
        # product with the codebook-reconstructed vectors
        d, M, nbits, centroids, codes, recon, _ = _exact_centroid_corpus(M=4, nbits=2, dsub=3, seed=1)
        Ks = 2 ** nbits
        index = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True
        faiss.copy_array_to_vector(centroids.reshape(-1).astype(np.float32), index.pq.centroids)
        packed = pack_pq_codes(codes, nbits)
        self.assertEqual(packed.shape, (len(codes) * ((M * nbits + 7) // 8),))
        faiss.copy_array_to_vector(packed, index.codes)
        index.ntotal = len(codes)
        rng = np.random.default_rng(2)
        Q = rng.standard_normal((5, d)).astype(np.float32)
        D, I = index.search(Q, min(10, len(codes)))
        for i in range(5):
            dids = I[i]
            expected = Q[i] @ recon[dids].T
            np.testing.assert_allclose(D[i], expected, atol=1e-4)

    def test_flat_pq_adc_exact(self):
        d, M, nbits, centroids, codes, recon, _ = _exact_centroid_corpus(seed=3)
        index = build_faiss_index(d, M, nbits, centroids, codes, nlist=1)
        rng = np.random.default_rng(4)
        Q = rng.standard_normal((6, d)).astype(np.float32)
        D, I = index.search(Q, 10)
        # exact: scores equal brute-force dot with the (reconstructed) corpus
        for i in range(6):
            expected = Q[i] @ recon.T
            order = np.argsort(-expected)
            np.testing.assert_array_equal(I[i], order[:10])
            np.testing.assert_allclose(D[i], expected[order[:10]], atol=1e-4)

    def test_nbits_8_unpack(self):
        d, M, nbits, centroids, codes, recon, _ = _exact_centroid_corpus(M=4, nbits=8, dsub=2, seed=5)
        index = build_faiss_index(d, M, nbits, centroids, codes, nlist=1)
        self.assertEqual(index.ntotal, len(codes))


class TestIVF(unittest.TestCase):

    def test_ivf_adc_all_probes_exact(self):
        # with nprobe == nlist, the IVF search is exhaustive and (for a
        # zero-quantisation-error corpus) exact
        d, M, nbits, centroids, codes, recon, x_rot = _exact_centroid_corpus(M=4, nbits=2, dsub=4, seed=6)
        # the codes are the argmin assignments of x_rot, so building the IVF
        # (which re-assigns codes on add) must reproduce them exactly
        index = build_faiss_index(d, M, nbits, centroids, codes, x_rot=x_rot, nlist=4)
        index.nprobe = index.nlist
        rng = np.random.default_rng(7)
        Q = rng.standard_normal((4, d)).astype(np.float32)
        D, I = index.search(Q, 8)
        for i in range(4):
            expected = Q[i] @ recon.T
            order = np.argsort(-expected)
            np.testing.assert_array_equal(I[i], order[:8])
            np.testing.assert_allclose(D[i], expected[order[:8]], atol=1e-4)

    def test_ivf_requires_corpus(self):
        d, M, nbits, centroids, codes, _, _ = _exact_centroid_corpus(seed=8)
        with self.assertRaises(ValueError):
            build_faiss_index(d, M, nbits, centroids, codes, x_rot=None, nlist=2)

    def test_ivf_nlist_exceeds_corpus(self):
        d, M, nbits, centroids, codes, _, _ = _exact_centroid_corpus(seed=9)
        with self.assertRaises(ValueError):
            build_faiss_index(d, M, nbits, centroids, codes, x_rot=np.zeros((codes.shape[0], d), dtype=np.float32),
                              nlist=codes.shape[0] + 1)

    def test_codebook_mismatch(self):
        d, M, nbits, centroids, codes, _, _ = _exact_centroid_corpus(seed=10)
        with self.assertRaises(ValueError):
            build_faiss_index(d, M, nbits + 1, centroids, codes, nlist=1)


if __name__ == "__main__":
    unittest.main()
