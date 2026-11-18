"""Tests for pyterrier_dr.repconc.index and pyterrier_dr.repconc.retriever.

Verifies:

* build/reload round-trips (metadata, docnos, codes, centroids, rotation),
* ADC (flat PQ) retriever scores and orderings are exact against
  brute-force dot products with the reconstructed corpus (zero
  quantisation error, tie-free corpus),
* the flat (reconstructed-vector) retriever agrees with the ADC retriever,
* the IVF (nlist > 1) index round-trips through disk (faiss binary) and is
  exact with nprobe == nlist,
* the text-based RepCONCRetriever encodes queries and retrieves.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyterrier as pt

from pyterrier_dr.repconc.index import RepCONCIndex
from pyterrier_dr.repconc.retriever import RepCONCRetriever


def _exact_corpus(M=4, nbits=2, dsub=3, seed=0):
    """Corpus with argmin codes (unique code tuples, so no score ties)."""
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


def _orthonormal(d, seed=0):
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.standard_normal((d, d)))[0].astype(np.float32)


class _HashQueryEncoder(pt.Transformer):
    """Encodes query texts with a deterministic hashed bag-of-vectors."""

    def __init__(self, W):
        super().__init__()
        self.W = W

    def transform(self, inp):
        out = inp.copy()
        vecs = []
        for t in out["query"]:
            v = np.zeros(self.W.shape[1], dtype=np.float32)
            for tok in str(t).split():
                v += self.W[abs(hash(tok)) % self.W.shape[0]]
            vecs.append(v)
        out["query_vec"] = vecs
        return out


class _HashBiEncoder:
    """A minimal stand-in for a BiEncoder: encodes query texts with a
    hashed bag-of-vectors scheme (deterministic)."""

    def __init__(self, d, vocab=1024, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((vocab, d)).astype(np.float32)

    def query_encoder(self, verbose=None, batch_size=None):
        return _HashQueryEncoder(self.W)

    def __str__(self):
        return "HashBiEncoder"


class TestRepCONCIndex(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "idx"

    def tearDown(self):
        self._tmp.cleanup()

    def _build(self, with_rotation=True, nlist=1, nprobe=10, **kw):
        mode = kw.pop("mode", None)
        d, M, nbits, centroids, codes, recon, x_rot = _exact_corpus(**kw)
        rotation = _orthonormal(d) if with_rotation else None
        docnos = [f"doc{i}" for i in range(len(codes))]
        args = dict(path=self.path, docnos=docnos, codes=codes, centroids=centroids,
                    rotation=rotation, x_rot=x_rot, nlist=nlist, nprobe=nprobe)
        if mode is not None:
            args["mode"] = mode
        return RepCONCIndex.build(**args), d, M, nbits, centroids, codes, recon, x_rot

    def test_build_and_reload(self):
        index, d, M, nbits, centroids, codes, recon, _ = self._build()
        self.assertEqual(len(index), len(codes))

        reloaded = RepCONCIndex(self.path)
        self.assertEqual(reloaded.meta.M, M)
        self.assertEqual(reloaded.meta.Ks, 2 ** nbits)
        self.assertEqual(reloaded.meta.dsub, d // M)
        self.assertEqual(reloaded.meta.doc_count, len(codes))
        self.assertTrue(reloaded.meta.rotation)
        self.assertEqual(reloaded.meta.nlist, 1)
        np.testing.assert_array_equal(reloaded.codes, codes)
        np.testing.assert_array_equal(reloaded.subvecs, centroids)
        np.testing.assert_array_equal(reloaded.docnos.fwd, [f"doc{i}" for i in range(len(codes))])
        self.assertEqual(reloaded.d, d)

    def test_no_rotation(self):
        index, d, *_ = self._build(with_rotation=False)
        self.assertFalse(index.meta.rotation)
        self.assertIsNone(index.rotation)
        reloaded = RepCONCIndex(self.path)
        self.assertIsNone(reloaded.rotation)

    def test_rotation_shape(self):
        index, d, *_ = self._build()
        self.assertEqual(index.rotation.shape, (d, d))
        # orthonormal
        np.testing.assert_allclose(index.rotation @ index.rotation.T, np.eye(d), atol=1e-5)

    def test_build_validation(self):
        d, M, nbits, centroids, codes, recon, x_rot = _exact_corpus()
        docnos = [f"doc{i}" for i in range(len(codes))]
        with self.assertRaises(ValueError):
            RepCONCIndex.build(self.path, docnos=docnos[:-1], codes=codes, centroids=centroids)
        with self.assertRaises(ValueError):
            RepCONCIndex.build(self.path, docnos=docnos, codes=codes,
                               centroids=centroids, rotation=np.eye(d + 1))
        # nlist > 1 requires x_rot of the right shape
        with self.assertRaises(ValueError):
            RepCONCIndex.build(self.path, docnos=docnos, codes=codes, centroids=centroids,
                               x_rot=np.zeros((len(codes), d + 1)), nlist=2)
        # overwrite mode
        self._build()
        with self.assertRaises(RuntimeError):
            self._build()
        index, *_ = self._build(mode="overwrite")
        self.assertTrue(index.index_path.exists())

    def test_ivf_requires_corpus(self):
        d, M, nbits, centroids, codes, recon, x_rot = _exact_corpus()
        docnos = [f"doc{i}" for i in range(len(codes))]
        with self.assertRaises(ValueError):
            RepCONCIndex.build(self.path, docnos=docnos, codes=codes, centroids=centroids,
                               x_rot=None, nlist=2)


class TestRetrievers(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "idx"
        self.d, self.M, self.nbits, self.centroids, self.codes, self.recon, self.x_rot = _exact_corpus()
        self.rotation = _orthonormal(self.d)
        self.docnos = [f"doc{i}" for i in range(len(self.codes))]
        self.index = RepCONCIndex.build(
            self.path, docnos=self.docnos, codes=self.codes, centroids=self.centroids,
            rotation=self.rotation, x_rot=self.x_rot, nlist=1,
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _topics(self, rng, nq=3, topk=10):
        Q = rng.standard_normal((nq, self.d)).astype(np.float32)
        return pd.DataFrame({
            "qid": [f"q{i}" for i in range(nq)],
            "query": [f"query {i} text" for i in range(nq)],
            "query_vec": [Q[i] for i in range(nq)],
        })

    def _brute_force(self, topics):
        Q = np.stack(topics["query_vec"]).astype(np.float32) @ self.rotation
        scores = Q @ self.recon.T
        return scores

    def test_flat_pq_retriever_exact(self):
        rng = np.random.default_rng(1)
        topics = self._topics(rng)
        res = self.index.retriever_pq(topk=10)(topics)
        self.assertEqual(len(res), 3 * 10)
        for i, qid in enumerate(topics["qid"]):
            expected = self._brute_force(topics)[i]
            order = np.argsort(-expected)
            sub = res[res["qid"] == qid].reset_index(drop=True)
            self.assertEqual(sub["docid"].tolist(), order[:10].tolist(),
                             f"qid {qid}: ADC ordering differs from brute force")
            np.testing.assert_allclose(sub["score"].to_numpy(), expected[order[:10]], atol=1e-4)
            self.assertEqual(sub["docno"].tolist(), [self.docnos[j] for j in order[:10]])
            self.assertEqual(sub["rank"].tolist(), list(range(1, 11)))
            self.assertEqual(sub["query"].tolist(), [topics.iloc[i]["query"]] * 10)

    def test_flat_retriever_matches_pq(self):
        rng = np.random.default_rng(2)
        topics = self._topics(rng)
        res_pq = self.index.retriever_pq(topk=10)(topics)
        res_flat = self.index.retriever_flat(topk=10)(topics)
        # same ordering; scores agree up to float32 accumulation differences
        # (ADC sums per-subspace table entries, flat search sums the full dot)
        np.testing.assert_array_equal(res_pq["docid"].to_numpy(), res_flat["docid"].to_numpy())
        np.testing.assert_allclose(res_pq["score"].to_numpy(), res_flat["score"].to_numpy(), atol=1e-5)

    def test_reloaded_index_retriever(self):
        rng = np.random.default_rng(3)
        topics = self._topics(rng)
        reloaded = RepCONCIndex(self.path)
        res_pq = self.index.retriever_pq(topk=8)(topics)
        res_rel = reloaded.retriever_pq(topk=8)(topics)
        self.assertEqual(res_pq[["qid", "docid"]].to_numpy().tolist(),
                         res_rel[["qid", "docid"]].to_numpy().tolist())

    def test_nbits_8(self):
        d, M, nbits, centroids, codes, recon, x_rot = _exact_corpus(M=2, nbits=8, dsub=3)
        rotation = _orthonormal(d)
        docnos = [f"doc{i}" for i in range(len(codes))]
        path = Path(self._tmp.name) / "idx8"
        index = RepCONCIndex.build(path, docnos=docnos, codes=codes, centroids=centroids,
                                   rotation=rotation, nlist=1)
        rng = np.random.default_rng(4)
        Q = rng.standard_normal((2, d)).astype(np.float32)
        topics = pd.DataFrame({"qid": ["q0", "q1"], "query_vec": [Q[0], Q[1]]})
        res = index.retriever_pq(topk=5)(topics)
        expected = (Q @ rotation) @ recon.T
        for i in range(2):
            order = np.argsort(-expected[i])
            sub = res[res["qid"] == f"q{i}"]
            self.assertEqual(sub["docid"].tolist(), order[:5].tolist())

    def test_ivf_roundtrip(self):
        d, M, nbits, centroids, codes, recon, x_rot = _exact_corpus(M=4, nbits=2, dsub=4)
        rotation = _orthonormal(d, seed=5)
        docnos = [f"doc{i}" for i in range(len(codes))]
        path = Path(self._tmp.name) / "idxivf"
        RepCONCIndex.build(path, docnos=docnos, codes=codes, centroids=centroids,
                           rotation=rotation, x_rot=x_rot, nlist=4, nprobe=4)

        # the faiss binary must exist for nlist > 1
        self.assertTrue((path / "faiss.index").exists())

        index = RepCONCIndex(path)
        self.assertEqual(index.meta.nlist, 4)
        self.assertEqual(index.meta.nprobe, 4)

        rng = np.random.default_rng(6)
        Q = rng.standard_normal((3, d)).astype(np.float32)
        topics = pd.DataFrame({"qid": ["q0", "q1", "q2"], "query_vec": [Q[i] for i in range(3)]})
        res = index.retriever_pq(topk=8)(topics)
        for i in range(3):
            expected = (Q[i] @ rotation) @ recon.T
            order = np.argsort(-expected)
            sub = res[res["qid"] == f"q{i}"].reset_index(drop=True)
            # nprobe == nlist: exhaustive and exact
            self.assertEqual(sub["docid"].tolist(), order[:8].tolist(),
                             f"qid q{i}: IVF ordering differs from brute force")
            np.testing.assert_allclose(sub["score"].to_numpy(), expected[order[:8]], atol=1e-4)

    def test_text_retriever(self):
        d = self.d
        be = _HashBiEncoder(d)
        r = RepCONCRetriever(be, self.index, topk=5)
        topics = pd.DataFrame({"qid": ["q0", "q1"], "query": ["hello world", "foo bar baz"]})
        res = r(topics)
        self.assertEqual(set(res.columns), {"qid", "docid", "docno", "score", "rank", "query"})
        self.assertEqual(len(res), 2 * 5)
        for qid in ("q0", "q1"):
            sub = res[res["qid"] == qid]
            self.assertEqual(len(sub), 5)
            self.assertEqual(sub["rank"].tolist(), [1, 2, 3, 4, 5])
            # deterministic: same input, same output
            res2 = r(topics)
            self.assertEqual(res2[["qid", "docid"]].to_numpy().tolist(),
                             res[["qid", "docid"]].to_numpy().tolist())

    def test_invalid_data(self):
        from pyterrier_dr.repconc.retriever import RepCONCRetrieverPQ
        with self.assertRaises(ValueError):
            RepCONCRetrieverPQ(["a", "b"], self.codes, self.centroids)
        with self.assertRaises(ValueError):
            RepCONCRetrieverPQ(self.docnos, self.codes, np.zeros((self.M, 3, self.d // self.M), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
