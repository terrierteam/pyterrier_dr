import math
import unittest
import numpy as np
import pandas as pd

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

from pyterrier_dr.jpq.retriever import JPQRetrieverFlat, JPQRetrieverPQ  # noqa: E402

def reconstruct_embeddings(codes: np.ndarray, sub_embeddings: np.ndarray) -> np.ndarray:
    """
    Reconstruct full vectors from PQ codes by concatenating selected centroids.
    codes: [N, M] uint8
    sub_embeddings: [M, Ks, dsub] float32
    returns: [N, d] float32
    """
    N, M = codes.shape
    M2, Ks, dsub = sub_embeddings.shape
    assert M == M2
    parts = []
    for m in range(M):
        # [Ks, dsub] -> index with [N] -> [N, dsub]
        parts.append(sub_embeddings[m, codes[:, m], :])
    X = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(X)


def brute_force_scores(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return inner products Q @ X^T as float32."""
    return (Q @ X.T).astype(np.float32, copy=False)


def dataframe_to_ranking(df: pd.DataFrame):
    """Return dict qid -> list[(docno, score)] ordered by rank."""
    out = {}
    df2 = df.sort_values(["qid", "rank"])
    for qid, g in df2.groupby("qid"):
        out[qid] = list(zip(g["docno"].tolist(), g["score"].tolist()))
    return out


@unittest.skipUnless(HAS_FAISS, "FAISS not installed; skipping JPQ retriever tests.")
class TestJPQRetrievers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(12345)

        # Small synthetic JPQ setup
        cls.N = 50
        cls.M = 4
        cls.Ks = 8        # must be a power of 2
        cls.dsub = 4
        cls.d = cls.M * cls.dsub
        assert abs(math.log2(cls.Ks) - round(math.log2(cls.Ks))) < 1e-9

        # centroids: [M, Ks, dsub] float32
        cls.sub_embeddings = rng.standard_normal((cls.M, cls.Ks, cls.dsub), dtype=np.float32)
        # codes: [N, M] uint8
        cls.codes = rng.integers(0, cls.Ks, size=(cls.N, cls.M), dtype=np.uint8)
        # docnos
        cls.docnos = [f"D{i:03d}" for i in range(cls.N)]
        # queries
        cls.B = 3
        Q = rng.standard_normal((cls.B, cls.d)).astype(np.float32)
        cls.topics = pd.DataFrame({
            "qid": [f"Q{i+1}" for i in range(cls.B)],
            "query_vec": list(Q),
            "query": ["synthetic"] * cls.B,  # harmless extra column
        })

        # Precompute brute-force baseline
        X = reconstruct_embeddings(cls.codes, cls.sub_embeddings)
        cls.BF_scores = brute_force_scores(Q, X)  # [B, N]


    def test_flat(self):
        topk = 10
        # expected from brute force
        expected = {}
        for b, qid in enumerate(self.topics["qid"]):
            scores_b = self.BF_scores[b]
            idx = np.arange(self.N)

            # order by score desc, then by doc index asc
            order = np.lexsort((idx, -scores_b))[:topk]
            expected[qid] = [(self.docnos[i], float(self.BF_scores[b, i])) for i in order]

        retr = JPQRetrieverFlat(self.docnos, self.codes, self.sub_embeddings, topk)
        res = retr.transform(self.topics)
        got = dataframe_to_ranking(res)

        for qid in expected:
            exp_docs = [d for d, _ in expected[qid]]
            exp_scores = np.array([s for _, s in expected[qid]], dtype=np.float32)
            got_docs = [d for d, _ in got[qid]]
            got_scores = np.array([s for _, s in got[qid]], dtype=np.float32)

            self.assertEqual(got_docs, exp_docs, f"Flat ranking mismatch for {qid}")
            np.testing.assert_allclose(got_scores, exp_scores, rtol=1e-5, atol=1e-6)

    def test_pq(self):
        topk = 10
        retr_flat = JPQRetrieverFlat(self.docnos, self.codes, self.sub_embeddings, topk)
        res_flat = retr_flat.transform(self.topics)
        flat_rank = dataframe_to_ranking(res_flat)

        retr_pq = JPQRetrieverPQ(self.docnos, self.codes, self.sub_embeddings, topk)
        res_pq = retr_pq.transform(self.topics)
        pq_rank = dataframe_to_ranking(res_pq)

        for qid in flat_rank:
            flat_docs = [d for d, _ in flat_rank[qid]]
            flat_scores = np.array([s for _, s in flat_rank[qid]], dtype=np.float32)
            pq_docs = [d for d, _ in pq_rank[qid]]
            pq_scores = np.array([s for _, s in pq_rank[qid]], dtype=np.float32)

            self.assertEqual(pq_docs, flat_docs, f"PQ vs Flat doc order mismatch for {qid}")
            np.testing.assert_allclose(pq_scores, flat_scores, rtol=1e-5, atol=1e-6)
