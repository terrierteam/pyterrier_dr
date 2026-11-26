import unittest

import numpy as np
import pandas as pd
import torch

from pyterrier_dr.jpq.prf import (
    PQPrf,
    PQPrfTF,
    PQPrfTFIDF,
    PQPrfProbabilisticTFIDF,
    PQPrfKLTFIDF,
    pq_prf_tf,
    pq_prf_tfidf,
    pq_prf_probabilistic_tfidf,
    pq_prf_kl_tfidf,
)


class _FakeJPQIndex:
    def __init__(self, centroids, codes):
        self.dvecs = np.asarray(centroids)
        self.codes = np.asarray(codes)

    def __len__(self):
        return len(self.codes)


class TestPQPrfFunctions(unittest.TestCase):

    def setUp(self):
        # One split, two centroids, two dimensions.
        self.centroids = torch.tensor(
            [[[0.0, 0.0], [3.0, 4.0]]], dtype=torch.float32
        )
        self.query = torch.tensor([1.0, 2.0], dtype=torch.float32)
        self.scores = torch.tensor([0.0], dtype=torch.float32)
        self.codes = torch.tensor([[1]], dtype=torch.long)
        self.freq = torch.tensor([[1, 1]], dtype=torch.long)
        self.n_docs = 10

    def test_abstract_base_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            PQPrf(centroids=self.centroids)

    def test_tf(self):
        out = pq_prf_tf(
            self.query, self.scores, self.codes, self.centroids, alpha=0.5
        )
        np.testing.assert_allclose(out.numpy(), [2.0, 3.0])

    def test_tf_alpha_one_returns_query(self):
        out = pq_prf_tf(
            self.query, self.scores, self.codes, self.centroids, alpha=1.0
        )
        np.testing.assert_allclose(out.numpy(), self.query.numpy())

    def test_tf_averages_equally_weighted_codes(self):
        centroids = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]]], dtype=torch.float32
        )
        codes = torch.tensor([[0], [1], [2]], dtype=torch.long)
        scores = torch.zeros(3)
        out = pq_prf_tf(
            torch.zeros(2), scores, codes, centroids, alpha=0.0
        )
        np.testing.assert_allclose(out.numpy(), [2.0 / 3.0, 4.0 / 3.0])

    def test_tfidf(self):
        out = pq_prf_tfidf(
            self.query,
            self.scores,
            self.codes,
            self.centroids,
            collection_code_freq=self.freq,
            n_docs=self.n_docs,
            alpha=0.0,
        )
        np.testing.assert_allclose(out.numpy(), [3.0, 4.0])

    def test_probabilistic_tfidf(self):
        out = pq_prf_probabilistic_tfidf(
            self.query,
            self.scores,
            self.codes,
            self.centroids,
            collection_code_freq=self.freq,
            n_docs=self.n_docs,
            alpha=0.0,
        )
        np.testing.assert_allclose(out.numpy(), [3.0, 4.0])

    def test_probabilistic_tfidf_clips_negative_idf(self):
        # df=9 with N=10 gives a negative Robertson IDF, which is clipped to 0.
        freq = torch.tensor([[1, 9]], dtype=torch.long)
        out = pq_prf_probabilistic_tfidf(
            self.query,
            self.scores,
            self.codes,
            self.centroids,
            collection_code_freq=freq,
            n_docs=self.n_docs,
            alpha=0.5,
        )
        np.testing.assert_allclose(out.numpy(), [0.5, 1.0])

    def test_kl_tfidf(self):
        out = pq_prf_kl_tfidf(
            self.query,
            self.scores,
            self.codes,
            self.centroids,
            collection_code_freq=self.freq,
            n_docs=self.n_docs,
            alpha=0.0,
        )
        np.testing.assert_allclose(out.numpy(), [3.0, 4.0])

    def test_kl_tfidf_only_expands_towards_enriched_codes(self):
        # Code 1 is no more frequent in the PRF set than in the corpus, so it
        # should not contribute; code 0 is enriched and should drive feedback.
        freq = torch.tensor([[1, 1]], dtype=torch.long)
        codes = torch.tensor([[0]], dtype=torch.long)
        centroids = torch.tensor(
            [[[0.0, 0.0], [3.0, 4.0]]], dtype=torch.float32
        )
        out = pq_prf_kl_tfidf(
            self.query,
            self.scores,
            codes,
            centroids,
            collection_code_freq=freq,
            n_docs=self.n_docs,
            alpha=0.0,
        )
        np.testing.assert_allclose(out.numpy(), [0.0, 0.0])


class TestPQPrfTransformers(unittest.TestCase):

    def setUp(self):
        self.centroids = np.array([
            [[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]],
        ], dtype=np.float32)
        self.codes = np.array([
            [0, 1],
            [1, 2],
            [2, 0],
            [2, 2],
        ], dtype=np.uint8)
        self.freq = np.array([
            [1, 1, 2],
            [1, 1, 2],
        ], dtype=np.int64)
        self.n_docs = len(self.codes)
        self.index = _FakeJPQIndex(self.centroids, self.codes)

    def _input(self):
        return pd.DataFrame([
            ['q1', 'query one', [0.5, 0.5, 0.5, 0.5], 'd1', 3.0, 1, [0, 1], [1.0, 0.0, 3.0, 0.0]],
            ['q1', 'query one', [0.5, 0.5, 0.5, 0.5], 'd2', 2.0, 2, [1, 2], [0.0, 1.0, 0.0, 3.0]],
            ['q1', 'query one', [0.5, 0.5, 0.5, 0.5], 'd3', 1.0, 3, [2, 0], [2.0, 2.0, 1.0, 1.0]],
            ['q2', 'query two', [1.0, 2.0, 3.0, 4.0], 'd1', 1.0, 1, [2, 2], [2.0, 2.0, 1.0, 1.0]],
        ], columns=['qid', 'query', 'query_vec', 'docno', 'score', 'rank', 'codes', 'doc_vec'])

    def _classes(self):
        return [
            (PQPrfTF, dict()),
            (PQPrfTFIDF, dict()),
            (PQPrfProbabilisticTFIDF, dict()),
            (PQPrfKLTFIDF, dict(smoothing=1e-6)),
        ]

    def test_transform_matches_direct_call(self):
        inp = self._input()
        for cls, extra in self._classes():
            with self.subTest(cls=cls.__name__):
                prf = cls(
                    centroids=self.centroids,
                    collection_code_freq=self.freq,
                    n_docs=self.n_docs,
                    alpha=0.4,
                    temperature=1.0,
                    k=2,
                    **extra,
                )
                out = prf(inp)

                self.assertEqual(
                    out.columns.tolist(), ['qid', 'query', 'query_vec']
                )
                self.assertEqual(len(out), 2)
                self.assertEqual(out['qid'].tolist(), ['q1', 'q2'])

                for i, qid in enumerate(['q1', 'q2']):
                    rows = inp[inp['qid'] == qid].sort_values('rank').head(2)
                    query = torch.as_tensor(rows['query_vec'].iloc[0])
                    scores = torch.as_tensor(rows['score'].to_numpy())
                    codes = torch.as_tensor(
                        np.stack(rows['codes'].to_list()), dtype=torch.long
                    )
                    expected = prf.pq_prf(
                        query,
                        scores,
                        codes,
                        torch.as_tensor(self.centroids),
                        torch.as_tensor(self.freq),
                        self.n_docs,
                        alpha=0.4,
                        temperature=1.0,
                    ).numpy()
                    np.testing.assert_allclose(
                        out['query_vec'].iloc[i], expected, rtol=1e-5, atol=1e-6
                    )

    def test_index_constructor_derives_collection_statistics(self):
        for cls, extra in self._classes():
            with self.subTest(cls=cls.__name__):
                prf = cls(self.index, alpha=0.5, **extra)
                np.testing.assert_array_equal(
                    prf.collection_code_freq.numpy(), self.freq
                )
                self.assertEqual(prf.n_docs, self.n_docs)
                self.assertEqual(len(prf(self._input())), 2)

    def test_empty_input(self):
        inp = pd.DataFrame(
            [], columns=['qid', 'query', 'query_vec', 'docno', 'score', 'rank', 'codes']
        )
        prf = PQPrfTF(self.index)
        out = prf(inp)
        self.assertEqual(
            out.columns.tolist(), ['qid', 'query', 'query_vec']
        )
        self.assertTrue(out.empty)

    def test_k_cutoff_via_compile(self):
        prf = PQPrfTF(self.index, k=1)
        self.assertIsNot(prf.compile(), prf)
        prf_no_k = PQPrfTF(self.index)
        self.assertIs(prf_no_k.compile(), prf_no_k)


if __name__ == '__main__':
    unittest.main()
