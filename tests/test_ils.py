import unittest
import tempfile
import numpy as np
import pandas as pd
from pyterrier_dr import ils, FlexIndex


class TestIls(unittest.TestCase):
    def test_ils_basic(self):
        results = pd.DataFrame([
            ['q0', 'd0', np.array([0, 1, 0])],
            ['q0', 'd1', np.array([0, 1, 1])],
            ['q0', 'd2', np.array([1, 1, 0])],
            ['q0', 'd3', np.array([1, 1, 1])],
            ['q1', 'd0', np.array([0, 1, 0])],
            ['q2', 'd0', np.array([0, 1, 0])],
            ['q2', 'd1', np.array([0, 1, 1])],
        ], columns=['qid', 'docno', 'doc_vec'])
        result = dict(ils(results))
        self.assertAlmostEqual(result['q0'], 0.6874, places=3)
        self.assertAlmostEqual(result['q1'], 0.0000, places=3)
        self.assertAlmostEqual(result['q2'], 0.7071, places=3)

    def test_ils_vec_from_index(self):
        with tempfile.TemporaryDirectory() as d:
            index = FlexIndex(f'{d}/index.flex')
            index.index([
                {'docno': 'd0', 'doc_vec': np.array([0, 1, 0])},
                {'docno': 'd1', 'doc_vec': np.array([0, 1, 1])},
                {'docno': 'd2', 'doc_vec': np.array([1, 1, 0])},
                {'docno': 'd3', 'doc_vec': np.array([1, 1, 1])},
            ])
            results = pd.DataFrame([
                ['q0', 'd0'],
                ['q0', 'd1'],
                ['q0', 'd2'],
                ['q0', 'd3'],
                ['q1', 'd0'],
                ['q2', 'd0'],
                ['q2', 'd1'],
            ], columns=['qid', 'docno'])
            result = dict(ils(results, index))
            self.assertAlmostEqual(result['q0'], 0.6874, places=3)
            self.assertAlmostEqual(result['q1'], 0.0000, places=3)
            self.assertAlmostEqual(result['q2'], 0.7071, places=3)

    def test_ils_measure_from_index(self):
        with tempfile.TemporaryDirectory() as d:
            index = FlexIndex(f'{d}/index.flex')
            index.index([
                {'docno': 'd0', 'doc_vec': np.array([0, 1, 0])},
                {'docno': 'd1', 'doc_vec': np.array([0, 1, 1])},
                {'docno': 'd2', 'doc_vec': np.array([1, 1, 0])},
                {'docno': 'd3', 'doc_vec': np.array([1, 1, 1])},
            ])
            results = pd.DataFrame([
                ['q0', 'd0'],
                ['q0', 'd1'],
                ['q0', 'd2'],
                ['q0', 'd3'],
                ['q1', 'd0'],
                ['q2', 'd0'],
                ['q2', 'd1'],
            ], columns=['query_id', 'doc_id'])
            qrels = pd.DataFrame(columns=['query_id', 'doc_id', 'relevance']) # qrels ignored
            result = index.ILS.calc(qrels, results) 
            self.assertAlmostEqual(result.aggregated, 0.4648, places=3)
            self.assertEqual(3, len(result.per_query))
            self.assertEqual(result.per_query[0].query_id, 'q0')
            self.assertAlmostEqual(result.per_query[0].value, 0.6874, places=3)
            self.assertEqual(result.per_query[1].query_id, 'q1')
            self.assertAlmostEqual(result.per_query[1].value, 0.0000, places=3)
            self.assertEqual(result.per_query[2].query_id, 'q2')
            self.assertAlmostEqual(result.per_query[2].value, 0.7071, places=3)

    def test_ils_measure_from_index_cutoff(self):
        with tempfile.TemporaryDirectory() as d:
            index = FlexIndex(f'{d}/index.flex')
            index.index([
                {'docno': 'd0', 'doc_vec': np.array([0, 1, 0])},
                {'docno': 'd1', 'doc_vec': np.array([0, 1, 1])},
                {'docno': 'd2', 'doc_vec': np.array([1, 1, 0])},
                {'docno': 'd3', 'doc_vec': np.array([1, 1, 1])},
            ])
            results = pd.DataFrame([
                ['q0', 'd0'],
                ['q0', 'd1'],
                ['q0', 'd2'],
                ['q0', 'd3'],
                ['q1', 'd0'],
                ['q2', 'd0'],
                ['q2', 'd1'],
            ], columns=['query_id', 'doc_id'])
            qrels = pd.DataFrame(columns=['query_id', 'doc_id', 'relevance']) # qrels ignored
            result = (index.ILS@2).calc(qrels, results) 
            self.assertAlmostEqual(result.aggregated, 0.4714, places=3)
            self.assertEqual(3, len(result.per_query))
            self.assertEqual(result.per_query[0].query_id, 'q0')
            self.assertAlmostEqual(result.per_query[0].value, 0.7071, places=3)
            self.assertEqual(result.per_query[1].query_id, 'q1')
            self.assertAlmostEqual(result.per_query[1].value, 0.0000, places=3)
            self.assertEqual(result.per_query[2].query_id, 'q2')
            self.assertAlmostEqual(result.per_query[2].value, 0.7071, places=3)


if __name__ == '__main__':
    unittest.main()
