import unittest
import numpy as np
import pandas as pd
from pyterrier_dr import AveragePrf, VectorPrf


class TestModels(unittest.TestCase):

    def test_average_prf(self):
        prf = AveragePrf()
        with self.subTest('single row'):
            inp = pd.DataFrame([['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])]], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            self.assertEqual(out['qid'].iloc[0], 'q1')
            self.assertEqual(out['query'].iloc[0], 'query')
            np.testing.assert_array_equal(out['query_vec'].iloc[0], np.array([2.5, 3.5, 4.5]))

        with self.subTest('multiple rows'):
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd2', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd3', np.array([8, 7, 1])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            np.testing.assert_array_equal(out['query_vec'].iloc[0], np.array([3.5, 4.5, 3.]))

        with self.subTest('multiple rows -- k=3'):
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd2', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd3', np.array([8, 7, 1])],
                ['q1', 'query', np.array([1, 2, 3]), 'd4', np.array([100, 100, 100])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            np.testing.assert_array_equal(out['query_vec'].iloc[0], np.array([3.5, 4.5, 3.]))

        with self.subTest('multiple rows -- k=1'):
            prf.k = 1
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd2', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd3', np.array([8, 7, 1])],
                ['q1', 'query', np.array([1, 2, 3]), 'd4', np.array([100, 100, 100])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            np.testing.assert_array_equal(out['query_vec'].iloc[0], np.array([2.5, 3.5, 4.5]))

        with self.subTest('multiple queries'):
            prf.k = 3
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([8, 7, 1])],
                ['q2', 'query2', np.array([4, 6, 1]), 'd1', np.array([9, 4, 2])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 2)
            self.assertEqual(out['qid'].iloc[0], 'q1')
            np.testing.assert_array_equal(out['query_vec'].iloc[0], np.array([3.5, 4.5, 3.]))
            self.assertEqual(out['qid'].iloc[1], 'q2')
            np.testing.assert_array_equal(out['query_vec'].iloc[1], np.array([6.5, 5., 1.5]))

    def test_vector_prf(self):
        prf = VectorPrf(alpha=0.5, beta=0.5)
        with self.subTest('single row'):
            inp = pd.DataFrame([['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])]], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            self.assertEqual(out['qid'].iloc[0], 'q1')
            self.assertEqual(out['query'].iloc[0], 'query')
            np.testing.assert_array_equal(out['query_vec'].iloc[0], np.array([2.5, 3.5, 4.5]))

        with self.subTest('multiple rows'):
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd2', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd3', np.array([8, 7, 1])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            np.testing.assert_almost_equal(out['query_vec'].iloc[0], np.array([2.666667, 3.666667, 3.]), decimal=5)

        with self.subTest('multiple rows -- k=3'):
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd2', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd3', np.array([8, 7, 1])],
                ['q1', 'query', np.array([1, 2, 3]), 'd4', np.array([100, 100, 100])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            np.testing.assert_almost_equal(out['query_vec'].iloc[0], np.array([2.666667, 3.666667, 3.]), decimal=5)

        with self.subTest('multiple rows -- k=1'):
            prf.k = 1
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd2', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd3', np.array([8, 7, 1])],
                ['q1', 'query', np.array([1, 2, 3]), 'd4', np.array([100, 100, 100])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 1)
            np.testing.assert_array_equal(out['query_vec'].iloc[0], np.array([2.5, 3.5, 4.5]))

        with self.subTest('multiple queries'):
            prf.k = 3
            inp = pd.DataFrame([
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])],
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([1, 4, 2])],
                ['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([8, 7, 1])],
                ['q2', 'query2', np.array([4, 6, 1]), 'd1', np.array([9, 4, 2])],
            ], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
            out = prf(inp)
            self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
            self.assertEqual(len(out), 2)
            self.assertEqual(out['qid'].iloc[0], 'q1')
            np.testing.assert_almost_equal(out['query_vec'].iloc[0], np.array([2.666667, 3.666667, 3.]), decimal=5)
            self.assertEqual(out['qid'].iloc[1], 'q2')
            np.testing.assert_array_equal(out['query_vec'].iloc[1], np.array([6.5, 5., 1.5]))



if __name__ == '__main__':
    unittest.main()
