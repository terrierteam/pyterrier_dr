import unittest
import numpy as np
import pandas as pd
from pyterrier_dr import AveragePrf


class TestModels(unittest.TestCase):

    def test_avg_prf(self):
        prf = AveragePrf()
        inp = pd.DataFrame([['q1', 'query', np.array([1, 2, 3]), 'd1', np.array([4, 5, 6])]], columns=['qid', 'query', 'query_vec', 'docno', 'doc_vec'])
        out = prf(inp)
        self.assertEqual(out.columns.tolist(), ['qid', 'query', 'query_vec'])
        self.assertEqual(len(out), 1)
        self.assertEqual(out['qid'][0], 'q1')
        self.assertEqual(out['query'][0], 'query')
        np.testing.assert_array_equal(out['query_vec'][0], np.array([2.5, 3.5, 4.5]))


if __name__ == '__main__':
    unittest.main()
