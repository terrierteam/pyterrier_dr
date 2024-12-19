import unittest
import numpy as np
import pandas as pd
from pyterrier_dr import MmrScorer


class TestMmr(unittest.TestCase):
    def test_mmr(self):
        mmr = MmrScorer()
        results = mmr(pd.DataFrame([
            ['q0', 'd0', 1.0, np.array([0, 1, 0])],
            ['q0', 'd1', 0.5, np.array([0, 1, 1])],
            ['q0', 'd2', 0.5, np.array([1, 1, 1])],
            ['q0', 'd3', 0.1, np.array([1, 1, 0])],
            ['q1', 'd0', 0.6, np.array([0, 1, 0])],
            ['q2', 'd0', 0.4, np.array([0, 1, 0])],
            ['q2', 'd1', 0.3, np.array([0, 1, 1])],
        ], columns=['qid', 'docno', 'score', 'doc_vec']))
        pd.testing.assert_frame_equal(results, pd.DataFrame([
            ['q0', 'd0', 0.0, 0],
            ['q0', 'd2', -1.0, 1],
            ['q0', 'd1', -2.0, 2],
            ['q0', 'd3', -3.0, 3],
            ['q1', 'd0', 0.0, 0],
            ['q2', 'd0', 0.0, 0],
            ['q2', 'd1', -1.0, 1],
        ], columns=['qid', 'docno', 'score', 'rank']))


if __name__ == '__main__':
    unittest.main()
