
import unittest
import pyterrier as pt
import pyterrier_dr
import numpy as np

from pyterrier_dr.jpq.retriever import build_inverted_index#, build_inverted_index_fast

class TestJPQ(unittest.TestCase):
    def test_jpq(self):

        M=4
        k=4
        codes_sel = np.array([
            [0,0,0,0],
            [0,0,0,3],
            [0,2,0,3],
        ])
        for docid in range(codes_sel.shape[0]):
            self.assertEqual(codes_sel[docid].shape[0], M)
            for m in range(M):
                self.assertTrue(codes_sel[docid][m] >= 0)
                self.assertTrue(codes_sel[docid][m] < k)

        inv = build_inverted_index(codes_sel,..., ..., ..., k=k)
        print(inv)
        self.assertEqual(M*k, inv.shape[0])
        self.assertEqual(3, inv.shape[1]) # no subid appears more than 3 times
        self.assertEqual(inv[0].tolist(),[0,1,2])
        self.assertEqual(inv[1].tolist(),[-1,-1,-1])

        # in second split, the first subid is used in docids 0 and 1
        self.assertEqual(inv[4].tolist(),[-1,0,1])
        # in second split, the third subid is used in docid 2
        self.assertEqual(inv[6].tolist(),[-1,-1,2])
        # in thhird split, the first subid is used in docid 0, 1, 2
        self.assertEqual(inv[8].tolist(),[0,1,2])
        # etc
        