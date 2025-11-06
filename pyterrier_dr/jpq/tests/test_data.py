import unittest
import numpy as np
import torch

from pyterrier_dr.jpq.data import get_dataset, get_dataloader


class TestGetDataLoader(unittest.TestCase):
    def setUp(self):
        # Documents in the training pool
        self.docnos = ["A", "B", "C", "D", "E"]
        N, M = len(self.docnos), 4  # 5 docs, 4 PQ subcodes each

        # Deterministic PQ codes: codes[i] == [i, i+1, i+2, i+3]
        self.codes = np.vstack([np.arange(i, i + M, dtype=np.int64) for i in range(N)])

        # Map docno -> row index in codes
        self.docno2pos = {dn: i for i, dn in enumerate(self.docnos)}

        # Training pairs (the last one should be filtered out)
        self.docpairs = [
            {"query": "what is AI",              "doc_id_a": "A", "doc_id_b": "B"},
            {"query": "information retrieval",   "doc_id_a": "C", "doc_id_b": "D"},
            {"query": "machine learning",        "doc_id_a": "A", "doc_id_b": "E"},
            {"query": "deep learning",           "doc_id_a": "X", "doc_id_b": "Y"},  # filtered
        ]

    def test_loader_basic(self):
        ds = get_dataset(
            docpairs=self.docpairs,
            docnos=self.docnos,
            codes=self.codes,
            docno2pos=self.docno2pos,
        )
        dl = get_dataloader(ds, batch_size=2)

        all_queries = []
        all_pos = []
        all_neg = []

        for batch in dl:
            # Check presence of keys
            self.assertIn("query_text", batch)
            self.assertIn("pos_codes", batch)
            self.assertIn("neg_codes", batch)

            # Check types/shapes
            self.assertIsInstance(batch["query_text"], list)
            self.assertIsInstance(batch["pos_codes"], torch.Tensor)
            self.assertIsInstance(batch["neg_codes"], torch.Tensor)
            self.assertEqual(batch["pos_codes"].dtype, torch.long)
            self.assertEqual(batch["neg_codes"].dtype, torch.long)
            self.assertEqual(batch["pos_codes"].shape[1], self.codes.shape[1])
            self.assertEqual(batch["neg_codes"].shape[1], self.codes.shape[1])

            # Accumulate for exact-value checks
            all_queries.extend(batch["query_text"])
            all_pos.append(batch["pos_codes"])
            all_neg.append(batch["neg_codes"])

        # One pair was invalid -> total of 3 kept
        self.assertEqual(len(all_queries), 3)

        # Concatenate across batches to compare with expected codes
        pos_all = torch.cat(all_pos, dim=0).cpu().numpy()
        neg_all = torch.cat(all_neg, dim=0).cpu().numpy()

        # Build the expected order after filtering:
        kept = [p for p in self.docpairs if p["doc_id_a"] in self.docno2pos and p["doc_id_b"] in self.docno2pos]
        self.assertEqual(len(kept), 3)

        expected_pos = np.vstack([self.codes[self.docno2pos[p["doc_id_a"]]] for p in kept])
        expected_neg = np.vstack([self.codes[self.docno2pos[p["doc_id_b"]]] for p in kept])

        # Order might differ if shuffle=True in get_dataloader; ensure it’s deterministic
        # If your get_dataloader sets shuffle=True, pass shuffle=False in the call above.

        np.testing.assert_array_equal(pos_all, expected_pos)
        np.testing.assert_array_equal(neg_all, expected_neg)

        # Also verify the queries line up
        expected_queries = [p["query"] for p in kept]
        self.assertEqual(all_queries, expected_queries)


if __name__ == "__main__":
    unittest.main()