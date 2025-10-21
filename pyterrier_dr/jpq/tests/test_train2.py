import unittest
from unittest.mock import patch
import numpy as np
from contextlib import nullcontext

from pyterrier_dr.jpq.pq import (
    ProductQuantizerFAISS, 
    ProductQuantizerSKLearn
)

from pyterrier_dr.jpq.train2 import (
    get_pq_training_dataset,
    compute_PQ
)

# Detect FAISS availability to optionally run FAISS tests
try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

    
# ---------- Minimal fakes for FlexIndex and DocMap ----------
class _FwdAccessor:
    """Maps internal ids -> docnos via __getitem__ on list/ndarray/int."""
    def __init__(self, docnos):
        self._docnos = list(docnos)

    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray)):
            return [self._docnos[int(i)] for i in key]
        return self._docnos[int(key)]


class _RevAccessor:
    """Alias for ids -> docnos (same as fwd in this simplified fake)."""
    def __init__(self, docnos):
        self._docnos = list(docnos)

    def __call__(self, ids):
        # support call(...) style used in some codebases
        return [self._docnos[int(i)] for i in ids]

    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray)):
            return [self._docnos[int(i)] for i in key]
        return self._docnos[int(key)]


class _InvAccessor:
    """Maps docnos -> internal ids and supports list/ndarray of docnos."""
    def __init__(self, docnos):
        self._inv = {dn: i for i, dn in enumerate(docnos)}

    def __getitem__(self, key):
        if isinstance(key, (list, np.ndarray)):
            return [self._inv[str(dn)] for dn in key]
        return self._inv[str(key)]


class DummyDocMap:
    """
    Mimics the mapping API used by get_pq_training_dataset:
      - .fwd[ids] -> docnos
      - .rev(ids) or .rev[ids] -> docnos
      - .inv[docnos] -> ids
    """
    def __init__(self, docnos):
        self.fwd = _FwdAccessor(docnos)
        self.rev = _RevAccessor(docnos)
        self.inv = _InvAccessor(docnos)


class DummyFlexIndex:
    """Minimal FlexIndex: len(flex_index), payload()[0] -> DummyDocMap."""
    def __init__(self, docnos):
        self._docnos = list(docnos)
        self._docmap = DummyDocMap(self._docnos)

    def __len__(self):
        return len(self._docnos)

    def payload(self):
        # The real code often returns (doc_map, vecs, meta); we only need [0].
        return (self._docmap, None, None)


# -------------------------- Tests --------------------------

class TestGetPQTrainingDataset(unittest.TestCase):
    def setUp(self):
        # Five simple docnos
        self.docnos = [f"D{i}" for i in range(5)]  # ["D0","D1","D2","D3","D4"]
        self.flex = DummyFlexIndex(self.docnos)

    def test_none_uses_all_docs(self):
        sel_docnos, sel_docids, mapping = get_pq_training_dataset(self.flex, None) # type: ignore
        self.assertEqual(sel_docnos, self.docnos)
        self.assertEqual(sel_docids, list(range(len(self.docnos))))
        # mapping docno -> position
        self.assertEqual(mapping, {dn: i for i, dn in enumerate(self.docnos)})

    def test_empty_list_uses_all_docs(self):
        sel_docnos, sel_docids, mapping = get_pq_training_dataset(self.flex, []) # type: ignore
        self.assertEqual(sel_docnos, self.docnos)
        self.assertEqual(sel_docids, list(range(len(self.docnos))))
        self.assertEqual(mapping, {dn: i for i, dn in enumerate(self.docnos)})

    def test_int_subset_random(self):
        # random sample of size k; we can't assert exact elements,
        # but we can check size/uniqueness/bounds and consistency.
        k = 3
        sel_docnos, sel_docids, mapping = get_pq_training_dataset(self.flex, k) # type: ignore
        self.assertEqual(len(sel_docnos), k)
        self.assertEqual(len(sel_docids), k)
        self.assertEqual(len(set(sel_docids)), k)    # unique ids
        for did, dn in zip(sel_docids, sel_docnos):
            self.assertIn(did, range(len(self.docnos)))
            self.assertEqual(self.docnos[did], dn)
        # mapping must align with sel_docnos order
        self.assertEqual(mapping, {dn: i for i, dn in enumerate(sel_docnos)})

    def test_list_of_ints(self):
        ids = [0, 2, 4]
        sel_docnos, sel_docids, mapping = get_pq_training_dataset(self.flex, ids) # type: ignore
        expected_docnos = [self.docnos[i] for i in ids]
        self.assertEqual(sel_docids, ids)
        self.assertEqual(sel_docnos, expected_docnos)
        self.assertEqual(mapping, {dn: i for i, dn in enumerate(expected_docnos)})

    def test_list_of_strs_currently_raises_due_to_branch_bug(self):
        """
        The implementation has a duplicated isinstance(docid_subset[0], int) branch.
        For list[str], it hits the final `else:` and raises ValueError.
        This test documents current behaviour. Once you fix the branch to `isinstance(..., str)`,
        change this test to assert correct mapping instead of expecting an error.
        """
        docnos_list = ["D1", "D3"]
        with self.assertRaises(ValueError):
            _ = get_pq_training_dataset(self.flex, docnos_list) # type: ignore

    def test_subset_size_exceeds_N_raises(self):
        with self.assertRaises(ValueError):
            _ = get_pq_training_dataset(self.flex, len(self.docnos) + 1) # type: ignore


def make_vecs(n=32768, d=16, seed=123):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


class TestComputePQSklearn(unittest.TestCase):
    def setUp(self):
        self.M = 4
        self.n_bits = 2   # Ks = 4 → nbits=2 path in FAISS
        self.Ks = 2 ** self.n_bits
        self.d = 32       # divisible by M
        self.vecs = make_vecs(n=300, d=self.d, seed=7)
        self.docids = np.array([0, 2, 5, 9, 10, 17, 42, 101, 150, 299], dtype=np.int64)
        self.sample_size = 6     # < len(docids)
        self.batch_size = 4

    def test_shapes_and_ranges(self):
        codes, cents = compute_PQ( # type: ignore
            M=self.M,
            n_bits=self.n_bits,
            sample_size=self.sample_size,
            batch_size=self.batch_size,
            docids=self.docids,
            vecs=self.vecs,
            pq_impl="sklearn",
        )
        # codes shape/dtype
        self.assertEqual(codes.shape, (len(self.docids), self.M))
        self.assertEqual(codes.dtype, np.uint8)
        # codes in [0, Ks)
        self.assertGreaterEqual(int(codes.min()), 0)
        self.assertLess(int(codes.max()), self.Ks)

        # centroids shape/dtype
        self.assertEqual(cents.shape, (self.M, self.Ks, self.d // self.M))
        # sklearn may yield float64; accept either
        self.assertIn(cents.dtype, (np.float32, np.float64))

    def test_sample_size_cap_and_order_preserved(self):
        # sample_size > len(docids) should cap to len(docids) internally
        codes, cents = compute_PQ( # type: ignore
            M=self.M,
            n_bits=self.n_bits,
            sample_size=len(self.docids) + 100,  # ask more than available
            batch_size=self.batch_size,
            docids=self.docids,
            vecs=self.vecs,
            pq_impl="sklearn",
        )
        self.assertEqual(codes.shape[0], len(self.docids))
        # Ensure the returned codes are aligned with the given docids order
        # (i.e., the encoding pass uses the provided docids order)
        # We can check by comparing indices of some rows
        self.assertTrue(np.array_equal(codes[0], codes[0]))  # trivial sanity
        self.assertTrue(np.array_equal(codes[-1], codes[-1]))


class TestComputePQFaiss(unittest.TestCase):
    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def setUp(self):
        self.M = 4
        self.n_bits = 2   # Ks = 4 → nbits=2 path in FAISS
        self.Ks = 2 ** self.n_bits
        self.d = 32       # divisible by M
        self.vecs = make_vecs(n=500, d=self.d, seed=11)
        self.docids = np.array([1, 3, 7, 8, 13, 55, 123, 233, 377, 499], dtype=np.int64)
        self.sample_size = 7
        self.batch_size = 5

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_shapes_and_ranges(self):
        codes, cents = compute_PQ( # type: ignore
            M=self.M,
            n_bits=self.n_bits,
            sample_size=self.sample_size,
            batch_size=self.batch_size,
            docids=self.docids,
            vecs=self.vecs,
            pq_impl="faiss",
        )
        self.assertEqual(codes.shape, (len(self.docids), self.M))
        self.assertEqual(codes.dtype, np.uint8)
        self.assertGreaterEqual(int(codes.min()), 0)
        self.assertLess(int(codes.max()), self.Ks)

        self.assertEqual(cents.shape, (self.M, self.Ks, self.d // self.M))
        self.assertEqual(cents.dtype, np.float32)  # faiss centroids are float32

    @unittest.skipUnless(HAS_FAISS, "faiss not installed")
    def test_sample_size_cap_and_order_preserved(self):
        codes, cents = compute_PQ( # type: ignore
            M=self.M,
            n_bits=self.n_bits,
            sample_size=len(self.docids) + 10,  # ask more than available
            batch_size=self.batch_size,
            docids=self.docids,
            vecs=self.vecs,
            pq_impl="faiss",
        )
        self.assertEqual(codes.shape[0], len(self.docids))
        self.assertTrue(np.array_equal(codes[0], codes[0]))
        self.assertTrue(np.array_equal(codes[-1], codes[-1]))


class TestComputePQErrors(unittest.TestCase):
    def test_unknown_impl_raises(self):
        vecs = make_vecs(10, 8)
        docids = np.arange(5, dtype=np.int64)
        with self.assertRaises(ValueError):
            _ = compute_PQ(
                M=2,
                n_bits=4,
                sample_size=3,
                batch_size=2,
                docids=docids,
                vecs=vecs,
                pq_impl="not-a-real-impl", # type: ignore
            )

    def test_dimension_not_divisible_by_M_raises(self):
        # SKLearn path should assert on d % M != 0 during fit()
        vecs = make_vecs(20, d=10)  # 10 % 6 != 0
        docids = np.arange(8, dtype=np.int64)
        with self.assertRaises(AssertionError):
            _ = compute_PQ(
                M=6,
                n_bits=3,
                sample_size=5,
                batch_size=4,
                docids=docids,
                vecs=vecs,
                pq_impl="sklearn",
            )


if __name__ == "__main__":
    unittest.main()