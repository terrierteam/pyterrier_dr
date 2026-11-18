"""Tests for pyterrier_dr.repconc.sinkhorn.

These tests verify the Sinkhorn-Knopp constrained clustering assignment
(paper Eq. 19) that drives RepCONC's uniform-usage quantisation:

* per-document marginals sum to exactly 1,
* per-centroid marginals equal B/K (within the iteration tolerance),
* hard codes are in range and recover well-separated clusters,
* the 3D [B, M, K] vectorisation matches the per-subspace 2D solution,
* the soft solution agrees with the linear-programming optimum of the
  constrained clustering problem (Eq. 19) when the cost matrix is small,
* the numerical stabilisation (center_distance) keeps values in range.

All tests run on CPU.
"""
import unittest

import torch

from pyterrier_dr.repconc.sinkhorn import center_distance, sinkhorn, sinkhorn_assignments


class TestCenterDistance(unittest.TestCase):

    def test_range_and_order(self):
        torch.manual_seed(0)
        d = torch.rand(64, 32) * 10 + 3
        c = center_distance(d)
        # values are centred/scaled to (approximately) [-1, 1]
        self.assertGreaterEqual(c.min().item(), -1.0001)
        self.assertLessEqual(c.max().item(), 1.0001)
        # the transformation is monotonic in each (b, :) row, so it must
        # preserve the ordering of distances
        for b in range(64):
            self.assertTrue(torch.equal(d[b].argsort(), c[b].argsort()))

    def test_per_subspace_3d(self):
        # each subspace must be normalised independently: the centred values
        # of subspace m must equal (d[:, m] - middle_m) / amplitude_m where
        # middle_m / amplitude_m depend only on subspace m, not on the doc
        torch.manual_seed(1)
        d = torch.rand(32, 4, 16)
        d[:, 1] *= 100.0  # one subspace has a very different scale
        c = center_distance(d)
        for m in range(4):
            max_m = d[:, m].max()
            min_m = d[:, m].min()
            middle = (max_m + min_m) / 2
            amplitude = max_m - middle + 1e-5
            torch.testing.assert_close(c[:, m], (d[:, m] - middle) / amplitude, rtol=0, atol=1e-12)
            self.assertGreaterEqual(c[:, m].min().item(), -1.0001)
            self.assertLessEqual(c[:, m].max().item(), 1.0001)


class TestSinkhornMarginals(unittest.TestCase):

    def _check_marginals(self, Q, B, K, marginal_tol=1e-3):
        """Per-doc marginals must be exactly 1; per-centroid ~= B/K."""
        self.assertFalse(torch.isnan(Q).any())
        self.assertFalse(torch.isinf(Q).any())
        self.assertLessEqual((Q < 0).sum().item(), 0)  # Q >= 0
        self.assertEqual(Q.shape, (B, K))
        row_sums = Q.sum(dim=-1)
        col_sums = Q.sum(dim=0)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=0, atol=1e-9)
        target = B / K
        torch.testing.assert_close(col_sums, torch.full_like(col_sums, target), rtol=0, atol=marginal_tol)

    def test_divisible_batch(self):
        torch.manual_seed(42)
        d = torch.rand(256, 256) ** 2
        Q = sinkhorn(center_distance(d))
        self._check_marginals(Q, B=256, K=256)

    def test_non_divisible_batch(self):
        # B not divisible by K: per-centroid target is still B/K, in expectation
        torch.manual_seed(42)
        d = torch.rand(300, 256) ** 2
        Q = sinkhorn(center_distance(d))
        self._check_marginals(Q, B=300, K=256)

    def test_small_batch_below_K(self):
        # more centroids than documents: each centroid should get <= 1 on average
        torch.manual_seed(7)
        d = torch.rand(4, 256) ** 2
        Q = sinkhorn(center_distance(d))
        self._check_marginals(Q, B=4, K=256, marginal_tol=1e-2)
        self.assertLessEqual(Q.sum(dim=0).max().item(), 1.0 + 1e-2)

    def test_single_document(self):
        torch.manual_seed(3)
        d = torch.rand(1, 8) ** 2
        Q = sinkhorn(center_distance(d))
        self._check_marginals(Q, B=1, K=8)
        # a single document must be fully assigned to a single centroid
        self.assertEqual(Q.argmax().item(), d.squeeze(0).argmin().item())

    def test_empty_batch(self):
        Q = sinkhorn(torch.empty(0, 1, 8))
        self.assertEqual(tuple(Q.shape), (0, 1, 8))
        Q = sinkhorn(torch.empty(0, 8))
        self.assertEqual(tuple(Q.shape), (0, 8))

    def test_invalid_arguments(self):
        d = torch.rand(8, 8)
        with self.assertRaises(ValueError):
            sinkhorn(d, epsilon=0.0)
        with self.assertRaises(ValueError):
            sinkhorn(d, iters=-1)
        with self.assertRaises(ValueError):
            sinkhorn(torch.rand(2, 3, 4, 5))


class TestSinkhornAssignments(unittest.TestCase):

    def test_codes_in_range_2d(self):
        torch.manual_seed(0)
        d = torch.rand(128, 256) ** 2
        codes = sinkhorn_assignments(d)
        self.assertEqual(tuple(codes.shape), (128,))
        self.assertGreaterEqual(codes.min().item(), 0)
        self.assertLess(codes.max().item(), 256)

    def test_codes_in_range_3d(self):
        torch.manual_seed(0)
        d = torch.rand(128, 4, 256) ** 2
        codes = sinkhorn_assignments(d)
        self.assertEqual(tuple(codes.shape), (128, 4))
        self.assertGreaterEqual(codes.min().item(), 0)
        self.assertLess(codes.max().item(), 256)

    def test_3d_matches_2d_per_subspace(self):
        torch.manual_seed(5)
        B, M, K = 64, 4, 32
        d = torch.rand(B, M, K) ** 2
        codes3d = sinkhorn_assignments(d)
        for m in range(M):
            codes2d = sinkhorn_assignments(d[:, m])
            torch.testing.assert_close(codes3d[:, m], codes2d)

    def test_recovers_true_clusters(self):
        # well-separated clusters (4-D hypercube vertices, min distance 1.0,
        # noise 0.05) with uniform cluster sizes (B/K per cluster):
        # the constrained optimum must coincide with the (unconstrained)
        # nearest-centroid assignment
        torch.manual_seed(11)
        K, B = 16, 512
        bits = torch.arange(K).unsqueeze(1).bitwise_right_shift(torch.arange(4)) & 1
        centroids = bits.to(torch.float64)  # 16 vertices of the 4-cube
        pair_dist = (centroids[:, None, :] - centroids[None]).norm(dim=-1)
        pair_dist.fill_diagonal_(float("inf"))
        self.assertGreaterEqual(pair_dist.amin().item(), 0.99)
        labels = torch.arange(B) % K  # exactly B/K documents per cluster
        docs = centroids[labels] + torch.randn(B, 4) * 0.05
        d = (docs.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(-1)  # [B, K]
        codes = sinkhorn_assignments(d)
        correct = (codes == labels).float().mean().item()
        self.assertGreaterEqual(correct, 0.99,
                                f"only {correct:.1%} of documents assigned to their true cluster")

    def test_constraint_spreads_usage(self):
        # without the constraint, one attractive centroid would swallow most
        # documents; the constraint (Eq. 19) equalises the *soft* assignment
        # marginals. NB: the constraint acts on the soft assignment Q (whose
        # column marginals are ~= B/K); the hard argmax codes are not
        # themselves uniform (the official implementation has the same
        # property), but they must improve substantially.
        torch.manual_seed(13)
        B, K = 512, 16
        centroids = torch.rand(K, 4) * 10
        centroids[0] *= 0.01  # make one centroid close to (many) documents
        docs = torch.randn(B, 4) * 5
        d = (docs.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(-1)
        Q = sinkhorn(center_distance(d))
        col_sums = Q.sum(dim=0)
        target = B / K
        rel_dev = ((col_sums - target) / target).abs().max().item()
        self.assertLess(rel_dev, 0.25,
                        f"soft per-centroid marginals deviate up to {rel_dev:.1%} from B/K")
        unconstrained_usage = (d.argmin(dim=-1) == 0).float().mean().item()
        constrained_usage = sinkhorn_assignments(d).eq(0).float().mean().item()
        self.assertGreater(unconstrained_usage, 0.5)
        self.assertLess(constrained_usage, unconstrained_usage / 2,
                        f"hard usage of the attractive centroid did not improve "
                        f"({constrained_usage:.1%} vs unconstrained {unconstrained_usage:.1%})")

    def test_empty(self):
        codes = sinkhorn_assignments(torch.empty(0, 8))
        self.assertEqual(tuple(codes.shape), (0,))
        codes = sinkhorn_assignments(torch.empty(0, 3, 8))
        self.assertEqual(tuple(codes.shape), (0, 3))

    def test_invalid_dim(self):
        with self.assertRaises(ValueError):
            sinkhorn_assignments(torch.rand(2, 3, 4, 5))

    def test_deterministic(self):
        torch.manual_seed(21)
        d = torch.rand(64, 32) ** 2
        torch.testing.assert_close(sinkhorn_assignments(d), sinkhorn_assignments(d))


class TestSinkhornAgainstLinearProgram(unittest.TestCase):
    """Compare the Sinkhorn solution against the LP optimum of Eq. 19.

    min_q  sum_{b,k} q[b,k] * cost[b,k]
    s.t.   sum_k q[b,k] = 1        (each document fully assigned)
           sum_b q[b,k] = B / K    (uniform centroid usage)
           q >= 0
    """

    def test_agrees_with_lp(self):
        try:
            from scipy.optimize import linprog
        except ImportError:
            self.skipTest("scipy not available")
        torch.manual_seed(99)
        B, K = 16, 8
        cost = torch.rand(B, K) * 4  # raw squared distances
        centered = center_distance(cost)
        Q = sinkhorn(centered)
        sinkhorn_obj = float((Q * centered).sum())

        c = centered.reshape(-1).double().numpy()
        A_eq, b_eq = [], []
        # row constraints: each document's mass sums to 1
        for b in range(B):
            row = [0.0] * (B * K)
            for k in range(K):
                row[b * K + k] = 1.0
            A_eq.append(row)
            b_eq.append(1.0)
        # column constraints: each centroid's mass sums to B/K
        for k in range(K):
            row = [0.0] * (B * K)
            for b in range(B):
                row[b * K + k] = 1.0
            A_eq.append(row)
            b_eq.append(B / K)
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0.0, None)] * (B * K), method="highs")
        self.assertTrue(res.success, res.message)
        lp_obj = float(res.fun)
        # NB: the centred cost takes negative values, so the objective can be negative
        rel_diff = abs(sinkhorn_obj - lp_obj) / max(abs(lp_obj), 1e-9)
        self.assertLess(rel_diff, 0.02,
                        f"sinkhorn objective {sinkhorn_obj:.6f} deviates {rel_diff:.2%} from LP optimum {lp_obj:.6f}")


if __name__ == "__main__":
    unittest.main()
