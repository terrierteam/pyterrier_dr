"""Tests for pyterrier_dr.repconc.losses.

Includes analytical verification of the RepCONC gradient policy (paper
Eqs. 9-10):

* document embedding gradient  =  dL_r/dd_hat + lambda * dL_m/dd
* centroid gradient            =  scatter of (dL_r/dd_hat + lambda * dL_m/dd_hat)
* query gradient               =  dL_r/dq (as for an ordinary contrastive loss)
"""
import unittest

import torch
import torch.nn.functional as F

from pyterrier_dr.repconc.losses import (
    StraightThrough,
    clustering_loss,
    paper_lambda,
    ranking_loss,
    straight_through,
)
from pyterrier_dr.repconc.pq import RepCONCPQ


class TestStraightThrough(unittest.TestCase):

    def test_forward_returns_quantized(self):
        a = torch.randn(4, 6)
        b = torch.randn(4, 6)
        out = straight_through(a, b)
        torch.testing.assert_close(out, b)

    def test_backward_passes_grad_to_both(self):
        a = torch.randn(4, 6, requires_grad=True)
        b = torch.randn(4, 6, requires_grad=True)
        out = straight_through(a, b)
        g = torch.randn(4, 6)
        out.backward(g)
        torch.testing.assert_close(a.grad, g)
        torch.testing.assert_close(b.grad, g)


class TestClusteringLoss(unittest.TestCase):

    def test_value(self):
        d = torch.randn(5, 8)
        d_hat = torch.randn(5, 8)
        expected = ((d_hat - d) ** 2).sum(-1).mean()
        torch.testing.assert_close(clustering_loss(d_hat, d), expected)

    def test_grads(self):
        d = torch.randn(5, 8, requires_grad=True)
        d_hat = torch.randn(5, 8, requires_grad=True)
        clustering_loss(d_hat, d).backward()
        B, D = d.shape
        torch.testing.assert_close(d_hat.grad, 2 * (d_hat - d) / B)
        torch.testing.assert_close(d.grad, -2 * (d_hat - d) / B)


class TestRankingLoss(unittest.TestCase):

    def test_matches_cross_entropy(self):
        torch.manual_seed(0)
        sims = torch.randn(4, 12)
        labels = torch.tensor([3, 7, 1, 11])
        torch.testing.assert_close(ranking_loss(sims, labels), F.cross_entropy(sims, labels))

    def test_hardneg_topk_masking(self):
        torch.manual_seed(1)
        B, N = 2, 6
        sims = torch.randn(B, N)
        labels = torch.tensor([2, 4])
        k = 2
        # hand-compute the official masking:
        mask = torch.ones(B, N)
        neg = sims.detach().clone()
        neg.scatter_(1, labels.unsqueeze(1), -10000.0)
        mask.scatter_(1, neg.topk(k).indices, 0.0)
        mask.scatter_(1, torch.arange(B).unsqueeze(1), 0.0)
        expected = F.cross_entropy(sims - 10000.0 * mask, labels)
        torch.testing.assert_close(ranking_loss(sims, labels, hardneg_topk=k), expected)

    def test_hardneg_topk_ignores_mask_for_grad(self):
        # the mask is computed on detached sims: grad w.r.t. sims must equal
        # the plain-CE grad (masked entries get no gradient beyond the -10000
        # shift, which is a constant w.r.t. sims)
        torch.manual_seed(2)
        sims = torch.randn(3, 8, requires_grad=True)
        labels = torch.arange(3)
        ranking_loss(sims, labels, hardneg_topk=2).backward()
        sims_ref = sims.detach().clone().requires_grad_(True)
        mask = torch.ones(3, 8)
        neg = sims_ref.detach().clone()
        neg.scatter_(1, labels.unsqueeze(1), -10000.0)
        mask.scatter_(1, neg.topk(2).indices, 0.0)
        mask.scatter_(1, labels.unsqueeze(1), 0.0)
        F.cross_entropy(sims_ref - 10000.0 * mask, labels).backward()
        torch.testing.assert_close(sims.grad, sims_ref.grad)


class TestPaperLambda(unittest.TestCase):

    def test_tabulated_values(self):
        self.assertEqual(paper_lambda(48), 0.05)
        self.assertEqual(paper_lambda(32), 0.05)
        self.assertEqual(paper_lambda(24), 0.05)
        self.assertEqual(paper_lambda(16), 0.07)
        self.assertEqual(paper_lambda(12), 0.1)
        self.assertEqual(paper_lambda(8), 0.2)
        self.assertEqual(paper_lambda(4), 0.3)

    def test_extrapolation(self):
        self.assertEqual(paper_lambda(96), 0.05)  # larger M -> smallest lambda
        self.assertEqual(paper_lambda(2), 0.3)    # smaller M -> largest lambda
        self.assertEqual(paper_lambda(10), 0.1)   # nearest tabulated above


class TestGradientPolicy(unittest.TestCase):
    """Analytical check of the full RepCONC gradient policy (Eqs. 9-10)."""

    def _build(self, B=3, D=6, M=2, K=4, seed=0, temperature=2.0, lam=0.25):
        torch.manual_seed(seed)
        dsub = D // M
        pq = RepCONCPQ(torch.randn(M, K, dsub), use_constraint=False)
        codes = torch.tensor([[0, 1], [1, 2], [0, 3]])[:B]
        d = torch.randn(B, D, requires_grad=True)      # continuous doc embeds
        q = torch.randn(B, D, requires_grad=True)      # continuous query embeds
        labels = torch.arange(B)
        return pq, codes, d, q, labels, temperature, lam

    def test_grad_policy_matches_analytical(self):
        pq, codes, d, q, labels, temperature, lam = self._build()
        B, D = d.shape

        # --- forward graph (as in the trainer) ---
        d_hat = pq.decode(codes)                       # [B, D], grad w.r.t. centroids
        d_st = straight_through(d, d_hat)
        sims = q @ d_st.T / temperature
        L_r = ranking_loss(sims, labels)
        L_m = clustering_loss(d_hat, d)
        (L_r + lam * L_m).backward()

        # --- reference: dL_r/dd_hat (official 'cached_grads', detached) ---
        d_hat_ref = d_hat.detach().clone().requires_grad_(True)
        L_r_ref = ranking_loss(q.detach() @ d_hat_ref.T / temperature, labels)
        L_r_ref.backward()
        g_r = d_hat_ref.grad

        # --- reference: dL_r/dq ---
        q_ref = q.detach().clone().requires_grad_(True)
        F.cross_entropy(q_ref @ d_hat.detach().T / temperature, labels).backward()
        g_q = q_ref.grad

        # --- document encoder: Eq. 9: dL/dd = dL_r/dd_hat + lambda * dL_m/dd ---
        # NB: dL_m/dd = -2 (d_hat - d) / B  (MSE is (d_hat - d)^2)
        expected_d = g_r - lam * 2 * (d_hat - d) / B
        torch.testing.assert_close(d.grad, expected_d)

        # --- query encoder: dL_r/dq ---
        torch.testing.assert_close(q.grad, g_q)

        # --- centroids: scatter of (dL_r/dd_hat + lambda * dL_m/dd_hat) ---
        expected_c = torch.zeros_like(pq.centroids)
        g_m = 2 * (d_hat - d) / B
        for b in range(B):
            for m in range(pq.M):
                expected_c[m, codes[b, m]] += (g_r[b] + lam * g_m[b])[m * pq.dsub:(m + 1) * pq.dsub]
        torch.testing.assert_close(pq.centroids.grad, expected_c)

    def test_zero_lambda(self):
        pq, codes, d, q, labels, temperature, lam = self._build(lam=0.0)
        B, D = d.shape
        d_hat = pq.decode(codes)
        d_st = straight_through(d, d_hat)
        ranking_loss(q @ d_st.T / temperature, labels).backward()
        d_hat_ref = d_hat.detach().clone().requires_grad_(True)
        ranking_loss(q.detach() @ d_hat_ref.T / temperature, labels).backward()
        torch.testing.assert_close(d.grad, d_hat_ref.grad)
        expected_c = torch.zeros_like(pq.centroids)
        for b in range(B):
            for m in range(pq.M):
                expected_c[m, codes[b, m]] += d_hat_ref.grad[b][m * pq.dsub:(m + 1) * pq.dsub]
        torch.testing.assert_close(pq.centroids.grad, expected_c)

    def test_mse_only(self):
        # with no ranking term, only the MSE gradients remain
        torch.manual_seed(3)
        M, K, D = 2, 4, 6
        pq = RepCONCPQ(torch.randn(M, K, D // M), use_constraint=False)
        codes = torch.tensor([[0, 1], [1, 2]])
        d = torch.randn(2, D, requires_grad=True)
        d_hat = pq.decode(codes)
        lam = 0.5
        (lam * clustering_loss(d_hat, d)).backward()
        B = 2
        torch.testing.assert_close(d.grad, -lam * 2 * (d_hat - d) / B)
        expected_c = torch.zeros_like(pq.centroids)
        for b in range(B):
            for m in range(M):
                expected_c[m, codes[b, m]] += lam * 2 * (d_hat[b] - d[b])[m * 3:(m + 1) * 3] / B
        torch.testing.assert_close(pq.centroids.grad, expected_c)


if __name__ == "__main__":
    unittest.main()
