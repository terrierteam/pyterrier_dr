"""Tests for pyterrier_dr.repconc.pq.RepCONCPQ (the learnable PQ)."""
import unittest

import torch

from pyterrier_dr.repconc.pq import RepCONCPQ
from pyterrier_dr.repconc.sinkhorn import sinkhorn_assignments


def _random_centroids(M=4, K=16, dsub=3, seed=0):
    torch.manual_seed(seed)
    return torch.randn(M, K, dsub)


class TestRepCONCPQShapes(unittest.TestCase):

    def test_dimensions(self):
        pq = RepCONCPQ(_random_centroids(4, 16, 3))
        self.assertEqual(pq.M, 4)
        self.assertEqual(pq.K, 16)
        self.assertEqual(pq.dsub, 3)
        self.assertEqual(pq.d, 12)
        self.assertEqual(tuple(pq.centroids.shape), (4, 16, 3))
        self.assertEqual(tuple(pq.rotation.shape), (12, 12))
        # centroids are a (trainable) parameter; rotation is a buffer
        self.assertTrue(pq.centroids.requires_grad)
        self.assertNotIn("rotation", dict(pq.named_parameters()))
        self.assertIn("rotation", dict(pq.named_buffers()))

    def test_invalid_centroids(self):
        with self.assertRaises(ValueError):
            RepCONCPQ(torch.randn(4, 16))
        with self.assertRaises(ValueError):
            RepCONCPQ(torch.randn(4, 16, 0))

    def test_invalid_rotation(self):
        with self.assertRaises(ValueError):
            RepCONCPQ(_random_centroids(2, 4, 3), rotation=torch.eye(11))

    def test_default_rotation_is_identity(self):
        pq = RepCONCPQ(_random_centroids(2, 4, 3))
        self.assertTrue(torch.eye(6).allclose(pq.rotation))


class TestRepCONCPQDistances(unittest.TestCase):

    def test_matches_naive(self):
        torch.manual_seed(1)
        centroids = _random_centroids(3, 8, 4)
        pq = RepCONCPQ(centroids)
        x = torch.randn(17, 12)
        d = pq.squared_distances(x)
        self.assertEqual(tuple(d.shape), (17, 3, 8))
        # naive per-subspace computation (the official implementation's formula)
        x_m = x.reshape(17, 3, 4)
        for m in range(3):
            naive = ((x_m[:, m:m+1] - centroids[m][None])**2).sum(-1)
            torch.testing.assert_close(d[:, m], naive, rtol=1e-4, atol=1e-4)

    def test_quantize_unconstrained_is_argmin(self):
        torch.manual_seed(2)
        centroids = _random_centroids(3, 8, 4)
        pq = RepCONCPQ(centroids, use_constraint=False)
        x = torch.randn(33, 12)
        codes = pq.quantize(x)
        x_m = x.reshape(33, 3, 4)
        for m in range(3):
            naive = ((x_m[:, m:m+1] - centroids[m][None])**2).sum(-1).argmin(-1)
            torch.testing.assert_close(codes[:, m], naive)
        # no grad on codes
        self.assertFalse(codes.requires_grad)

    def test_quantize_constrained_matches_sinkhorn(self):
        torch.manual_seed(3)
        centroids = _random_centroids(3, 8, 4)
        pq = RepCONCPQ(centroids, use_constraint=True)
        x = torch.randn(32, 12)
        codes = pq.quantize(x)
        expected = sinkhorn_assignments(pq.squared_distances(x), epsilon=pq.epsilon, iters=pq.iters)
        torch.testing.assert_close(codes, expected)
        self.assertFalse(codes.requires_grad)

    def test_rotation(self):
        torch.manual_seed(4)
        d = 12
        R = torch.linalg.qr(torch.randn(d, d))[0]  # orthonormal
        pq = RepCONCPQ(_random_centroids(3, 8, 4), rotation=R)
        x = torch.randn(5, d)
        torch.testing.assert_close(pq.rotate(x), x @ R)


class TestRepCONCPQDecode(unittest.TestCase):

    def test_matches_naive_lookup(self):
        torch.manual_seed(5)
        centroids = _random_centroids(3, 8, 4)
        pq = RepCONCPQ(centroids)
        codes = torch.tensor([[0, 3, 7], [5, 5, 5], [2, 1, 4]])
        out = pq.decode(codes)
        self.assertEqual(tuple(out.shape), (3, 12))
        for b in range(3):
            for m in range(3):
                torch.testing.assert_close(out[b, m*4:(m+1)*4], centroids[m, codes[b, m]])

    def test_decode_gradient_is_sparse(self):
        # d/dcentroids of decode: the gradient is non-zero only at the
        # selected centroids, with value grad * (number of uses)
        torch.manual_seed(6)
        centroids = _random_centroids(3, 8, 4)
        pq = RepCONCPQ(centroids)
        codes = torch.tensor([[0, 3, 3], [3, 0, 5]])  # centroid (3,3) used twice
        out = pq.decode(codes)
        g = torch.randn_like(out)
        out.backward(g, retain_graph=False)
        grad = pq.centroids.grad
        self.assertEqual(tuple(grad.shape), (3, 8, 4))
        for m in range(3):
            for k in range(8):
                members = [b for b in range(2) if codes[b, m] == k]
                if not members:
                    self.assertTrue((grad[m, k] == 0).all(), f"non-zero grad at unused centroid ({m},{k})")
                else:
                    expected = torch.stack([g[b, m*4:(m+1)*4] for b in members]).sum(0)
                    torch.testing.assert_close(grad[m, k], expected)

    def test_mse_gradient_matches_analytical(self):
        # paper Eq. 10 / official: L_c = ((d_hat - d)^2).sum(-1).mean()
        # dL_c/dcentroids[m,k] = (2/B) * sum_{b: code[b,m]==k} (d_hat_b - d_b)
        torch.manual_seed(7)
        centroids = _random_centroids(2, 4, 3)
        pq = RepCONCPQ(centroids)
        B = 4
        codes = torch.tensor([[0, 1], [1, 0], [0, 1], [3, 2]])
        x_rot = torch.randn(B, 6)
        d_hat = pq.decode(codes)
        loss = ((d_hat - x_rot) ** 2).sum(-1).mean()
        loss.backward()
        for m in range(2):
            for k in range(4):
                members = [b for b in range(B) if codes[b, m] == k]
                if not members:
                    self.assertTrue((pq.centroids.grad[m, k] == 0).all())
                else:
                    # L_c = (1/B) sum_b ||d_hat_b - x_b||^2, so
                    # dL_c/dC[m,k] = (2/B) sum_{b in members} (d_hat_b - x_b)[m]
                    expected = (2.0 / B) * torch.stack([d_hat[b] - x_rot[b] for b in members]).sum(0)[m*3:(m+1)*3]
                    torch.testing.assert_close(pq.centroids.grad[m, k], expected)


class TestRepCONCPQForward(unittest.TestCase):

    def test_forward(self):
        torch.manual_seed(8)
        pq = RepCONCPQ(_random_centroids(2, 8, 3), use_constraint=False)
        x = torch.randn(9, 6)
        d_hat, codes = pq.forward(x)
        self.assertEqual(tuple(d_hat.shape), (9, 6))
        self.assertEqual(tuple(codes.shape), (9, 2))
        self.assertTrue(d_hat.requires_grad)  # differentiable w.r.t. centroids
        self.assertFalse(codes.requires_grad)
        # forward == rotate -> quantize -> decode
        d_hat2 = pq.decode(pq.quantize(pq.rotate(x)))
        torch.testing.assert_close(d_hat, d_hat2)

    def test_reconstruction_is_close_for_well_separated_data(self):
        # documents drawn from near the centroids should reconstruct closely
        torch.manual_seed(9)
        M, K, dsub = 2, 8, 4
        pq = RepCONCPQ(torch.randn(M, K, dsub), use_constraint=False)
        codes = torch.randint(0, K, (64, M))
        x = pq.decode(codes) + torch.randn(64, 8) * 0.01
        d_hat, codes_out = pq.forward(x)
        torch.testing.assert_close(codes_out, codes)
        err = ((d_hat - x) ** 2).sum(-1).mean().sqrt().item()
        self.assertLess(err, 0.1)

    def test_empty_batch(self):
        pq = RepCONCPQ(_random_centroids(2, 4, 3))
        d_hat, codes = pq.forward(torch.empty(0, 6))
        self.assertEqual(tuple(d_hat.shape), (0, 6))
        self.assertEqual(tuple(codes.shape), (0, 2))

    def test_rotation_preserves_inner_products(self):
        # the OPQ rotation is orthonormal, so dot-product similarity is
        # unchanged in the rotated space
        torch.manual_seed(10)
        d = 12
        R = torch.linalg.qr(torch.randn(d, d))[0]
        pq = RepCONCPQ(_random_centroids(3, 4, 4), rotation=R)
        q = torch.randn(3, d)
        docs = torch.randn(5, d)
        # both sides rotated: dot-product similarity is invariant
        torch.testing.assert_close(
            pq.rotate(q) @ pq.rotate(docs).T,
            q @ docs.T,
            rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
