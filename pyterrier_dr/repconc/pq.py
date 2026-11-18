"""Learnable product quantizer for RepCONC.

Implements the PQ core of Zhan et al. (WSDM'22): the sub-vector centroids are
``nn.Parameter`` (learnable, unlike ordinary PQ), document assignments are the
constrained-clustering (Sinkhorn) hard codes of the paper, and decoded
embeddings are differentiable with respect to the centroids so that the
reconstruction (MSE) term can train them.

Conventions (matching the official implementation):

* Centroids are stored in the official layout ``[M, K, dsub]`` where
  ``d = M * dsub``.
* The rotation is the *fixed* OPQ rotation from the warm-up (paper §3.6.1).
  It is a ``register_buffer`` (not a learnable parameter), unlike JPQ where
  the rotation is trained. It is stored in the same convention as
  :class:`pyterrier_dr.jpq.pq.ProductQuantizerFAISSIndexPQOPQ` (``opq = A.T``),
  so a warmed-up JPQ PQ object can be used to initialise this module; the
  rotated space is ``x @ rotation`` (equivalently ``x @ A.T`` in the official
  code, which stores ``A`` and computes ``x @ A.T``).
* Assignments (``quantize``) are computed under ``torch.no_grad()``: the codes
  are discrete, so gradients reach the centroids only through ``decode``.
* Similarities are computed in the rotated space; because the OPQ rotation is
  orthonormal this is equivalent to dot-product similarity in the original
  space.
"""

import torch
from torch import nn

from .sinkhorn import sinkhorn_assignments

__all__ = ["RepCONCPQ"]


class RepCONCPQ(nn.Module):
    """PQ with learnable centroids and Sinkhorn-constrained assignments.

    Args:
        centroids: initial sub-vector centroids, shape ``[M, K, dsub]``
            (e.g. from the OPQ+PQ warm-up).
        rotation: fixed ``[d, d]`` rotation applied to queries and documents
            before quantisation (``A.T`` of the OPQ warm-up matrix). If
            ``None``, the identity is used.
        epsilon: entropic regularisation for the Sinkhorn constraint.
        iters: number of Sinkhorn-Knopp iterations.
        use_constraint: if False, use plain nearest-centroid (argmin)
            assignment instead of the constrained clustering (this is the
            paper's ablation setting and the mode used to compute the final
            index codes once training has finished).
    """

    def __init__(self, centroids, rotation=None, *, epsilon: float = 0.003,
                 iters: int = 100, use_constraint: bool = True):
        super().__init__()
        centroids = torch.as_tensor(centroids, dtype=torch.float32)
        if centroids.dim() != 3:
            raise ValueError(f"centroids must have shape [M, K, dsub], got {tuple(centroids.shape)}")
        M, K, dsub = centroids.shape
        if dsub <= 0:
            raise ValueError(f"invalid centroids shape {tuple(centroids.shape)}")
        self.register_parameter("centroids", nn.Parameter(centroids.contiguous()))
        d = M * dsub
        if rotation is None:
            rotation = torch.eye(d, dtype=torch.float32)
        else:
            rotation = torch.as_tensor(rotation, dtype=torch.float32)
            if rotation.shape != (d, d):
                raise ValueError(f"rotation must have shape ({d}, {d}), got {tuple(rotation.shape)}")
        # fixed at init: the rotation is NOT learned by RepCONC
        self.register_buffer("rotation", rotation.contiguous())
        self.epsilon = epsilon
        self.iters = iters
        self.use_constraint = use_constraint

    # -- dimensions (derived from the centroids parameter) -------------------

    @property
    def M(self) -> int:
        """Number of subspaces."""
        return self.centroids.shape[0]

    @property
    def K(self) -> int:
        """Number of centroids per subspace."""
        return self.centroids.shape[1]

    @property
    def dsub(self) -> int:
        """Sub-vector dimensionality."""
        return self.centroids.shape[2]

    @property
    def d(self) -> int:
        """Full embedding dimensionality."""
        return self.M * self.dsub

    # -- forward components ---------------------------------------------------

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fixed rotation: ``[B, d] -> [B, d]``."""
        return x @ self.rotation

    def squared_distances(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Squared distances from rotated embeddings to all centroids.

        Args:
            x_rot: rotated embeddings ``[B, d]`` (i.e. already ``x @ rotation``).

        Returns:
            ``[B, M, K]`` tensor of squared distances: for each document,
            subspace and centroid, ``||x_m - c_{m,k}||^2``.
        """
        B = x_rot.shape[0]
        x_m = x_rot.reshape(B, self.M, self.dsub)
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c  (avoids the [B, M, K, dsub]
        # intermediate of the official implementation)
        xx = (x_m * x_m).sum(dim=-1)                      # [B, M]
        cc = (self.centroids * self.centroids).sum(dim=-1)  # [M, K]
        xc = torch.einsum("bmd,mkd->bmk", x_m, self.centroids)
        return xx.unsqueeze(-1) + cc.unsqueeze(0) - 2 * xc

    @torch.no_grad()
    def quantize(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Assign hard centroid codes to rotated embeddings.

        Args:
            x_rot: rotated embeddings ``[B, d]``.

        Returns:
            ``[B, M]`` long tensor of centroid identifiers (paper Eq. 17:
            ``argmax`` of the constrained-assignment solution).
        """
        distances = self.squared_distances(x_rot)
        if self.use_constraint:
            return sinkhorn_assignments(distances, epsilon=self.epsilon, iters=self.iters)
        return distances.argmin(dim=-1)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct rotated embeddings from codes (differentiable).

        Args:
            codes: ``[B, M]`` long tensor of centroid identifiers.

        Returns:
            ``[B, d]`` reconstructed (rotated-space) embeddings. Gradients
            flow to :attr:`centroids`.
        """
        if codes.shape[1] != self.M:
            raise ValueError(f"codes must have shape [B, {self.M}], got {tuple(codes.shape)}")
        first = torch.arange(self.M, device=codes.device).unsqueeze(0).expand(codes.shape[0], self.M)
        gathered = self.centroids[first, codes]  # [B, M, dsub]
        return gathered.reshape(codes.shape[0], self.d)

    def forward(self, x: torch.Tensor):
        """Rotate, quantise (no grad) and decode.

        Args:
            x: unrotated embeddings ``[B, d]``.

        Returns:
            Tuple ``(quantized, codes)``: ``quantized`` is ``[B, d]`` in the
            rotated space (differentiable w.r.t. the centroids) and ``codes``
            is ``[B, M]``.
        """
        x_rot = self.rotate(x)
        codes = self.quantize(x_rot)
        return self.decode(codes), codes
