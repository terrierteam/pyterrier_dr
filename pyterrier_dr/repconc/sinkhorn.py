"""
Sinkhorn-Knopp optimal transport for RepCONC's constrained clustering.

Implements Eq. 19 of Zhan et al. (2022), "Learning Discrete Representations via
Constrained Clustering for Effective and Efficient Dense Retrieval" (WSDM'22).

For each PQ subspace i, we seek an assignment matrix ``q`` over a mini-batch
``B`` of documents::

    min_q  sum_{d in B} sum_j q(j | d) * ||c_{i,j} - d_i||^2
    s.t.   sum_j q(j | d) = 1                 for all d      (each doc fully assigned)
           sum_{d in B} q(j | d) = |B| / K     for all j      (uniform centroid utilisation)

Relaxing ``q`` to be continuous turns this into an entropically-regularised
optimal transport problem, solved with Sinkhorn-Knopp iterations (Cuturi, 2013).

After the final rescaling the soft assignment ``Q`` satisfies:

* ``Q >= 0``
* ``sum_k Q[b, k] == 1`` exactly (per document),
* ``sum_b Q[b, k] ~= |B| / K`` (per centroid, within the iteration tolerance).

Hard centroid identifiers are obtained with ``argmax`` (paper Eq. 17).

The batch size does not need to be divisible by ``K``: the per-centroid
marginal ``|B| / K`` is only required to hold in expectation, and Sinkhorn
converges to the target marginals for any positive target (see
https://en.wikipedia.org/wiki/Sinkhorn%E2%80%93Knopp_algorithm).
"""

import torch

__all__ = ["center_distance", "sinkhorn", "sinkhorn_assignments"]


def center_distance(distances: torch.Tensor) -> torch.Tensor:
    """Numerical stabilisation for :func:`sinkhorn`.

    Center and scale the (non-negative) distances of each subspace to
    (approximately) the range ``[-1, 1]``, so that ``exp(-cost / epsilon)``
    remains within a safe range (in float64) even for small ``epsilon``.
    This is the same stabilisation used by the official RepCONC
    implementation (``center_distance_for_constraint``).

    Args:
        distances: non-negative cost tensor of shape ``[..., B, K]``
            (e.g. ``[B, K]`` for one subspace or ``[B, M, K]`` for all subspaces).

    Returns:
        Centered/scaled tensor of the same shape.
    """
    # NB: the normalisation is per subspace (the last-but-one axis, M for 3D
    # input [B, M, K], i.e. dim 0 for the 2D input [B, K]). This matches the
    # official RepCONC implementation, which reduces over documents after
    # reducing over centroids.
    max_distance = distances.amax(dim=-1, keepdim=True).amax(dim=0, keepdim=True)
    min_distance = distances.amin(dim=-1, keepdim=True).amin(dim=0, keepdim=True)
    middle = (max_distance + min_distance) / 2
    amplitude = max_distance - middle + 1e-5
    return (distances - middle) / amplitude


def sinkhorn(cost: torch.Tensor, epsilon: float = 0.003, iters: int = 100) -> torch.Tensor:
    """Sinkhorn-Knopp iterations on a cost matrix.

    Args:
        cost: non-negative cost of shape ``[B, K]`` or ``[B, M, K]``.
            For 3D input, each of the ``M`` subspaces is solved independently
            (the constraint is per-subspace in RepCONC). The cost is assumed
            to be stabilised (see :func:`center_distance`) if ``epsilon`` is
            small; otherwise costs up to ~700 * epsilon are representable.
        epsilon: entropic regularisation parameter. Smaller values give a
            sharper (closer to hard) solution.
        iters: number of alternating normalisation iterations.

    Returns:
        Soft assignment ``Q`` of shape ``[B, K]`` or ``[B, M, K]`` with
        ``Q[b, :].sum() == 1`` (exact) and ``Q[:, m, k].sum() ~= B / K``.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if iters < 0:
        raise ValueError(f"iters must be non-negative, got {iters}")

    single = cost.dim() == 2
    if single:
        cost = cost.unsqueeze(1)  # [B, 1, K]
    elif cost.dim() != 3:
        raise ValueError(f"cost must have shape [B, K] or [B, M, K], got {tuple(cost.shape)}")

    B, M, K = cost.shape
    if B == 0:
        out = cost.new_zeros(B, M, K)
        return out.squeeze(1) if single else out

    # float64 for numerical stability: with the default epsilon, the exponent
    # range of exp(-cost / epsilon) can approach +-700 after center_distance
    Q = torch.exp(-cost.double() / epsilon)
    # normalise the total mass of each subspace to 1
    Q = Q / Q.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)

    for _ in range(iters):
        # normalise each centroid (over documents) to 1 / K
        Q = Q / Q.sum(dim=0, keepdim=True)
        Q = Q / K
        # normalise each document (over centroids) to 1 / B
        Q = Q / Q.sum(dim=2, keepdim=True)
        Q = Q / B

    # final rescaling: each document's mass must sum to exactly 1, so that
    # Q can be read as an assignment (the centroid marginals are then
    # ~= B / K, within the iteration tolerance)
    Q = Q * B
    if torch.isnan(Q).any() or torch.isinf(Q).any():
        raise FloatingPointError("Sinkhorn iteration produced NaN/Inf values")
    return Q.squeeze(1) if single else Q


def sinkhorn_assignments(distances: torch.Tensor, epsilon: float = 0.003, iters: int = 100) -> torch.Tensor:
    """Constrained-clustering hard assignments for a batch of document subvectors.

    Args:
        distances: squared quantisation distances of shape ``[B, K]`` or
            ``[B, M, K]`` (document subvectors vs. centroids per subspace).
        epsilon: entropic regularisation parameter.
        iters: number of Sinkhorn iterations.

    Returns:
        Hard centroid identifiers: ``[B]`` for 2D input or ``[B, M]`` for 3D
        input: for each document (and, in the 3D case, each subspace) the
        centroid with the largest constrained assignment probability
        (paper Eq. 17).
    """
    single = distances.dim() == 2
    if single:
        distances = distances.unsqueeze(1)
    elif distances.dim() != 3:
        raise ValueError(f"distances must have shape [B, K] or [B, M, K], got {tuple(distances.shape)}")
    if distances.shape[0] == 0:
        codes = torch.empty(distances.shape[:2], dtype=torch.long, device=distances.device)
        return codes.squeeze(1) if single else codes
    stabilized = center_distance(distances)
    Q = sinkhorn(stabilized, epsilon=epsilon, iters=iters)  # [B, M, K]
    codes = Q.argmax(dim=-1)  # [B, M]
    return codes.squeeze(1) if single else codes
