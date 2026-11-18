"""Losses and gradient policy for RepCONC training.

Implements Eqs. (8)-(10) of Zhan et al. (WSDM'22)::

    L  = L_r + lambda * L_m

    L_r: contrastive ranking loss over the similarity between the
          *continuous* query embedding and the *quantised* document
          embedding (cross-entropy, temperature-scaled). All other documents
          in the ranking batch (in-batch documents plus explicit/static hard
          negatives) are negatives.
    L_m: clustering (MSE) loss ``((d_hat - d)^2).sum(-1).mean()`` (summed
          over the embedding dimension, averaged over the batch, as in the
          official implementation).

Gradient policy (paper Eqs. 9-10; identical to the official GradCache-based
implementation):

* the **document encoder** receives ``dL_r/dd_hat + lambda * dL_m/dd``: the
  ranking gradient w.r.t. the quantised embedding is applied to the
  *continuous* document embedding (straight-through), and the MSE gradient
  flows through the continuous embedding as usual;
* the **centroids** receive ``dL_r/dd_hat + lambda * dL_m/dd_hat`` through
  :meth:`pyterrier_dr.repconc.pq.RepCONCPQ.decode`;
* the **query encoder** receives ``dL_r/dq`` as usual (queries are not
  quantised).

The non-differentiable assignment (Sinkhorn/argmin) is therefore isolated in
the codes, which are computed under ``torch.no_grad()``; gradients reach the
centroids only through the decoded embedding.
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["StraightThrough", "straight_through", "ranking_loss", "clustering_loss",
           "paper_lambda"]

#: lambda (Eq. 8) recommended by the paper, per number of subspaces M.
#: Paper §4.1: 0.05 for M = 48/32/24, 0.07 for M = 16, 0.1 for M = 12,
#: 0.2 for M = 8, 0.3 for M = 4.
_PAPER_LAMBDA_BY_M = {48: 0.05, 32: 0.05, 24: 0.05, 16: 0.07, 12: 0.1, 8: 0.2, 4: 0.3}


def paper_lambda(M: int) -> float:
    """The paper's recommended lambda for a given number of subspaces M.

    For M values not in the paper's table, the value for the nearest (larger)
    tabulated M is used; larger M means smaller lambda, so M > 48 maps to 0.05
    and M < 4 maps to 0.3.
    """
    if M in _PAPER_LAMBDA_BY_M:
        return _PAPER_LAMBDA_BY_M[M]
    for m in sorted(_PAPER_LAMBDA_BY_M):  # ascending: 4, 8, 12, 16, 24, 32, 48
        if M <= m:
            return _PAPER_LAMBDA_BY_M[m]
    return _PAPER_LAMBDA_BY_M[48]  # M > 48


class StraightThrough(torch.autograd.Function):
    """Straight-through estimator (paper Eqs. 9-10).

    The forward pass returns the quantised embedding; the backward pass gives
    the gradient w.r.t. the quantised embedding to *both* inputs, so that the
    ranking gradient computed on the quantised embedding is applied both to
    the continuous document embedding (and, through it, the document encoder)
    and to the decoded quantised embedding (and, through it, the centroids).
    """

    @staticmethod
    def forward(ctx, continuous: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, grad_output


def straight_through(continuous: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Apply :class:`StraightThrough` to (continuous, quantised) embeddings."""
    return StraightThrough.apply(continuous, quantized)


def ranking_loss(similarities: torch.Tensor, labels: torch.Tensor,
                 hardneg_topk: int | None = None) -> torch.Tensor:
    """Contrastive (cross-entropy) ranking loss.

    Args:
        similarities: ``[B, N]`` similarity matrix (already temperature
            scaled), row ``b`` over the ``N`` candidate documents, with the
            positive document at column ``labels[b]``.
        labels: ``[B]`` index of the positive document in each row.
        hardneg_topk: if set, only the ``hardneg_topk`` most similar
            (non-positive) documents remain active negatives for each query;
            all other candidates are masked out (the official
            ``dynamic_topk_hard_negative`` mechanism). The mask is computed
            on detached similarities, as in the official code.

    Returns:
        The cross-entropy loss.
    """
    B, N = similarities.shape
    if hardneg_topk is not None and hardneg_topk > 0:
        k = min(hardneg_topk, N - 1)
        mask = torch.ones_like(similarities)
        neg_similarities = similarities.detach().clone()
        neg_similarities.scatter_(1, labels.unsqueeze(1), -10000.0)
        mask.scatter_(1, neg_similarities.topk(k).indices, 0.0)
        # the positive is always active
        mask.scatter_(1, torch.arange(B, device=similarities.device).unsqueeze(1), 0.0)
        similarities = similarities - 10000.0 * mask
    return F.cross_entropy(similarities, labels)


def clustering_loss(d_hat: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Clustering (MSE) loss ``L_m = ((d_hat - d)^2).sum(-1).mean()``.

    Summed over the embedding dimension, averaged over the batch (the
    official implementation's convention). Differentiable w.r.t. both the
    quantised embeddings (centroids) and the continuous embeddings (encoder).
    """
    return ((d_hat - d) ** 2).sum(dim=-1).mean()
