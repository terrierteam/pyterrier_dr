"""Pseudo-relevance feedback for (J)PQ retrieval.

The transformers in this module take the result frame produced by a first-pass
JPQ retrieval pipeline, for example::

    model.query_encoder() >> jpq_index.retriever() >> jpq_index.vec_loader()

and replace the query vector with a feedback vector built from the PQ codes of
the top-ranked documents.  The expected input columns are::

    qid, query, docno, score, rank, query_vec, doc_vec, codes

Only ``query_vec``, ``score`` and ``codes`` are used by the feedback methods;
``doc_vec`` is accepted (and preserved by the pipeline) but is not required by
the PRF computation itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import torch
from torch import Tensor

if TYPE_CHECKING:
    from pyterrier_dr.jpq.index import JPQIndex


def _validate(
    query: Tensor,
    scores: Tensor,
    codes: Tensor,
    centroids: Tensor,
) -> tuple[int, int, int, int]:
    k, M = codes.shape
    M2, ks, dsub = centroids.shape

    assert M == M2
    assert scores.shape == (k,)
    assert query.numel() == M * dsub
    assert codes.dtype in (torch.int32, torch.int64)

    return k, M, ks, dsub


def _score_weights(scores: Tensor, temperature: float = 1.0) -> Tensor:
    """
    Converts retrieval scores into non-negative PRF document weights.
    Softmax is usually preferable to using raw similarity scores directly.
    """
    if not torch.is_floating_point(scores):
        scores = scores.to(torch.get_default_dtype())
    return torch.softmax(scores / temperature, dim=0)


def _weighted_code_histogram(
    scores: Tensor,
    codes: Tensor,
    ks: int,
    temperature: float = 1.0,
) -> Tensor:
    """
    Returns weighted pseudo-relevant code frequencies.

    Shape: (M, ks)
    Each row sums to 1.
    """
    k, M = codes.shape
    doc_w = _score_weights(scores, temperature)

    tf = torch.zeros(
        M, ks,
        device=doc_w.device,
        dtype=doc_w.dtype,
    )

    # scatter_add_ requires a LongTensor index.  Convert once so that callers
    # may pass either int32 or int64 codes.
    codes = codes.to(device=doc_w.device, dtype=torch.long)
    for m in range(M):
        tf[m].scatter_add_(0, codes[:, m], doc_w)

    return tf


def _weights_to_feedback_vector(
    weights: Tensor,       # (M, ks)
    centroids: Tensor,     # (M, ks, dsub)
    eps: float = 1e-12,
) -> Tensor:
    """
    Converts centroid weights into one dense feedback vector.

    We normalize independently inside each PQ split.
    """
    weights = weights.clamp_min(0)

    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    weights = weights / denom

    centroids = centroids.to(device=weights.device, dtype=weights.dtype)

    # (M, ks) x (M, ks, dsub) -> (M, dsub)
    feedback = torch.einsum("mk,mkd->md", weights, centroids)

    return feedback.flatten()


def _interpolate(
    query: Tensor,
    feedback: Tensor,
    alpha: float = 0.7,
) -> Tensor:
    """
    alpha = weight on original query.
    """
    if not torch.is_floating_point(query):
        query = query.to(torch.get_default_dtype())
    feedback = feedback.to(device=query.device, dtype=query.dtype)
    return alpha * query + (1.0 - alpha) * feedback


def _collection_code_freq(codes, ks: int) -> Tensor:
    """Counts how often each PQ centroid is used in a collection.

    Args:
        codes: PQ codes with shape ``(N, M)``.  May be a numpy array, a
            ``torch.Tensor`` or anything else that can be converted to one.
        ks: Number of centroids per split.

    Returns:
        A ``(M, ks)`` long tensor of code frequencies.
    """
    if isinstance(codes, torch.Tensor):
        codes = codes.detach().to(dtype=torch.long)
        M = codes.shape[1]
        freq = torch.empty((M, ks), dtype=torch.long, device=codes.device)
        for m in range(M):
            counts = torch.bincount(codes[:, m], minlength=ks)
            if counts.numel() != ks:
                raise ValueError(
                    f"codes contain a value outside [0, {ks}) for split {m}"
                )
            freq[m] = counts
        return freq

    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"codes must have shape (N, M), got {codes.shape}")

    M = codes.shape[1]
    freq = np.empty((M, ks), dtype=np.int64)
    for m in range(M):
        counts = np.bincount(codes[:, m].astype(np.int64), minlength=ks)
        if len(counts) != ks:
            raise ValueError(
                f"codes contain a value outside [0, {ks}) for split {m}"
            )
        freq[m] = counts
    return torch.from_numpy(freq)


def _as_tensor(value, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Converts numpy arrays / sequences to tensors without unnecessary copies."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    array = np.asarray(value)
    # Read-only arrays (e.g. memmaps) trigger a PyTorch warning and cannot be
    # written to safely, so take a writable copy in that case.
    if not array.flags.writeable:
        array = array.copy()
    tensor = torch.as_tensor(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


class PQPrf(pt.Transformer, ABC):
    """Abstract base class for PQ pseudo-relevance feedback.

    A concrete subclass implements :meth:`pq_prf`, which maps a query vector
    and the PQ codes of its top-ranked documents to a new query vector.

    The supplied query vector and the PQ centroids must be in the same vector
    space.  In particular, for an OPQ index both must be in the rotated space.

    The collection-level statistics needed by IDF-style feedback methods can be
    supplied either through a :class:`~pyterrier_dr.jpq.JPQIndex` or directly:

    >>> prf = PQPrfTF(jpq_index, alpha=0.7, k=10)
    >>> prf = PQPrfTF(
    ...     centroids=centroids,
    ...     collection_code_freq=freq,
    ...     n_docs=len(codes),
    ...     alpha=0.7,
    ...     k=10,
    ... )

    Args:
        index: An optional :class:`~pyterrier_dr.jpq.JPQIndex`.  When given,
            ``centroids``, ``collection_code_freq`` and ``n_docs`` are read
            from it unless explicitly provided.
        centroids: PQ centroids with shape ``(M, ks, dsub)``.
        collection_code_freq: Collection code frequencies with shape
            ``(M, ks)``.  Required by the IDF-style feedback methods.
        n_docs: Number of documents in the collection.  Required by the
            IDF-style feedback methods.
        alpha: Weight of the original query in the interpolated query.
        temperature: Softmax temperature used to weight feedback documents.
        k: Optional number of top-ranked documents to use.  When set,
            :meth:`compile` adds a :class:`pyterrier.RankCutoff` and
            :meth:`transform` also truncates the input.
        jpq_index: Alias for ``index``.
    """

    #: Whether this feedback method needs collection-level code statistics.
    requires_collection_stats = True

    def __init__(
        self,
        index: Optional["JPQIndex"] = None,
        *,
        centroids: Optional[Tensor] = None,
        collection_code_freq: Optional[Tensor] = None,
        n_docs: Optional[int] = None,
        alpha: float = 0.7,
        temperature: float = 1.0,
        k: Optional[int] = None,
        jpq_index: Optional["JPQIndex"] = None,
    ):
        super().__init__()

        if index is not None and jpq_index is not None:
            raise ValueError("Pass either index or jpq_index, not both")
        if index is None:
            index = jpq_index

        if centroids is None:
            if index is None:
                raise ValueError(
                    "centroids must be provided when no JPQIndex is given"
                )
            centroids = index.dvecs

        self.centroids = _as_tensor(centroids)

        if collection_code_freq is None and index is not None:
            collection_code_freq = _collection_code_freq(
                index.codes, self.centroids.shape[1]
            )
        self.collection_code_freq = (
            None
            if collection_code_freq is None
            else _as_tensor(collection_code_freq)
        )

        if n_docs is None and index is not None:
            n_docs = len(index)
        self.n_docs = n_docs

        if self.collection_code_freq is not None:
            expected = tuple(self.centroids.shape[:2])
            actual = tuple(self.collection_code_freq.shape)
            if actual != expected:
                raise ValueError(
                    "collection_code_freq must have shape "
                    f"{expected}, got {actual}"
                )

        if self.requires_collection_stats:
            if self.collection_code_freq is None:
                raise ValueError(
                    f"{type(self).__name__} requires collection_code_freq"
                )
            if self.n_docs is None:
                raise ValueError(f"{type(self).__name__} requires n_docs")

        self.alpha = alpha
        self.temperature = temperature
        self.k = k

    @abstractmethod
    def pq_prf(
        self,
        query: Tensor,                    # float, (M*dsub,)
        scores: Tensor,                   # float, (k,)
        codes: Tensor,                    # long,  (k, M)
        centroids: Tensor,                # float, (M, ks, dsub)
        collection_code_freq: Tensor,     # int/float, (M, ks)
        n_docs: int,
        *,
        alpha: float = 0.7,
        temperature: float = 1.0,
    ) -> Tensor:                          # float, (M*dsub,)
        """Returns a new query vector for one query."""
        ...

    def compile(self) -> pt.Transformer:
        """Adds a rank cutoff before this transformer when ``k`` is set."""
        if self.k is None:
            return self
        return pt.RankCutoff(self.k) >> self

    @pta.transform.by_query(add_ranks=False)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Applies PQ PRF to each query in the input result frame."""
        pta.validate.result_frame(
            inp, extra_columns=['query_vec', 'codes']
        )

        query_cols = [
            col for col in inp.columns
            if col.startswith('q') and col != 'query_vec'
        ]

        if len(inp) == 0:
            return pd.DataFrame([], columns=query_cols + ['query_vec'])

        if 'rank' in inp.columns:
            inp = inp.sort_values('rank')
        if self.k is not None:
            inp = inp.head(self.k)

        query = _as_tensor(inp['query_vec'].iloc[0])
        scores = torch.as_tensor(inp['score'].to_numpy())
        codes = torch.as_tensor(
            np.stack(inp['codes'].to_list()), dtype=torch.long
        )

        # Prefer a non-CPU device if the centroids have been placed there.
        device = query.device
        if self.centroids.device.type != 'cpu':
            device = self.centroids.device

        query = query.to(device)
        scores = scores.to(device)
        codes = codes.to(device)
        centroids = self.centroids.to(device)
        collection_code_freq = (
            None
            if self.collection_code_freq is None
            else self.collection_code_freq.to(device)
        )

        new_query = self.pq_prf(
            query,
            scores,
            codes,
            centroids,
            collection_code_freq,
            self.n_docs,
            alpha=self.alpha,
            temperature=self.temperature,
        )

        new_query = new_query.detach().cpu().numpy()
        row = [inp[col].iloc[0] for col in query_cols] + [new_query]
        return pd.DataFrame([row], columns=query_cols + ['query_vec'])

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(alpha={self.alpha}, "
            f"temperature={self.temperature}, k={self.k})"
        )


def pq_prf_tf(
    query: Tensor,
    scores: Tensor,
    codes: Tensor,
    centroids: Tensor,
    *,
    alpha: float = 0.7,
    temperature: float = 1.0,
) -> Tensor:
    """
    PRF weight = score-weighted pseudo-relevant code frequency.
    """
    _, _, ks, _ = _validate(query, scores, codes, centroids)

    tf = _weighted_code_histogram(
        scores, codes, ks, temperature
    )

    feedback = _weights_to_feedback_vector(tf, centroids)

    return _interpolate(query, feedback, alpha)


def pq_prf_tfidf(
    query: Tensor,
    scores: Tensor,
    codes: Tensor,
    centroids: Tensor,
    *,
    collection_code_freq: Tensor,
    n_docs: int,
    alpha: float = 0.7,
    temperature: float = 1.0,
) -> Tensor:
    """
    Classical TF-IDF over PQ centroid IDs.
    """
    _, M, ks, _ = _validate(query, scores, codes, centroids)

    assert collection_code_freq.shape == (M, ks)

    tf = _weighted_code_histogram(
        scores, codes, ks, temperature
    )

    df = collection_code_freq.to(device=tf.device, dtype=tf.dtype)

    idf = torch.log(
        (float(n_docs) + 1.0) / (df + 1.0)
    )

    weights = tf * idf

    feedback = _weights_to_feedback_vector(weights, centroids)

    return _interpolate(query, feedback, alpha)


def pq_prf_probabilistic_tfidf(
    query: Tensor,
    scores: Tensor,
    codes: Tensor,
    centroids: Tensor,
    *,
    collection_code_freq: Tensor,
    n_docs: int,
    alpha: float = 0.7,
    temperature: float = 1.0,
) -> Tensor:
    """
    TF x Robertson-style probabilistic IDF.

    Negative IDFs are clipped because the result is being interpreted as
    mixture weights over centroids.
    """
    _, M, ks, _ = _validate(query, scores, codes, centroids)

    assert collection_code_freq.shape == (M, ks)

    tf = _weighted_code_histogram(
        scores, codes, ks, temperature
    )

    df = collection_code_freq.to(device=tf.device, dtype=tf.dtype)

    idf = torch.log(
        (float(n_docs) - df + 0.5)
        / (df + 0.5)
    )

    idf = idf.clamp_min(0)

    weights = tf * idf

    feedback = _weights_to_feedback_vector(weights, centroids)

    return _interpolate(query, feedback, alpha)


def pq_prf_kl_tfidf(
    query: Tensor,
    scores: Tensor,
    codes: Tensor,
    centroids: Tensor,
    *,
    collection_code_freq: Tensor,
    n_docs: int,
    alpha: float = 0.7,
    temperature: float = 1.0,
    smoothing: float = 1e-6,
) -> Tensor:
    """
    Codeword weight:

        p_R(code) * log[p_R(code) / p_C(code)]

    Negative contributions are dropped for expansion: we only move toward
    codes enriched in the pseudo-relevant set.
    """
    _, M, ks, _ = _validate(query, scores, codes, centroids)

    assert collection_code_freq.shape == (M, ks)

    p_r = _weighted_code_histogram(
        scores, codes, ks, temperature
    )

    counts = collection_code_freq.to(device=p_r.device, dtype=p_r.dtype)

    # In ordinary PQ, every document contributes exactly one assignment
    # to each split, so dividing by N produces p_C(code | split).
    p_c = (counts + smoothing) / (
        float(n_docs) + smoothing * ks
    )

    p_r_smooth = p_r + smoothing
    p_r_smooth = p_r_smooth / p_r_smooth.sum(
        dim=-1, keepdim=True
    )

    log_ratio = torch.log(p_r_smooth / p_c)

    weights = p_r * log_ratio

    # Only expansion toward positively enriched codes.
    weights = weights.clamp_min(0)

    feedback = _weights_to_feedback_vector(weights, centroids)

    return _interpolate(query, feedback, alpha)


class PQPrfTF(PQPrf):
    """TF-only PQ pseudo-relevance feedback."""

    requires_collection_stats = False

    def pq_prf(
        self,
        query: Tensor,
        scores: Tensor,
        codes: Tensor,
        centroids: Tensor,
        collection_code_freq: Optional[Tensor] = None,
        n_docs: Optional[int] = None,
        *,
        alpha: float = 0.7,
        temperature: float = 1.0,
    ) -> Tensor:
        return pq_prf_tf(
            query,
            scores,
            codes,
            centroids,
            alpha=alpha,
            temperature=temperature,
        )


class PQPrfTFIDF(PQPrf):
    """Classical TF-IDF PQ pseudo-relevance feedback."""

    def pq_prf(
        self,
        query: Tensor,
        scores: Tensor,
        codes: Tensor,
        centroids: Tensor,
        collection_code_freq: Tensor,
        n_docs: int,
        *,
        alpha: float = 0.7,
        temperature: float = 1.0,
    ) -> Tensor:
        return pq_prf_tfidf(
            query,
            scores,
            codes,
            centroids,
            collection_code_freq=collection_code_freq,
            n_docs=n_docs,
            alpha=alpha,
            temperature=temperature,
        )


class PQPrfProbabilisticTFIDF(PQPrf):
    """Robertson/Sparck-Jones probabilistic TF-IDF PQ PRF."""

    def pq_prf(
        self,
        query: Tensor,
        scores: Tensor,
        codes: Tensor,
        centroids: Tensor,
        collection_code_freq: Tensor,
        n_docs: int,
        *,
        alpha: float = 0.7,
        temperature: float = 1.0,
    ) -> Tensor:
        return pq_prf_probabilistic_tfidf(
            query,
            scores,
            codes,
            centroids,
            collection_code_freq=collection_code_freq,
            n_docs=n_docs,
            alpha=alpha,
            temperature=temperature,
        )


class PQPrfKLTFIDF(PQPrf):
    """Relevance-model / KL-divergence PQ pseudo-relevance feedback."""

    def __init__(self, *args, smoothing: float = 1e-6, **kwargs):
        super().__init__(*args, **kwargs)
        self.smoothing = smoothing

    def pq_prf(
        self,
        query: Tensor,
        scores: Tensor,
        codes: Tensor,
        centroids: Tensor,
        collection_code_freq: Tensor,
        n_docs: int,
        *,
        alpha: float = 0.7,
        temperature: float = 1.0,
        smoothing: Optional[float] = None,
    ) -> Tensor:
        return pq_prf_kl_tfidf(
            query,
            scores,
            codes,
            centroids,
            collection_code_freq=collection_code_freq,
            n_docs=n_docs,
            alpha=alpha,
            temperature=temperature,
            smoothing=self.smoothing if smoothing is None else smoothing,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(alpha={self.alpha}, "
            f"temperature={self.temperature}, k={self.k}, "
            f"smoothing={self.smoothing})"
        )


__all__ = [
    "PQPrf",
    "PQPrfTF",
    "PQPrfTFIDF",
    "PQPrfProbabilisticTFIDF",
    "PQPrfKLTFIDF",
    "pq_prf_tf",
    "pq_prf_tfidf",
    "pq_prf_probabilistic_tfidf",
    "pq_prf_kl_tfidf",
]
