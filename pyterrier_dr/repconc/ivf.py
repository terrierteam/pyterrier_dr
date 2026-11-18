"""FAISS index construction for RepCONC retrieval (ADC).

Two layouts are supported:

* **Flat PQ** (``nlist == 1``): a ``faiss.IndexPQ`` with the *learned*
  centroids injected (the same pattern the official evaluation uses).
* **IVF PQ** (``nlist > 1``): a ``faiss.IndexIVFPQ`` whose coarse (IVF)
  quantiser is trained with unsupervised k-means over the (rotated) corpus,
  as described in paper §3.6.2, while the PQ codebook is the learned
  centroids (replacing the PQ part after training, as the official
  ``replace_pq_centroids`` does).

Note: the official evaluation code uses a degenerate IVF (a single
single-entry coarse cell, i.e. effectively flat PQ); this module follows the
paper and supports a real IVF. Both use asymptotically-exact
approximate dot-product (ADC) search.
"""

import faiss
import numpy as np

__all__ = ["pack_pq_codes", "build_faiss_index", "rebuild_faiss_index"]


def pack_pq_codes(codes: np.ndarray, nbits: int) -> np.ndarray:
    """Pack ``[N, M]`` codes (values in ``[0, 2**nbits)``) into FAISS's
    bit-packed layout (a flat ``uint8`` array of length ``N * code_size``).

    FAISS stores PQ codes bit-packed when ``nbits != 8``. This is the same
    packing used by :class:`pyterrier_dr.jpq.retriever.JPQRetrieverPQ`.
    """
    assert codes.dtype in [np.uint8, np.uint16], str(codes.dtype)
    assert codes.ndim == 2, str(codes.shape)
    N, M = codes.shape
    code_size = (M * nbits + 7) // 8  # bytes per vector
    out = np.zeros((N, code_size), dtype=np.uint8)
    bit_cursor = 0
    for m in range(M):
        v = codes[:, m].astype(np.uint32)  # [N]
        for b in range(nbits):
            byte_idx = (bit_cursor + b) // 8
            bit_in_byte = (bit_cursor + b) % 8
            out[:, byte_idx] |= (((v >> b) & 1).astype(np.uint8)) << bit_in_byte
        bit_cursor += nbits
    return out.reshape(-1)


def build_faiss_index(
        d: int,
        M: int,
        nbits: int,
        centroids: np.ndarray,  # [M, Ks, dsub]
        codes: np.ndarray,      # [N, M]
        x_rot: np.ndarray | None = None,  # [N, d] rotated corpus (required for nlist > 1)
        nlist: int = 1,
        seed: int = 42,
) -> faiss.Index:
    """Build a FAISS ADC index with the provided (learned) PQ codebook.

    Args:
        d: embedding dimensionality.
        M: number of subspaces.
        nbits: bits per subspace code (codebook size is ``2**nbits``).
        centroids: learned centroids, ``[M, 2**nbits, d//M]``.
        codes: document codes, ``[N, M]`` (argmin assignments).
        x_rot: rotated corpus vectors, ``[N, d]``; required when ``nlist > 1``
            (to train the IVF k-means and populate the inverted lists).
        nlist: number of IVF clusters (``1`` = flat PQ).
        seed: k-means seed (IVF only).

    Returns:
        A trained, populated FAISS index (``IndexPQ`` or ``IndexIVFPQ``)
        suitable for ADC search with rotated queries.
    """
    from pyterrier_dr.util import ensure_faiss_omp_compat
    ensure_faiss_omp_compat()
    Ks = centroids.shape[1]
    if Ks != 2 ** nbits:
        raise ValueError(f"codebook size {Ks} != 2**{nbits}")
    if nlist < 1:
        raise ValueError(f"nlist must be >= 1, got {nlist}")

    N = codes.shape[0]
    if nlist == 1:
        index = faiss.IndexPQ(d, M, nbits, faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True
        faiss.copy_array_to_vector(np.ascontiguousarray(centroids, dtype=np.float32).reshape(-1),
                                   index.pq.centroids)
        packed = (codes.reshape(-1) if nbits == 8 else pack_pq_codes(np.ascontiguousarray(codes), nbits))
        faiss.copy_array_to_vector(packed, index.codes)
        index.ntotal = N
        return index

    # --- IVF ---
    if x_rot is None:
        raise ValueError("x_rot (rotated corpus) is required to build an IVF index")
    x_rot = np.ascontiguousarray(x_rot, dtype=np.float32)
    if x_rot.shape != (N, d):
        raise ValueError(f"x_rot must have shape ({N}, {d}), got {x_rot.shape}")
    if N < nlist:
        raise ValueError(f"cannot build an IVF with nlist={nlist} for N={N} documents")
    nlist_eff = min(nlist, N)

    # train the coarse (IVF) quantiser with unsupervised k-means (paper
    # §3.6.2); the PQ codebook stays the *learned* one. The coarse
    # quantiser is a flat inner-product index holding the k-means
    # centroids: faiss 1.13's swig bindings expose no direct `centroids`
    # setter on IndexFlat, so the centroids are added as ordinary vectors.
    km = faiss.Kmeans(d, nlist_eff, seed=seed, niter=20)
    km.train(x_rot)
    coarse = faiss.IndexFlatIP(d)
    coarse.add(km.centroids)
    coarse.is_trained = True

    ivfpq = faiss.IndexIVFPQ(coarse, d, nlist_eff, M, nbits, faiss.METRIC_INNER_PRODUCT)
    # NB: this faiss version's IndexIVFPQ defaults to by_residual=True
    # (codes are assigned to x - coarse_centroid). RepCONC's codes must be
    # the assignments of the full (rotated) vector to the learned codebook
    # (paper §3.6.2, and the official evaluation's convention), so
    # residual encoding is disabled.
    ivfpq.by_residual = False
    # the constructor creates a fresh, untrained ProductQuantizer; replace its
    # centroids with the learned ones (and mark it trained)
    faiss.copy_array_to_vector(np.ascontiguousarray(centroids, dtype=np.float32).reshape(-1),
                               ivfpq.pq.centroids)
    ivfpq.pq.is_trained = True
    ivfpq.is_trained = True
    ivfpq.add(x_rot)
    return ivfpq


def rebuild_faiss_index(index: faiss.Index, x_rot: np.ndarray) -> faiss.Index:
    """Rebuild an IVF index's inverted lists from rotated corpus vectors.

    (Used after the IVF quantiser has been re-trained elsewhere.)
    """
    from pyterrier_dr.util import ensure_faiss_omp_compat
    ensure_faiss_omp_compat()
    index.reset()
    if x_rot is not None and len(x_rot):
        index.add(np.ascontiguousarray(x_rot, dtype=np.float32))
    return index
