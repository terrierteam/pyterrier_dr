"""Retrievers for RepCONC indexes.

* :class:`RepCONCRetrieverPQ` -- ADC search. Either an in-memory
  ``faiss.IndexPQ`` built from the stored codes and (learned) centroids
  (flat, ``nlist == 1``; the same pattern :class:`pyterrier_dr.jpq.retriever.JPQRetrieverPQ`
  uses), or a loaded ``faiss.IndexIVFPQ`` (``nlist > 1``).
* :class:`RepCONCRetrieverFlat` -- exact inner-product search over the
  PQ-reconstructed vectors (useful as a sanity check against the ADC).
* :class:`RepCONCRetriever` -- text-based convenience wrapper: encodes the
  query texts with a :class:`~pyterrier_dr.biencoder.BiEncoder` and runs the
  index's ADC retriever.

All retrievers expect ``query_vec`` in the *unrotated* space; the fixed
rotation is applied internally before search (JPQ convention).
"""

from abc import abstractmethod
import math
import pandas as pd
import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier import tqdm

import faiss


def _pack_pq_codes(codes: np.ndarray, nbits: int) -> np.ndarray:
    """
    Pack [N, M] uint8 codes (values in [0, 2^nbits)) into FAISS bit-packed layout.
    Returns a flat uint8 array of length N * code_size, where code_size = ceil(M*nbits/8).
    Packing is LSB-first per code value, contiguous over sub-quantizers.
    """
    assert codes.dtype in [np.uint8, np.uint16], str(codes.dtype)
    assert codes.ndim == 2, str(codes.ndim)
    N, M = codes.shape
    code_size = (M * nbits + 7) // 8  # bytes per vector
    out = np.zeros((N, code_size), dtype=np.uint8)

    bit_cursor = 0
    for m in range(M):
        v = codes[:, m].astype(np.uint32)  # [N]
        # write nbits LSB-first into the bitstream
        for b in range(nbits):
            byte_idx = (bit_cursor + b) // 8
            bit_in_byte = (bit_cursor + b) % 8
            out[:, byte_idx] |= (((v >> b) & 1).astype(np.uint8)) << bit_in_byte
        bit_cursor += nbits

    return out.reshape(-1)  # flat vector as FAISS expects


class RepCONCRetrieverBase(pt.Transformer):
    """Base class holding the index data (docnos, codes, centroids, rotation)."""

    def __init__(
        self,
        docnos,                          # [N] docids
        codes: np.ndarray,               # PQ codes of shape [N, M] (uint8 or uint16)
        sub_embeddings: np.ndarray,      # PQ centroids of shape [M, Ks, dsub], float32
        topk: int = 1000,                # number of results to retrieve
        rotation: np.ndarray | None = None,  # [d, d], applied to queries before search
    ):
        super().__init__()

        self._index = None
        self.docnos = list(map(str, docnos))
        self.codes = np.ascontiguousarray(codes)
        self.sub_embeddings = np.ascontiguousarray(sub_embeddings)
        self.topk = int(topk)
        self.rotation = None if rotation is None else np.ascontiguousarray(rotation, dtype=np.float32)

        self._validate_data()

        self._name = self.__class__.__name__

    def __str__(self) -> str:
        return self._name

    def _validate_data(self):
        if self.codes.ndim != 2:
            raise ValueError(f"codes must be [N, M], got {self.codes.shape}")
        if self.sub_embeddings.ndim != 3:
            raise ValueError(f"sub_embeddings must be [M, Ks, dsub], got {self.sub_embeddings.shape}")

        N, M = self.codes.shape
        M2, Ks, dsub = self.sub_embeddings.shape
        if len(self.docnos) != N:
            raise ValueError(f"len(docnos)={len(self.docnos)} != N={N}")
        if M != M2:
            raise ValueError(f"M mismatch: codes has {M}, sub_embeddings has {M2}")

        nbits_f = math.log2(Ks)
        if abs(nbits_f - round(nbits_f)) > 1e-9:
            raise ValueError(f"Ks must be a power of 2, got {Ks}")

        self.N = N
        self.M = M
        self.Ks = Ks
        self.dsub = dsub
        self.d = M * dsub
        self.nbits = int(round(nbits_f))

    def _validate_queries(self, topics: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        pta.validate.query_frame(topics, extra_columns=["query_vec"])
        qvecs = topics["query_vec"].to_list()
        if not len(topics):
            return np.empty((0, self.d), dtype=np.float32), []
        Q = np.stack(qvecs).astype(np.float32, copy=False)
        Q = np.ascontiguousarray(Q)
        if self.rotation is not None:
            Q = Q @ self.rotation
        qids = topics["qid"].astype(str).tolist()
        return Q, qids


class RepCONCRetrieverFaissBase(RepCONCRetrieverBase):

    def __init__(self, *args, name: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = None
        self._name = name or self.__class__.__name__

    def __del__(self):
        if self._index is not None:
            del self._index
            self._index = None

    @abstractmethod
    def _ensure(self, bs: int = 20000) -> None:
        ...

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        from pyterrier_dr.util import ensure_faiss_omp_compat
        ensure_faiss_omp_compat()
        Q, qids = self._validate_queries(topics)
        self._ensure()
        k = min(self.topk, len(self.docnos))
        D, doc_ids = self._index.search(Q, k)  # type: ignore

        rows = []
        for i, qid in enumerate(qids):
            dids = doc_ids[i].astype(int)
            scores = D[i].astype(float)

            # stable, deterministic ordering: by score desc, then by doc index asc
            order = np.lexsort((dids, -scores))  # primary: -scores, secondary: dids
            dids = dids[order]
            scores = scores[order]

            for rank, (did, score) in enumerate(zip(dids, scores), start=1):
                row = [qid, did, self.docnos[did], score, rank]
                if "query" in topics:
                    row.append(topics.iloc[i]["query"])

                rows.append(row)

        cols = ['qid', 'docid', 'docno', 'score', 'rank']
        if "query" in topics:
            cols.append("query")

        return pd.DataFrame(rows, columns=cols)


class RepCONCRetrieverFlat(RepCONCRetrieverFaissBase):
    """Flat inner-product retriever over the PQ-reconstructed vectors
    (exact w.r.t. the stored codes)."""

    def _ensure(self, bs: int = 20_000):
        if self._index is not None:
            return

        index = faiss.IndexFlatIP(self.d)

        for start in tqdm(range(0, self.N, bs), desc=f"{self._name} / build flat", leave=False):
            stop = min(start + bs, self.N)
            chunk_codes = self.codes[start:stop, :]           # [b, M] uint8 or uint16
            parts = [self.sub_embeddings[m, chunk_codes[:, m], :] for m in range(self.M)]
            embs = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)  # [b, M*dsub]
            embs = np.ascontiguousarray(embs)
            index.add(embs)  # type: ignore

        self._index = index
        self._name = "RepCONCRetrieverFlat"


class RepCONCRetrieverPQ(RepCONCRetrieverFaissBase):
    """ADC retriever.

    * flat (``faiss_index is None``): an in-memory ``faiss.IndexPQ`` is
      constructed directly from the stored codes and centroids;
    * IVF: ``faiss_index`` is a loaded ``faiss.IndexIVFPQ`` (codes, codebook
      and inverted lists already populated).
    """

    def __init__(self, *args, faiss_index: faiss.Index | None = None, nprobe: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._faiss_index = faiss_index
        self._nprobe = nprobe

    def _ensure(self, bs: int = 20_000) -> None:
        if self._index is not None:
            return

        if self._faiss_index is not None:
            index = self._faiss_index
            if self._nprobe is not None:
                index.nprobe = min(self._nprobe, index.nlist)
            self._index = index
            self._name = "RepCONCRetrieverPQ(IVF)"
            return

        index = faiss.IndexPQ(self.d, self.M, self.nbits, faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True

        # Copy centroids: flat float array length M*Ks*dsub
        faiss.copy_array_to_vector(self.sub_embeddings.reshape(-1), index.pq.centroids)

        # Copy codes: flat uint8 array length N*M (bit-packed when nbits != 8)
        if self.nbits == 8:
            packed = self.codes.reshape(-1)
        else:
            packed = _pack_pq_codes(self.codes, self.nbits)

        faiss.copy_array_to_vector(packed, index.codes)
        index.ntotal = len(self.docnos)

        self._index = index
        self._name = "RepCONCRetrieverPQ"


class RepCONCRetriever(pt.Transformer):
    """Text-based RepCONC retriever: encodes query texts with a
    :class:`~pyterrier_dr.biencoder.BiEncoder` and runs the index's ADC
    retriever.

    Args:
        biencoder: the (trained) query/document encoder.
        index: a :class:`~pyterrier_dr.repconc.index.RepCONCIndex`.
        topk: number of documents to retrieve per query.
        verbose: whether to show progress bars during query encoding.
        batch_size: batch size for query encoding.
    """

    def __init__(self, biencoder, index, topk: int = 1000, *, verbose: bool = False,
                 batch_size: int | None = None):
        super().__init__()
        self.biencoder = biencoder
        self.index = index
        self.topk = int(topk)
        self.query_encoder = biencoder.query_encoder(verbose=verbose, batch_size=batch_size)
        self.retriever = index.retriever_pq(topk=self.topk)

    def __str__(self) -> str:
        return f"RepCONCRetriever({str(self.biencoder) if not isinstance(self.biencoder, str) else self.biencoder}, {self.index.index_path})"

    def transform(self, topics: pd.DataFrame) -> pd.DataFrame:
        pta.validate.query_frame(topics)
        topics = self.query_encoder(topics)
        return self.retriever(topics)
