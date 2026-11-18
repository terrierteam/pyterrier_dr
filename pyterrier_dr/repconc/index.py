"""RepCONC index: learned PQ codebook + document codes (+ optional IVF).

File layout (mirroring :class:`pyterrier_dr.jpq.index.JPQIndex`):

* ``pt_meta.json``  -- serialisable metadata
* ``docnos.npids``  -- docno lookup
* ``rotation.f4``   -- fixed (OPQ) rotation, float32 ``[d, d]`` (optional)
* ``subvecs.f4``    -- learned centroids, float32 ``[M, Ks, dsub]``
* ``codes.f4``      -- document codes (argmin assignments), ``[N, M]``
* ``faiss.index``   -- full IVF index, present only when ``nlist > 1``

The codes and centroids live in the *rotated* space: document vectors are
rotated by the (fixed, warm-up) rotation before assignment, and the
retrievers rotate queries before ADC search. Because the rotation is
orthonormal, dot-product similarity in the rotated space equals similarity
in the original space.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import pyterrier as pt
import numpy as np
import json
import faiss
from npids import Lookup
from pyterrier_dr.flex.core import IndexingMode
from pyterrier_dr.jpq.utils import code_type_from_Ks

from .retriever import RepCONCRetrieverPQ, RepCONCRetrieverFlat


@dataclass(slots=True)
class Metadata:
    """Serialisable metadata for a RepCONC index (stored as JSON)."""
    type: str
    format: str
    M: int
    Ks: int
    dsub: int
    doc_count: int
    rotation: bool
    nlist: int
    nprobe: int

    @classmethod
    def load(cls, path: str | Path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))

    def save(self, path: str | Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f)


class RepCONCIndex(pt.Artifact):
    """RepCONC (Zhan et al., WSDM'22) ADC index.

    Use :meth:`retriever_pq` (ADC) or :meth:`retriever_flat` (exact search
    over the PQ-reconstructed vectors) to get retrievers.

    Args:
        path (str | Path): Index path.

    Attributes:
        index_path (Path): Index path.

    """

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'repconc'

    # file names
    _META_FN = "pt_meta.json"
    _DOCNOS_FN = "docnos.npids"
    _ROTATION_FN = "rotation.f4"   # float32 [d, d]
    _CODES_FN = "codes.f4"         # uint8 or uint16 [N, M]
    _SUBVECS_FN = "subvecs.f4"     # float32 [M, Ks, dsub]
    _FAISS_FN = "faiss.index"      # IVF index, only when nlist > 1

    def __init__(self, path: str | Path):
        super().__init__(path)
        self.index_path = Path(path)

        self._meta: Metadata | None = None
        self._docnos: Lookup | None = None
        self._codes: np.ndarray | None = None
        self._subvecs: np.ndarray | None = None
        self._rotation: np.ndarray | None = None
        self._faiss_index: faiss.Index | None = None

    @property
    def meta(self) -> Metadata:
        """Meta data for this RepCONC index."""
        if self._meta is None:
            self._meta = Metadata.load(self.index_path / RepCONCIndex._META_FN)
        return self._meta  # type: ignore

    @property
    def docnos(self) -> Lookup:
        """Docno lookup."""
        if self._docnos is None:
            self._docnos = Lookup(self.index_path / RepCONCIndex._DOCNOS_FN)
        return self._docnos

    @property
    def codes(self) -> np.ndarray:
        """Document codes, shape ``[N, M]``."""
        if self._codes is None:
            shape = (self.meta.doc_count, self.meta.M)
            dtype = code_type_from_Ks(self.meta.Ks)
            self._codes = np.memmap(self.index_path / RepCONCIndex._CODES_FN, mode="r", dtype=dtype, shape=shape)
        return self._codes

    @property
    def subvecs(self) -> np.ndarray:
        """Learned centroids with shape ``[M, Ks, dsub]``."""
        if self._subvecs is None:
            shape = (self.meta.M, self.meta.Ks, self.meta.dsub)
            self._subvecs = np.memmap(self.index_path / RepCONCIndex._SUBVECS_FN, mode="r", dtype=np.float32, shape=shape)
        return self._subvecs

    @property
    def rotation(self) -> np.ndarray | None:
        """Fixed rotation matrix ``[d, d]``. None if no rotation was used."""
        if not self.meta.rotation:
            return None
        if self._rotation is None:
            d = self.meta.M * self.meta.dsub
            self._rotation = np.memmap(self.index_path / RepCONCIndex._ROTATION_FN, mode="r", dtype=np.float32, shape=(d, d))
        return self._rotation

    @property
    def d(self) -> int:
        """Embedding dimensionality."""
        return self.meta.M * self.meta.dsub

    def _load_faiss(self) -> faiss.Index | None:
        """Load the serialized IVF index (only present when nlist > 1)."""
        if self.meta.nlist == 1:
            return None
        if self._faiss_index is None:
            path = self.index_path / RepCONCIndex._FAISS_FN
            if not path.exists():
                raise FileNotFoundError(f"IVF index file not found: {path}")
            self._faiss_index = faiss.read_index(str(path))
            self._faiss_index.nprobe = self.meta.nprobe
        return self._faiss_index

    def __len__(self) -> int:
        return self.meta.doc_count

    @staticmethod
    def build(
        path: str | Path,
        docnos: list[str],
        codes: np.ndarray,        # [N, M]
        centroids: np.ndarray,    # [M, Ks, dsub]
        rotation: np.ndarray | None = None,  # [d, d]
        x_rot: np.ndarray | None = None,     # [N, d] rotated corpus (required for nlist > 1)
        nlist: int = 1,
        nprobe: int = 10,
        ivf_seed: int = 42,
        mode: IndexingMode = IndexingMode.create,
    ) -> "RepCONCIndex":
        """Build a RepCONC index from document codes and a learned codebook.

        Args:
            path: destination directory.
            docnos: document identifiers, in the same order as ``codes``/``x_rot``.
            codes: document codes, ``[N, M]`` (argmin assignments in the
                rotated space).
            centroids: learned centroids, ``[M, Ks, dsub]``.
            rotation: fixed rotation applied to queries and documents before
                assignment/search. If None, no rotation is applied.
            x_rot: rotated corpus vectors, ``[N, d]``; required when
                ``nlist > 1`` (to train the IVF k-means and populate the
                inverted lists).
            nlist: number of IVF clusters (``1`` = flat PQ, the default).
            nprobe: IVF probes per query (stored in the metadata; applied at
                load time).
            ivf_seed: k-means seed (IVF only).
            mode: how to handle an existing index path.
        """
        from .ivf import build_faiss_index

        if isinstance(mode, str):
            mode = IndexingMode[mode]

        # validate inputs before touching the filesystem, so a bad call does
        # not leave a half-created index directory behind
        M, Ks, dsub = centroids.shape
        d = M * dsub
        codes = np.asarray(codes, dtype=code_type_from_Ks(Ks))
        centroids = np.asarray(centroids, dtype=np.float32)
        rotation = np.ascontiguousarray(rotation, dtype=np.float32) if rotation is not None else None
        docnos = list(docnos)
        N = len(docnos)

        if codes.shape != (N, M):
            raise ValueError(f"codes must have shape ({N}, {M}), got {codes.shape}")
        if rotation is not None and rotation.shape != (d, d):
            raise ValueError(f"rotation must have shape ({d}, {d}), got {rotation.shape}")
        if nlist < 1:
            raise ValueError(f"nlist must be >= 1, got {nlist}")
        if nlist > 1:
            if x_rot is None:
                raise ValueError("x_rot (rotated corpus) is required when nlist > 1")
            x_rot = np.ascontiguousarray(x_rot, dtype=np.float32)
            if x_rot.shape != (N, d):
                raise ValueError(f"x_rot must have shape ({N}, {d}), got {x_rot.shape}")

        path = Path(path)
        if path.exists():
            if mode == IndexingMode.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {path}. If you want to delete and re-create an existing index, you can pass mode="overwrite"')
        path.mkdir(parents=True, exist_ok=True)

        # docnos
        Lookup.build(docnos, path / RepCONCIndex._DOCNOS_FN)
        # sub-vectors (centroids)
        with open(path / RepCONCIndex._SUBVECS_FN, "wb") as f:
            centroids.tofile(f)
        # rotation (optional)
        if rotation is not None:
            with open(path / RepCONCIndex._ROTATION_FN, "wb") as f:
                rotation.tofile(f)
        # codes
        with open(path / RepCONCIndex._CODES_FN, "wb") as f:
            codes.tofile(f)

        # IVF index (nlist > 1): train the coarse quantiser and store the
        # fully populated index
        nlist_eff = nlist
        if nlist > 1:
            nlist_eff = min(nlist, N)
            ivfpq = build_faiss_index(d=d, M=M, nbits=int(round(np.log2(Ks))),
                                       centroids=centroids, codes=codes,
                                       x_rot=x_rot, nlist=nlist_eff, seed=ivf_seed)
            faiss.write_index(ivfpq, str(path / RepCONCIndex._FAISS_FN))

        Metadata(
            type=RepCONCIndex.ARTIFACT_TYPE,
            format=RepCONCIndex.ARTIFACT_FORMAT,
            M=int(M),
            Ks=int(Ks),
            dsub=int(dsub),
            doc_count=int(N),
            rotation=rotation is not None,
            nlist=int(nlist_eff),
            nprobe=int(min(nprobe, nlist_eff)),
        ).save(path / RepCONCIndex._META_FN)

        return RepCONCIndex(path)

    def retriever_pq(self, topk: int = 1000) -> RepCONCRetrieverPQ:
        """Return an ADC (:class:`RepCONCRetrieverPQ`) retriever.

        With ``nlist == 1`` the index is an in-memory flat PQ built from the
        stored codes and centroids (the JPQ pattern); with ``nlist > 1`` the
        serialized IVF index is loaded.

        Args:
            topk (int): Number of documents to return per query.
        """
        if self.meta.nlist == 1:
            return RepCONCRetrieverPQ(self.docnos, self.codes, self.subvecs,
                                      topk=topk, rotation=self.rotation, name="RepCONC-PQ")
        return RepCONCRetrieverPQ(self.docnos, self.codes, self.subvecs,
                                  topk=topk, rotation=self.rotation, name="RepCONC-IVFPQ",
                                  faiss_index=self._load_faiss(), nprobe=self.meta.nprobe)

    def retriever_flat(self, topk: int = 1000) -> RepCONCRetrieverFlat:
        """Return a flat inner-product retriever (:class:`RepCONCRetrieverFlat`)
        over the PQ-reconstructed vectors (exact w.r.t. the codes, useful as a
        sanity check against the ADC retriever).

        Args:
            topk (int): Number of documents to return per query.
        """
        return RepCONCRetrieverFlat(self.docnos, self.codes, self.subvecs,
                                    topk=topk, rotation=self.rotation, name="RepCONC-Flat")

    def get_corpus_iter(self, full_vecs: bool = False):
        docnos = self.docnos
        codes = self.codes
        subvecs = self.subvecs
        for i in range(len(self)):
            if full_vecs:
                parts = [subvecs[m, codes[i, m], :] for m in range(self.meta.M)]
                vec = np.concatenate(parts).astype(np.float32)
            else:
                vec = codes[i, :].tolist()
            yield {'docno': docnos.fwd[i], 'doc_vec': vec}
