from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import pyterrier as pt
import numpy as np
import json
from npids import Lookup
from pyterrier_dr.flex.core import IndexingMode
from .retriever import JPQRetrieverFlat, JPQRetrieverPrune


@dataclass(slots=True)
class Metadata:
    """Serialisable metadata for a JPQ index (stored as JSON)."""
    type: str
    format: str
    M: int
    Ks: int
    dsub: int
    doc_count: int

    @classmethod
    def load(cls, path: str | Path) -> Metadata: # type: ignore
        with open(path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))

    def save(self, path: str | Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f)

            
class JPQIndex(pt.Artifact):

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'jpq'

    # file names
    _META_FN = "pt_meta.json"
    _DOCNOS_FN = "docnos.npids"
    _CODES_FN = "codes.f4"     # uint8   [N, M]
    _SUBVECS_FN = "subvecs.f4" # float32 [M, Ks, dsub]

    def __init__(self, path: str | Path):
        super().__init__(path)
        self.index_path = Path(path)

        self._meta: Metadata | None = None
        self._dvecs: np.ndarray | None = None
        self._codes: np.ndarray | None = None
        self._docnos: Lookup | None = None


    @property
    def meta(self) -> Metadata:
        if self._meta is None:
            self._meta = Metadata.load(self.index_path / JPQIndex._META_FN)
        return self._meta # type: ignore


    @property
    def docnos(self) -> Lookup:
        if self._docnos is None:
            self._docnos = Lookup(self.index_path / JPQIndex._DOCNOS_FN)
        return self._docnos


    @property
    def codes(self) -> np.ndarray:
        if self._codes is None:
            shape = (self.meta.doc_count, self.meta.M)
            self._codes = np.memmap(self.index_path / JPQIndex._CODES_FN, mode="r", dtype=np.uint8, shape=shape)
        return self._codes


    @property
    def dvecs(self) -> np.ndarray:
        if self._dvecs is None:
            shape = (self.meta.M, self.meta.Ks, self.meta.dsub)
            self._dvecs = np.memmap(self.index_path / JPQIndex._SUBVECS_FN, mode="r", dtype=np.float32, shape=shape)
        return self._dvecs


    def __len__(self) -> int:
        return self.meta.doc_count


    @staticmethod
    def build(
        path: str | Path, 
        docnos: list[str], 
        codes: np.ndarray, # [N, M]
        centroids: np.ndarray, # [M, Ks, dsub]
        mode = IndexingMode.create
    ) -> "JPQIndex": # type: ignore
        path = Path(path)

        if path.exists():
            if mode == IndexingMode.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {path}. If you want to delete and re-create an existing index, you can pass mode="overwrite"')
        path.mkdir(parents=True, exist_ok=True)

        centroids = np.asarray(centroids, dtype=np.float32)
        codes = np.asarray(codes, dtype=np.uint8)
        docnos = list(docnos)

        if centroids.ndim != 3:
            raise ValueError(f"centroids must have shape [M, Ks, dsub], got {centroids.shape}")
        if codes.ndim != 2:
            raise ValueError(f"codes must have shape [N, M], got {codes.shape}")

        M, Ks, dsub = centroids.shape
        N, M_codes = codes.shape
        if M != M_codes:
            raise ValueError(f"M mismatch: embs={M}, codes={M_codes}")

        # We save the docnos
        Lookup.build(docnos, path / JPQIndex._DOCNOS_FN)

        # We save the sub-vectors
        with open(path / JPQIndex._SUBVECS_FN, "wb") as f:
            centroids.tofile(f)

        # We save the codes
        with open(path / JPQIndex._CODES_FN, "wb") as f:
            codes.tofile(f)

        # We save the metadata
        Metadata(
            type=JPQIndex.ARTIFACT_TYPE,
            format=JPQIndex.ARTIFACT_FORMAT,
            M=int(M),
            Ks=int(Ks),
            dsub=int(dsub),
            doc_count=int(N),
        ).save(path / JPQIndex._META_FN)


    def retriever_flat(self, topk: int = 1000) -> "JPQRetrieverFlat":
        return JPQRetrieverFlat(self.docnos, self.codes, self.dvecs, topk=topk, name="JPQ-Flat")


    def retriever_prune(self, topk: int = 1000, ub_inflation: float =1.) -> "JPQRetrieverPrune":
        return JPQRetrieverPrune(self.docnos, self.codes, self.dvecs, topk=topk, name="JPQ-Prune", ub_inflation=ub_inflation)