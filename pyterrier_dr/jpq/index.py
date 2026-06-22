from dataclasses import asdict, dataclass
from pathlib import Path
import shutil
import pyterrier as pt
import numpy as np
import more_itertools
import json
from npids import Lookup
from pyterrier_dr.flex.core import IndexingMode
from pyterrier_dr import FlexIndex
from pyterrier_dr.jpq.pq import (
    ProductQuantizerFAISSIndexPQ, 
    ProductQuantizerFAISSIndexPQOPQ,
)
from pyterrier_dr.jpq.utils import code_type_from_Ks
from .retriever import JPQRetrieverFlat, JPQRetrieverPrune, JPQRetrieverPQ


@dataclass(slots=True)
class Metadata:
    """Serialisable metadata for a JPQ index (stored as JSON)."""
    type: str
    format: str
    M: int
    Ks: int
    dsub: int
    doc_count: int
    opq: bool

    @classmethod
    def load(cls, path: str | Path):
        with open(path, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))

    def save(self, path: str | Path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f)


class JPQIndex(pt.Artifact):
    """Joint Product Quantization index.

    Use :meth:`retriever_pq` to get a PQ retriever.

    Args:
        path (str | Path): Index path.
    
    Attributes:
        index_path (Path): Index path.

    """

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'jpq'

    # file names
    _META_FN = "pt_meta.json"
    _DOCNOS_FN = "docnos.npids"
    _OPQ_FN = "opq.f4"       # float32 [d, d]
    _CODES_FN = "codes.f4"     # uint8 or uint16 or uint32  [N, M]
    _SUBVECS_FN = "subvecs.f4" # float32 [M, Ks, dsub]

    def __init__(self, path: str | Path):
        super().__init__(path)
        self.index_path = Path(path)

        self._meta: Metadata | None = None
        self._dvecs: np.ndarray | None = None
        self._codes: np.ndarray | None = None
        self._docnos: Lookup | None = None
        self._opq: np.ndarray | None = None

    @property
    def meta(self) -> Metadata:
        """Meta data for this JPQ index."""
        if self._meta is None:
            self._meta = Metadata.load(self.index_path / JPQIndex._META_FN)
        return self._meta # type: ignore

    @property
    def docnos(self) -> Lookup:
        """Docno lookup."""
        if self._docnos is None:
            self._docnos = Lookup(self.index_path / JPQIndex._DOCNOS_FN)
        return self._docnos

    @property
    def codes(self) -> np.ndarray:
        """PQ code book."""
        if self._codes is None:
            shape = (self.meta.doc_count, self.meta.M)
            dtype = code_type_from_Ks(self.meta.Ks)
            self._codes = np.memmap(self.index_path / JPQIndex._CODES_FN, mode="r", dtype=dtype, shape=shape)
        return self._codes

    @property
    def dvecs(self) -> np.ndarray:
        """PQ centroids with shape [M, Ks, dsub].
        
        Note:
            M -> numer of splits
            Ks -> number of centroids per split
            dsub -> dimension of subvector
        """
        if self._dvecs is None:
            shape = (self.meta.M, self.meta.Ks, self.meta.dsub)
            self._dvecs = np.memmap(self.index_path / JPQIndex._SUBVECS_FN, mode="r", dtype=np.float32, shape=shape)
        return self._dvecs
    
    @property
    def opq(self) -> np.ndarray | None:
        """Optional OPQ rotation matrix. None if OPQ is disabled."""
        if not self.meta.opq:
            return None
        if self._opq is None:
            shape = (self.meta.M * self.meta.dsub, self.meta.M * self.meta.dsub)
            self._opq = np.memmap(self.index_path / JPQIndex._OPQ_FN, mode="r", dtype=np.float32, shape=shape)
        return self._opq

    def __len__(self) -> int:
        return self.meta.doc_count


    @staticmethod
    def build(
        path: str | Path, 
        docnos: list[str], 
        codes: np.ndarray, # [N, M]
        centroids: np.ndarray, # [M, Ks, dsub]
        opq = None | np.ndarray,
        mode = IndexingMode.create
    ) -> "JPQIndex": # type: ignore
        path = Path(path)

        if path.exists():
            if mode == IndexingMode.overwrite:
                shutil.rmtree(path)
            else:
                raise RuntimeError(f'Index already exists at {path}. If you want to delete and re-create an existing index, you can pass mode="overwrite"')
        path.mkdir(parents=True, exist_ok=True)

        _, Ks, _ = centroids.shape

        centroids = np.asarray(centroids, dtype=np.float32)
        opq = np.asarray(opq, dtype=np.float32) if opq is not None else None
        codes = np.asarray(codes, dtype=code_type_from_Ks(Ks))
        docnos = list(docnos)

        if centroids.ndim != 3:
            raise ValueError(f"centroids must have shape [M, Ks, dsub], got {centroids.shape}")
        if codes.ndim != 2:
            raise ValueError(f"codes must have shape [N, M], got {codes.shape}")

        M, _, dsub = centroids.shape # we already have Ks
        N, M_codes = codes.shape
        if M != M_codes:
            raise ValueError(f"M mismatch: embs={M}, codes={M_codes}")

        # We save the docnos
        Lookup.build(docnos, path / JPQIndex._DOCNOS_FN)

        # We save the sub-vectors
        with open(path / JPQIndex._SUBVECS_FN, "wb") as f:
            centroids.tofile(f)

        if opq is not None:
            with open(path / JPQIndex._OPQ_FN, "wb") as f:
                opq.tofile(f)

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
            opq = opq is not None
        ).save(path / JPQIndex._META_FN)

        return JPQIndex(path)

    @staticmethod
    def build_zero_shot_index(
        source_index: "JPQIndex",
        target_index: FlexIndex,
        path: str,
        batch_size: int = 8192,
        mode: IndexingMode = IndexingMode.create,
    ) -> "JPQIndex":
        """Return a :class:`JPQIndex` that quantizes `target_index` using learned centroids in `source_index`.

        Args:
            source_index (JPQIndex): An existing :class:`JPQIndex`.
            target_index (FlexIndex): Target index with document vectors to be quantized.
            path (str): Path to save the resulting :class:`JPQIndex`.
            batch_size (int, optional): Batch size to iterate over the corpus of ``target_index``. Defaults to 8192.
            mode (IndexingMode, optional): How to handle an existing index path.
                Use :attr:`IndexingMode.overwrite` to allow overwriting. Defaults to :attr:`IndexingMode.create`.

        """
        opq = source_index.opq
        centroids = source_index.dvecs
        pq_cls = ProductQuantizerFAISSIndexPQ if opq is None else ProductQuantizerFAISSIndexPQOPQ
        pq = pq_cls(M=centroids.shape[0], Ks=centroids.shape[1])
        if opq is not None:
            pq.opq = opq
        pq.centroids = centroids
        pq.d = centroids.shape[0] * centroids.shape[-1]

        # encode_batch for all docs from targetindex to get codes for new index
        all_codes = []
        all_docnos = []
        for docs_batch in more_itertools.batched(target_index.get_corpus_iter(), batch_size):
            vecs = []
            for doc in docs_batch:
                vecs.append(doc["doc_vec"])
                all_docnos.append(doc["docno"])

            X_batch = np.stack(vecs, axis=0)
            codes = pq.encode_batch(X_batch, selected=np.arange(X_batch.shape[0]), bs=batch_size, verbose=False)
            all_codes.append(codes)
            
        all_codes = np.concatenate(all_codes, axis=0)

        # call and return JPQIndex.build() using the docnos from targetindex, centroids from jpqsource index, and newly computed codes
        return JPQIndex.build(
            path=path,
            docnos=all_docnos,
            codes=all_codes,
            centroids=centroids,
            opq=opq,
            mode=mode,
        )
    
    def get_corpus_iter(self, full_vecs=False):
        docnos = self.docnos
        codes = self.codes
        assert not full_vecs, "full_vecs not yet supported"
        for i in range(len(self)):
            docdict = {'docno' : docnos.fwd[i], 'codes' : codes[i, :].tolist() }
            yield docdict

    def retriever_pq(self, topk: int = 1000) -> "JPQRetrieverPQ":
        """Return a PQ retriever.

        Args:
            topk (int, optional): Number of documents to return per query. Defaults to 1000.

        """
        return JPQRetrieverPQ(self.docnos, self.codes, self.dvecs, topk=topk, name="JPQ-PQ", opq = self.opq)

    def retriever_flat(self, topk: int = 1000, gpu : bool =False) -> "JPQRetrieverFlat":
        """Return a flat inner-product retriever (:class:`JPQRetrieverFlat`) over reconstructed vectors.

        Args:
            topk (int, optional): Number of documents to return per query. Defaults to 1000.
            gpu (bool, optional): Whether to load the flat index into GPU.

        """
        return JPQRetrieverFlat(self.docnos, self.codes, self.dvecs, topk=topk, name="JPQ-Flat", opq = self.opq, gpu=gpu)

    def retriever_prune(self, topk: int = 1000, ub_inflation: float =1.) -> "JPQRetrieverPrune":
        """Return a dynamically pruned retriever.

        Args:
            topk (int, optional): Number of documents to return per query. Defaults to 1000.
            ub_inflation (float, optional): Multiplier applied to pruning upper bounds; 
                larger values prune less aggressively. Defaults to 1.0.

        See Also:
            https://arxiv.org/pdf/2505.00560
        
        """
        return JPQRetrieverPrune(self.docnos, self.codes, self.dvecs, topk=topk, name="JPQ-Prune", ub_inflation=ub_inflation, opq = self.opq)