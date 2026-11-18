"""Trainer for RepCONC (Zhan et al., WSDM'22).

Implements the paper's two-stage training of learnable product quantizers
with constrained (Sinkhorn) clustering:

* **Stage 1** (:meth:`RepCONCTrainer.train_stage1`) jointly optimises the
  (bi)encoder and the PQ centroids on *text triples*, minimising the
  ranking loss on quantised documents plus the (weighted) clustering loss
  (paper Eqs. 8-10), with Sinkhorn-constrained assignments.
* **Stage 2** (:meth:`RepCONCTrainer.train_stage2`) fine-tunes with *fixed*
  document codes (argmin under the stage-1 centroids) and dynamic negatives
  retrieved from the current PQ index (the paper's optional second stage,
  which the official code implements as ``finetune_jpq``).

Afterwards, :meth:`RepCONCTrainer.repconc_index` materialises a searchable
:class:`~pyterrier_dr.repconc.index.RepCONCIndex` (ADC, optional IVF).

Conventions and documented deviations:

* The fixed OPQ rotation and initial centroids come from a warm-up that
  reuses :class:`pyterrier_dr.jpq.pq.ProductQuantizerFAISSIndexPQOPQ`
  (the official warm-up does the same: faiss OPQ+PQ on a corpus sample).
* Stage-1 negatives: all documents of the batch (each query's positive plus
  its explicit negatives) are quantised in a *single* Sinkhorn batch, and
  in-batch negatives are included (official behaviour). There is no
  false-negative masking. ``hardneg_topk`` implements the official
  ``dynamic_topk_hard_negative`` masking.
* Final index codes are the *unconstrained* argmin assignments (the official
  evaluation loads the model with ``use_constraint=False``); the constraint
  is applied only during training.
* :meth:`RepCONCTrainer.repconc_index` re-encodes the corpus with the
  trained document encoder when ``corpus_text`` is provided (the official
  convention); otherwise the :class:`~pyterrier_dr.flex.flex.FlexIndex`
  vectors (from the *pre-training* encoder) are used, as FlexIndex stores
  vectors, not text.
* Learning rates default to the paper's (§4.1: encoder 5e-6, centroids
  2e-4 for stage 1); the official script uses 2e-5 / 5e-4 (stage 1) and
  2e-6 / 2e-5 (stage 2) — all overridable. Stage 2 trains the query side
  and the centroids; the document encoder is frozen by construction (stage-2
  document vectors are never re-encoded, so it has no gradient path).
* No GradCache (the official implementation uses it for memory); the
  straight-through gradient policy (paper Eqs. 9-10) is applied directly.
"""

import atexit
import logging
import math
import os
import shutil
import tempfile
import time
from typing import Iterable, Literal

import faiss
import numpy as np
import pandas as pd
import torch

import pyterrier as pt
from ir_measures import RR, Recall, nDCG

from pyterrier_dr import FlexIndex
from pyterrier_dr.biencoder import BiEncoder
from pyterrier_dr.flex.core import IndexingMode
from pyterrier_dr.jpq.data import prepare_validation_data
from pyterrier_dr.jpq.pq import ProductQuantizerFAISSIndexPQOPQ
from pyterrier_dr.jpq.utils import code_type_from_Ks, timer
from pyterrier_dr.util import ensure_faiss_omp_compat, infer_device

from .checkpointing import _load_checkpoint, _save_checkpoint
from .data import get_dataloader, normalize_stage1, normalize_stage2
from .index import RepCONCIndex
from .ivf import pack_pq_codes
from .losses import clustering_loss, paper_lambda, ranking_loss, straight_through
from .model import RepCONCModel
from .pq import RepCONCPQ
from .retriever import RepCONCRetrieverPQ

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S', force=True)
logger = logging.getLogger(__name__)


def _argmin_codes_np(x_rot: np.ndarray, centroids: np.ndarray,
                     batch_size: int = 10_000) -> np.ndarray:
    """L2-argmin PQ codes for rotated vectors.

    The same assignment as
    :meth:`pyterrier_dr.repconc.pq.RepCONCPQ.quantize` with
    ``use_constraint=False`` (the mode used for final index codes).
    """
    N, d = x_rot.shape
    M, K, dsub = centroids.shape
    codes = np.empty((N, M), dtype=code_type_from_Ks(K))
    cc = (centroids.astype(np.float32) ** 2).sum(-1)  # [M, K]
    for i in range(0, N, batch_size):
        x_m = np.ascontiguousarray(x_rot[i:i + batch_size]).reshape(-1, M, dsub)
        xc = np.einsum("nmd,mkd->nmk", x_m, centroids.astype(np.float32), optimize=True)
        dist = (x_m ** 2).sum(-1)[:, :, None] + cc[None] - 2 * xc
        codes[i:i + batch_size] = dist.argmin(-1)
    return codes


class RepCONCTrainer:
    """Trainer for RepCONC (Zhan et al., WSDM'22).

    Expects an existing :class:`~pyterrier_dr.flex.flex.FlexIndex` of document
    vectors (from the pre-training encoder) and a :class:`BiEncoder`. It
    warms up OPQ+PQ on a corpus sample, then trains per
    :meth:`train_stage1` / :meth:`train_stage2`. After fitting, call
    :meth:`repconc_index` to build a searchable :class:`RepCONCIndex`.

    Args:
        biencoder (BiEncoder): the query/document encoder. Must implement
            :meth:`BiEncoder.encode_queries_torch` and
            :meth:`BiEncoder.encode_docs_torch` (with gradients) for training.
        index (FlexIndex): the dense document index (provides the document
            vectors and docnos used for the warm-up and for building the
            final index).
        M (int): number of subspaces. The embedding dimension must be
            divisible by this. Defaults to 8.
        nbits (int): bits per subspace code (codebook size ``2**nbits``).
            Defaults to 8.
        device (str | torch.device | None): device. If None, CUDA if
            available, else CPU.
        sk_epsilon (float): Sinkhorn entropic regularisation. Defaults to
            0.003 (the paper/official default).
        sk_iters (int): Sinkhorn-Knopp iterations. Defaults to 100.
        seed (int): RNG seed (data shuffling, warm-up sampling). Defaults to 42.

    Attributes:
        fitted (bool): whether :meth:`fit` has been called.
        model (RepCONCModel): the model container (set after warm-up).
        pq (RepCONCPQ): the learnable PQ (set after warm-up).
    """

    def __init__(
        self,
        biencoder: BiEncoder,
        index: FlexIndex,
        M: int = 8,
        nbits: int = 8,
        device=None,
        sk_epsilon: float = 0.003,
        sk_iters: int = 100,
        seed: int = 42,
    ):
        self.biencoder = biencoder
        self.index = index
        self.d = index.payload()[2]['vec_size']
        if self.d % M != 0:
            raise ValueError(f"d={self.d} not divisible by M={M}")
        self.M = M
        if not (1 <= nbits <= 16):
            raise ValueError(f"nbits should be in range 1-16, got {nbits}")
        self.nbits = nbits
        self.Ks = 2 ** nbits
        self.device = infer_device(device)
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.seed = seed

        self.fitted = False
        self.model: RepCONCModel | None = None
        self.pq: RepCONCPQ | None = None

        self._docnos: list[str] | None = None
        self._corpus_vecs: np.ndarray | None = None   # [N, d] (pre-training encoder)
        self._rotation_np: np.ndarray | None = None   # [d, d]
        self._codes: np.ndarray | None = None         # [N, M]
        self._docno2pos: dict[str, int] | None = None

        self._optimizer: torch.optim.Optimizer | None = None
        self._stage: str | None = None
        self._step: int = 0
        self._best_metric: float = float("-inf")
        logger.info(f"[RepCONCTrainer init] d={self.d} M={self.M} nbits={self.nbits} device={self.device}")

    # ------------------------------------------------------------------ warm-up

    def _warmup(self, sample_size: int = 10_000) -> None:
        """Learn the initial OPQ rotation and PQ centroids (faiss OPQ+PQ on a
        random corpus sample), initialise :attr:`model`, and compute the
        full-corpus argmin codes. Reuses the JPQ warm-up implementation."""
        ensure_faiss_omp_compat()
        if self.model is not None:
            return
        docnos, vecs, _ = self.index.payload(return_docnos=True, return_dvecs=True)
        self._docnos = [str(x) for x in docnos.fwd]
        self._corpus_vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        N, d = self._corpus_vecs.shape
        if d != self.d:
            raise ValueError(f"index vectors have dim {d}, expected {self.d}")
        self._docno2pos = {docno: i for i, docno in enumerate(self._docnos)}

        sample_size = min(sample_size, N)
        rng = np.random.default_rng(self.seed)
        sample_ids = np.sort(rng.choice(N, size=sample_size, replace=False))
        logger.info(f"[Warmup] training OPQ+PQ (M={self.M}, Ks={self.Ks}) on {sample_size} of {N} docs")
        pq_opq = ProductQuantizerFAISSIndexPQOPQ(self.M, Ks=self.Ks)
        with timer("Warmup / OPQ+PQ train"):
            pq_opq.fit(self._corpus_vecs[sample_ids])
        self._rotation_np = np.ascontiguousarray(pq_opq.opq, dtype=np.float32)
        centroids = np.ascontiguousarray(pq_opq.centroids, dtype=np.float32)

        self.pq = RepCONCPQ(
            centroids=torch.from_numpy(centroids).to(self.device),
            rotation=torch.from_numpy(self._rotation_np).to(self.device),
            epsilon=self.sk_epsilon, iters=self.sk_iters, use_constraint=True)
        self.model = RepCONCModel(self.biencoder, self.pq).to(self.device)

        with timer("Warmup / full-corpus argmin codes"):
            self._codes = _argmin_codes_np(self._corpus_vecs @ self._rotation_np, centroids)
        logger.info(f"[Warmup] codes {self._codes.shape} {self._codes.dtype}")

    # --------------------------------------------------------------- stage 1

    def train_stage1(
        self,
        triples: Iterable[dict],
        *,
        batch_size: int = 32,
        epochs: int = 1,
        total_steps: int | None = None,
        lr_encoder: float = 5e-6,
        lr_centroid: float = 2e-4,
        lambda_mse: float | None = None,
        hardneg_topk: int | None = None,
        temperature: float = 1.0,
        eval_queries: pd.DataFrame | None = None,
        eval_qrels: pd.DataFrame | None = None,
        valid_every: int = 100,
        patience: int = 5,
        metric: str = "nDCG@10",
        checkpoint_dir: str | None = None,
        resume: bool = False,
        encode_batch_size: int = 256,
    ) -> float:
        """Stage-1 training: encoder + centroids, ranking + clustering loss.

        Args:
            triples: stage-1 text triples, each
                ``{"qid"?: str, "query": str, "doc_id_a": <positive text>,
                "doc_id_b": <negative text> | [<negative text>, ...]}``.
            batch_size: queries per training step (all of a batch's documents
                are quantised in a single Sinkhorn batch). Defaults to 32.
            epochs: number of passes over ``triples`` (used unless
                ``total_steps`` is given). Defaults to 1.
            total_steps: explicit step budget (overrides ``epochs``).
            lr_encoder: learning rate for the biencoder parameters. Defaults
                to 5e-6 (paper §4.1; the official script uses 2e-5).
            lr_centroid: learning rate for the PQ centroids. Defaults to
                2e-4 (paper; the official script uses 5e-4).
            lambda_mse: clustering-loss weight (paper Eq. 8). Defaults to
                :func:`pyterrier_dr.repconc.losses.paper_lambda` for ``M``
                (the official script default is 1e-4).
            hardneg_topk: if set, only the ``hardneg_topk`` most similar
                non-positive documents per query remain active negatives
                (the official ``dynamic_topk_hard_negative``).
            temperature: similarity temperature (the official default is 1).
            eval_queries / eval_qrels: validation data (``qid``/``query`` and
                ``qid``/``docno``/``label``). Qrels are filtered to the
                index's documents. If omitted, validation and early stopping
                are disabled.
            valid_every: run validation every this many steps (ignored
                without eval data, in which case a periodic checkpoint is
                written). Defaults to 100.
            patience: validations without improvement before early stopping.
                Defaults to 5.
            metric: validation metric for early stopping. Defaults to
                ``nDCG@10`` (RR@10, Recall@1000 and nDCG@10 are all computed).
            checkpoint_dir: where to write ``best.pt``/``last.pt``. If None,
                a temporary directory is used (deleted at exit).
            resume: resume from ``last.pt`` in ``checkpoint_dir`` if present.
            encode_batch_size: batch size for the biencoder's text encoding.

        Returns:
            The best validation metric achieved (``-inf`` if no eval data).
        """
        if self.model is None:
            self._warmup()
        if lambda_mse is None:
            lambda_mse = paper_lambda(self.M)
        items = normalize_stage1(triples)
        if not items:
            raise ValueError("stage 1 training data is empty")

        optimizer = torch.optim.AdamW([
            {"params": self.model.encoder_parameters(), "lr": lr_encoder},
            {"params": [self.pq.centroids], "lr": lr_centroid},
        ], weight_decay=0.0)
        self.model.set_train(True)
        self._optimizer = optimizer
        self._stage = "stage1"

        if checkpoint_dir is None:
            checkpoint_dir = tempfile.mkdtemp("repconc_checkpoints")
            atexit.register(shutil.rmtree, checkpoint_dir, ignore_errors=True)
        best_metric = float("-inf")
        valids_since_improve = 0
        best_step = 0
        if resume:
            last_path = os.path.join(checkpoint_dir, "last.pt")
            if os.path.isfile(last_path):
                st, bm = _load_checkpoint(last_path, model=self.model, optimizer=optimizer, trainer_self=self)
                best_metric, self._step = bm, st
                logger.info(f"[CKPT] Resumed from {last_path}: step={st}, best={best_metric:.6f}")

        eval_queries, eval_qrels = prepare_validation_data(eval_queries, eval_qrels, self._docnos)
        if eval_queries is None or eval_qrels is None:
            logger.warning("[RepCONC][stage1] no eval data: validation and early stopping disabled")

        loader = get_dataloader(items, batch_size, shuffle=True, seed=self.seed)
        n_steps = total_steps if total_steps is not None else epochs * max(1, math.ceil(len(items) / batch_size))
        logger.info(f"[RepCONC][stage1] {len(items)} triples, batch={batch_size}, steps={n_steps}, "
                    f"lambda_mse={lambda_mse}, hardneg_topk={hardneg_topk}")

        best_metric = self._train_loop(
            n_steps, loader, optimizer,
            lambda batch: self._stage1_step(batch, optimizer, temperature, lambda_mse,
                                            hardneg_topk, encode_batch_size),
            tag="stage1", eval_queries=eval_queries, eval_qrels=eval_qrels,
            valid_every=valid_every, patience=patience, metric=metric,
            checkpoint_dir=checkpoint_dir, best_metric=best_metric,
            valids_since_improve=valids_since_improve, best_step=best_step)
        return best_metric

    def _stage1_step(self, batch: list[dict], optimizer: torch.optim.Optimizer,
                     temperature: float, lambda_mse: float,
                     hardneg_topk: int | None, encode_batch_size: int) -> float:
        B = len(batch)
        queries = [it["query"] for it in batch]
        doc_texts: list[str] = []
        starts: list[int] = []
        counts: list[int] = []
        for it in batch:
            starts.append(len(doc_texts))
            counts.append(1 + len(it["negs"]))
            doc_texts.append(it["pos"])
            doc_texts.extend(it["negs"])
        Nmax = max(counts)

        # document assignments: ALL documents of the batch in a single
        # Sinkhorn batch (the official convention), so the per-subspace
        # uniform-utilisation constraint holds across the whole ranking set
        # (encode_* return embeddings in the *rotated* space)
        q = self.model.encode_queries(queries, batch_size=encode_batch_size).to(self.device)     # [B, d]
        d = self.model.encode_docs(doc_texts, batch_size=encode_batch_size).to(self.device)       # [Ntot, d]
        codes = self.pq.quantize(d)                                                               # [Ntot, M] (no grad, constrained)
        d_hat = self.pq.decode(codes)                                                             # [Ntot, d], grad via centroids
        d_st = straight_through(d, d_hat)                                                         # fwd: d_hat; bwd: d AND d_hat

        idx = torch.full((B, Nmax), 0, dtype=torch.long, device=self.device)
        valid = torch.zeros((B, Nmax), dtype=torch.bool, device=self.device)
        for b in range(B):
            s, n = starts[b], counts[b]
            idx[b, :n] = torch.arange(s, s + n, device=self.device)
            valid[b, :n] = True
        sims = (d_st[idx] * q.unsqueeze(1)).sum(-1) / temperature                                # [B, Nmax]
        sims = sims.masked_fill(~valid, -10000.0)
        labels = torch.zeros(B, dtype=torch.long, device=self.device)  # each row's positive is first
        loss_r = ranking_loss(sims, labels, hardneg_topk=hardneg_topk)
        loss_m = clustering_loss(d_hat, d)
        loss = loss_r + lambda_mse * loss_m

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    # --------------------------------------------------------------- stage 2

    def train_stage2(
        self,
        pairs: Iterable[dict],
        *,
        batch_size: int = 32,
        epochs: int = 1,
        total_steps: int | None = None,
        lr_query: float = 2e-6,
        lr_centroid: float = 2e-5,
        temperature: float = 1.0,
        dynamic_negs: int = 10,
        eval_queries: pd.DataFrame | None = None,
        eval_qrels: pd.DataFrame | None = None,
        valid_every: int = 100,
        patience: int = 5,
        metric: str = "nDCG@10",
        checkpoint_dir: str | None = None,
        resume: bool = False,
        encode_batch_size: int = 256,
    ) -> float:
        """Stage-2 training: fixed codes, dynamic negatives, ranking loss only.

        At the start of the stage the corpus codes are recomputed as the
        unconstrained argmin assignments of the (stage-1) centroids and held
        fixed for the rest of the stage. Negatives are the top-``dynamic_negs``
        documents (excluding the positive) retrieved from an in-memory PQ
        index whose centroids are re-synced with the current (training)
        centroids at every step. The document encoder is frozen by
        construction: stage-2 document vectors are never re-encoded, so it
        has no gradient path.

        Args:
            pairs: stage-2 pairs, each
                ``{"qid"?: str, "query": str, "doc_id_a": <positive docno>}``.
            batch_size: queries per training step. Defaults to 32.
            epochs / total_steps: as in :meth:`train_stage1`.
            lr_query: learning rate for the biencoder parameters. Defaults to
                2e-6 (official script; the paper reports marginal gains).
            lr_centroid: learning rate for the PQ centroids. Defaults to 2e-5.
            temperature: similarity temperature.
            dynamic_negs: dynamic negatives per query. Defaults to 10 (the
                official finetune script uses 200).
            eval_queries / eval_qrels / valid_every / patience / metric /
            checkpoint_dir / resume / encode_batch_size: as in
            :meth:`train_stage1`.

        Returns:
            The best validation metric achieved (``-inf`` if no eval data).
        """
        if self.model is None:
            self._warmup()
        items = normalize_stage2(pairs)
        if not items:
            raise ValueError("stage 2 training data is empty")
        unknown = [it["docno"] for it in items if it["docno"] not in self._docno2pos]
        if unknown:
            raise ValueError(f"stage 2 pairs reference {len(unknown)} docnos not in the index, e.g. {unknown[:3]}")
        N = len(self._docnos)
        dynamic_negs = max(0, min(dynamic_negs, N - 1))
        if dynamic_negs == 0:
            logger.warning("[RepCONC][stage2] dynamic_negs=0: no negatives, ranking loss is uninformative")

        # fix the codes for this stage (argmin under the current centroids)
        with timer("Stage2 / recompute fixed codes"):
            self._recompute_codes()

        optimizer = torch.optim.AdamW([
            {"params": self.model.encoder_parameters(), "lr": lr_query},
            {"params": [self.pq.centroids], "lr": lr_centroid},
        ], weight_decay=0.0)
        self.model.set_train(True)
        self._optimizer = optimizer
        self._stage = "stage2"

        neg_index = self._build_dynamic_neg_index()

        if checkpoint_dir is None:
            checkpoint_dir = tempfile.mkdtemp("repconc_checkpoints")
            atexit.register(shutil.rmtree, checkpoint_dir, ignore_errors=True)
        best_metric = float("-inf")
        valids_since_improve = 0
        best_step = 0
        if resume:
            last_path = os.path.join(checkpoint_dir, "last.pt")
            if os.path.isfile(last_path):
                st, bm = _load_checkpoint(last_path, model=self.model, optimizer=optimizer, trainer_self=self)
                best_metric, self._step = bm, st
                logger.info(f"[CKPT] Resumed from {last_path}: step={st}, best={best_metric:.6f}")

        eval_queries, eval_qrels = prepare_validation_data(eval_queries, eval_qrels, self._docnos)
        if eval_queries is None or eval_qrels is None:
            logger.warning("[RepCONC][stage2] no eval data: validation and early stopping disabled")

        loader = get_dataloader(items, batch_size, shuffle=True, seed=self.seed)
        n_steps = total_steps if total_steps is not None else epochs * max(1, math.ceil(len(items) / batch_size))
        logger.info(f"[RepCONC][stage2] {len(items)} pairs, batch={batch_size}, steps={n_steps}, "
                    f"dynamic_negs={dynamic_negs}")

        best_metric = self._train_loop(
            n_steps, loader, optimizer,
            lambda batch: self._stage2_step(batch, optimizer, temperature, neg_index,
                                            dynamic_negs, encode_batch_size),
            tag="stage2", eval_queries=eval_queries, eval_qrels=eval_qrels,
            valid_every=valid_every, patience=patience, metric=metric,
            checkpoint_dir=checkpoint_dir, best_metric=best_metric,
            valids_since_improve=valids_since_improve, best_step=best_step)
        del neg_index
        return best_metric

    def _recompute_codes(self) -> None:
        """(Re)compute the corpus codes as the unconstrained argmin
        assignments of the rotated corpus to the current centroids."""
        with torch.no_grad():
            self.pq.use_constraint = False
        try:
            x_rot = self._corpus_vecs @ self._rotation_np
            centroids = self.pq.centroids.detach().cpu().numpy()
            self._codes = _argmin_codes_np(x_rot, centroids)
        finally:
            self.pq.use_constraint = True

    def _build_dynamic_neg_index(self) -> faiss.Index:
        """An in-memory flat PQ index over the fixed stage-2 codes, used to
        retrieve dynamic negatives at every step (the official
        ``finetune_jpq`` convention). Its centroids are re-synced from the
        training model each step."""
        ensure_faiss_omp_compat()
        index = faiss.IndexPQ(self.d, self.M, self.nbits, faiss.METRIC_INNER_PRODUCT)
        index.is_trained = True
        self._sync_dynamic_neg_index(index)
        codes = np.ascontiguousarray(self._codes)
        packed = codes.reshape(-1) if self.nbits == 8 else pack_pq_codes(codes, self.nbits)
        faiss.copy_array_to_vector(packed, index.codes)
        index.ntotal = len(self._docnos)
        return index

    def _sync_dynamic_neg_index(self, index: faiss.Index) -> None:
        centroids = self.pq.centroids.detach().cpu().numpy()
        faiss.copy_array_to_vector(centroids.reshape(-1), index.pq.centroids)

    def _stage2_step(self, batch: list[dict], optimizer: torch.optim.Optimizer,
                     temperature: float, neg_index: faiss.Index,
                     dynamic_negs: int, encode_batch_size: int) -> float:
        B = len(batch)
        queries = [it["query"] for it in batch]
        pos_idx = np.array([self._docno2pos[it["docno"]] for it in batch], dtype=np.int64)

        q = self.model.encode_queries(queries, batch_size=encode_batch_size).to(self.device)   # [B, d] rotated
        pos_codes = torch.from_numpy(self._codes[pos_idx]).long().to(self.device)              # [B, M] fixed
        pos_dhat = self.pq.decode(pos_codes)                                                   # [B, d], grad via centroids

        # dynamic negatives from the current PQ index (centroids re-synced)
        with torch.no_grad():
            self._sync_dynamic_neg_index(neg_index)
            q_np = q.detach().cpu().numpy().astype(np.float32)
            _, dids = neg_index.search(q_np, dynamic_negs + 1)  # type: ignore
        neg_idx = np.empty((B, max(1, dynamic_negs)), dtype=np.int64)
        for b in range(B):
            cands = dids[b]
            cands = cands[cands != pos_idx[b]]
            neg_idx[b, :min(dynamic_negs, len(cands))] = cands[:dynamic_negs]
        neg_codes = torch.from_numpy(self._codes[neg_idx[:, :dynamic_negs]]).long().to(self.device)
        neg_dhat = self.pq.decode(neg_codes.reshape(-1, self.M)).reshape(B, dynamic_negs, self.pq.d)  # [B, K, d]

        sims_pos = (pos_dhat * q).sum(-1, keepdim=True) / temperature
        sims_neg = (neg_dhat * q.unsqueeze(1)).sum(-1) / temperature
        sims = torch.cat([sims_pos, sims_neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=self.device)
        loss = ranking_loss(sims, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.item()

    # ------------------------------------------------------------- training loop

    def _train_loop(
        self,
        n_steps: int,
        loader,
        optimizer: torch.optim.Optimizer,
        step_fn,
        *,
        tag: str,
        eval_queries: pd.DataFrame | None,
        eval_qrels: pd.DataFrame | None,
        valid_every: int,
        patience: int,
        metric: str,
        checkpoint_dir: str,
        best_metric: float,
        valids_since_improve: int,
        best_step: int,
    ) -> float:
        data_loader_iter = iter(loader)
        early_stop = False
        running_loss = 0.0
        current_time = time.time()

        if eval_queries is not None and eval_qrels is not None:
            val_stats = self._validation_step(eval_queries, eval_qrels)
            logger.info(f"[RepCONC][{tag}][val] at step {self._step} {str(val_stats)}")

        for _ in range(n_steps):
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(loader)
                batch = next(data_loader_iter)

            loss = step_fn(batch)
            running_loss += loss
            self._step += 1

            if self._step % 100 == 0:
                end_time = time.time()
                logger.info(f"[RepCONC][{tag}] step {self._step} ({end_time - current_time:.0f}s): "
                            f"running loss {running_loss / 100:.4f}")
                running_loss = 0.0
                current_time = end_time

            if valid_every and self._step % valid_every == 0:
                if eval_queries is not None and eval_qrels is not None:
                    val_stats = self._validation_step(eval_queries, eval_qrels)
                    logger.info(f"[RepCONC][{tag}][val] at step {self._step} {str(val_stats)}")
                    current = float(val_stats[metric])
                    if current > best_metric:
                        logger.info(f"[RepCONC][{tag}] new best {metric}: {best_metric:.6f} -> {current:.6f} "
                                    f"(step {best_step} -> {self._step})")
                        best_metric = current
                        valids_since_improve = 0
                        best_step = self._step
                        _save_checkpoint(os.path.join(checkpoint_dir, "best.pt"), model=self.model,
                                         optimizer=optimizer, step=self._step,
                                         best_metric=best_metric, trainer_self=self)
                    else:
                        valids_since_improve += 1
                        logger.info(f"[RepCONC][{tag}] no improvement in {valids_since_improve} validations "
                                    f"({patience - valids_since_improve} until early stop)")
                else:
                    logger.info(f"[RepCONC][{tag}] periodic checkpoint at step {self._step} (no eval data)")
                    _save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model=self.model,
                                     optimizer=optimizer, step=self._step,
                                     best_metric=best_metric, trainer_self=self)

            if valids_since_improve and valids_since_improve >= patience:
                logger.info(f"[RepCONC][{tag}] early stopping at step {self._step}: "
                            f"no improvement in {patience} validations (since step {best_step})")
                _save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model=self.model,
                                 optimizer=optimizer, step=self._step,
                                 best_metric=best_metric, trainer_self=self)
                best_path = os.path.join(checkpoint_dir, "best.pt")
                if os.path.isfile(best_path):
                    _load_checkpoint(best_path, model=self.model, optimizer=optimizer, trainer_self=self)
                    logger.info(f"[RepCONC][{tag}] reloaded best checkpoint from {best_path}")
                early_stop = True
                break

        if not early_stop:
            _save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model=self.model,
                             optimizer=optimizer, step=self._step,
                             best_metric=best_metric, trainer_self=self)
        self._best_metric = best_metric
        self._optimizer = optimizer
        return best_metric

    def _validation_step(self, eval_queries: pd.DataFrame, eval_qrels: pd.DataFrame,
                         topk: int = 1000):
        """Validate with an ADC retriever over the current centroids and the
        (stage-fixed) codes — the JPQ pattern of a temporary index that is
        cheap to rebuild (here: an in-memory IndexPQ, no disk)."""
        start_time = time.time()
        self.model.set_train(False)
        try:
            centroids = self.pq.centroids.detach().cpu().numpy()
            rotation = self.pq.rotation.detach().cpu().numpy()
            retr = (self.biencoder.query_encoder() >> RepCONCRetrieverPQ(
                self._docnos, self._codes, centroids, rotation=rotation,
                topk=topk, name="RepCONC-val"))
            rtr = pt.Evaluate(retr(eval_queries), eval_qrels,
                              metrics=[RR@10, Recall@1000, nDCG@10])
        finally:
            self.model.set_train(True)
        logger.info(f"[RepCONC] validation took {time.time() - start_time:.1f}s")
        return rtr

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        triples: Iterable[dict],
        stage2_pairs: Iterable[dict] | None = None,
        *,
        pq_sample_size: int = 10_000,
        stage1_kwargs: dict | None = None,
        stage2_kwargs: dict | None = None,
    ) -> None:
        """Warm up, train stage 1 on ``triples``, and (optionally) train
        stage 2 on ``stage2_pairs``.

        Args:
            triples: stage-1 text triples (see :meth:`train_stage1`).
            stage2_pairs: stage-2 (query, docno) pairs (see
                :meth:`train_stage2`). If None, stage 2 is skipped.
            pq_sample_size: corpus sample size for the OPQ+PQ warm-up.
            stage1_kwargs / stage2_kwargs: keyword arguments passed to
                :meth:`train_stage1` / :meth:`train_stage2`.
        """
        start_time = time.time()
        self._warmup(sample_size=pq_sample_size)
        self.train_stage1(triples, **(stage1_kwargs or {}))
        if stage2_pairs is not None:
            self.train_stage2(stage2_pairs, **(stage2_kwargs or {}))
        self.model.set_train(False)
        self.fitted = True
        logger.info(f"[RepCONC] fit complete in {time.time() - start_time:.0f}s")

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, path: str) -> None:
        """Save the current training state (model, PQ, optimizer, RNG) to
        ``path`` (a ``torch.save`` dict; see
        :mod:`pyterrier_dr.repconc.checkpointing`)."""
        if self.model is None:
            raise ValueError("nothing to checkpoint: the trainer has not been warmed up")
        _save_checkpoint(path, model=self.model, optimizer=self._optimizer,
                         step=self._step, best_metric=self._best_metric, trainer_self=self)

    def load_checkpoint(self, path: str) -> tuple[int, float]:
        """Load a checkpoint previously written by :meth:`save_checkpoint`.

        The optimizer state is restored into the *current* optimizer, so the
        checkpoint must come from the same stage. Returns ``(step,
        best_metric)``.
        """
        if self.model is None:
            raise ValueError("load a checkpoint only after warm-up (call _warmup/fit/train_stage*)")
        step, best_metric = _load_checkpoint(path, model=self.model,
                                             optimizer=self._optimizer, trainer_self=self)
        self._step, self._best_metric = step, best_metric
        return step, best_metric

    # ----------------------------------------------------------------- export

    def repconc_index(self, dest: str, nlist: int = 1, nprobe: int = 10,
                      corpus_text: Iterable[str] | Iterable[dict] | None = None) -> RepCONCIndex:
        """Build a searchable :class:`RepCONCIndex` from the trained model.

        Args:
            dest: destination directory (overwritten if it exists).
            nlist: number of IVF clusters (``1`` = flat PQ, the default —
                matching the official evaluation's degenerate IVF).
            nprobe: IVF probes per query (stored in the index metadata).
            corpus_text: document texts to re-encode with the *trained*
                document encoder before quantisation (the official
                evaluation's convention). Either:

                * an iterable of texts, in index docno order (every index
                  docno must be covered), or
                * an iterable of ``{"docno": ..., "text": ...}`` dicts —
                  the standard PyTerrier ``Indexer.index()`` input format —
                  where each text is matched to its index position by docno,
                  so the order does not matter.

                Index docnos without a text fall back to the
                :class:`FlexIndex` vectors (produced by the *pre-training*
                encoder). If None, the FlexIndex vectors are used for all
                documents — a documented deviation, since FlexIndex stores
                vectors rather than text.

        Returns:
            The built :class:`RepCONCIndex`.
        """
        if self.model is None or self.pq is None:
            raise ValueError("RepCONCTrainer not fitted (call fit/train_stage1 first)")
        docnos = self._docnos
        centroids = self.pq.centroids.detach().cpu().numpy()
        rotation = self.pq.rotation.detach().cpu().numpy()
        N = len(docnos)

        # resolve corpus_text into {index position: text}; positions without a
        # text fall back to the pre-training FlexIndex vectors
        texts_by_pos: dict[int, str] = {}
        if corpus_text is not None:
            if isinstance(corpus_text, dict):
                raise ValueError("corpus_text must be an iterable of texts or an iterable of "
                                 "'docno'/'text' dicts, not a single dict")
            n_strs = 0
            n_dicts = 0
            for rec in corpus_text:
                if isinstance(rec, dict):
                    if "docno" not in rec or "text" not in rec:
                        raise ValueError(f"corpus_text dict items must have 'docno' and 'text' keys: {rec!r}")
                    pos = self._docno2pos.get(str(rec["docno"]))
                    if pos is None:
                        logger.warning(f"[RepCONC] corpus docno {rec['docno']!r} not in the index; skipping")
                        continue
                    texts_by_pos[pos] = str(rec["text"])
                    n_dicts += 1
                elif isinstance(rec, str):
                    if n_strs >= N:
                        raise ValueError(f"corpus_text must have {N} texts, got {n_strs + 1}")
                    texts_by_pos[n_strs] = rec
                    n_strs += 1
                else:
                    raise ValueError(f"corpus_text items must be str or dicts with 'docno'/'text' "
                                     f"keys, got {type(rec).__name__}: {rec!r}")
            if n_strs and n_dicts:
                raise ValueError("corpus_text must be either all texts (in index docno order) "
                                 "or all {'docno', 'text'} dicts")
            if n_strs and n_strs != N:
                raise ValueError(f"corpus_text must have {N} texts, got {n_strs}")
            if not texts_by_pos:
                raise ValueError("corpus_text is empty: no texts to re-encode")
            if n_dicts and len(texts_by_pos) < N:
                logger.warning(f"[RepCONC] {N - len(texts_by_pos)} index docnos have no corpus text; "
                               "their vectors fall back to the pre-training FlexIndex vectors")

        if texts_by_pos:
            # start from the pre-training vectors, replacing the re-encoded
            # positions (in index order, chunked)
            x_rot = np.empty((N, self.d), dtype=np.float32)
            x_rot[:] = self._corpus_vecs
            positions = sorted(texts_by_pos)
            for i in range(0, len(positions), encode_chunk := 256):
                sel = positions[i:i + encode_chunk]
                with torch.no_grad():
                    chunk = self.biencoder.encode_docs_torch(
                        [texts_by_pos[p] for p in sel], batch_size=encode_chunk)
                x_rot[sel] = chunk.detach().cpu().numpy().astype(np.float32)
            x_rot = x_rot @ rotation
        else:
            x_rot = self._corpus_vecs @ rotation
        x_rot = np.ascontiguousarray(x_rot, dtype=np.float32)

        with torch.no_grad():
            self.pq.use_constraint = False
        try:
            # final index codes are the unconstrained (argmin) assignments,
            # as in the official evaluation (use_constraint=False)
            codes = _argmin_codes_np(x_rot, centroids)
        finally:
            self.pq.use_constraint = True

        logger.info(f"[RepCONC] building RepCONCIndex at {dest} (nlist={nlist}, nprobe={nprobe})")
        return RepCONCIndex.build(
            dest,
            docnos=docnos,
            codes=codes,
            centroids=centroids,
            rotation=rotation,
            x_rot=x_rot if nlist > 1 else None,
            nlist=nlist,
            nprobe=nprobe,
            mode=IndexingMode.overwrite,
        )
