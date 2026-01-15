import logging
import math
import os
from typing import Callable, Literal
from ir_measures import RR, Recall, nDCG
import numpy as np
import pandas as pd
from pyterrier import tqdm
import tempfile
import torch
import time
from datasets import Dataset

import pyterrier as pt
import shutil
from pyterrier_dr import FlexIndex
from pyterrier_dr.biencoder import BiEncoder
from pyterrier_dr.flex.core import IndexingMode
from pyterrier_dr.jpq.checkpointing import _export_pq, _load_checkpoint, _save_checkpoint
from pyterrier_dr.jpq.data import (
    get_dataset, 
    get_dataloader, 
    get_pq_training_dataset, 
    prepare_validation_data, 
)
from pyterrier_dr.jpq.losses import JPQCELoss, JPQCELossWithJPQNegs, JPQCELossInBatchNegs, JPQCELossJPQNegsLambdaRank
from pyterrier_dr.jpq.model import JPQBiencoder, OPQQueryEncoder, PassageEncoder, QueryEncoder
from pyterrier_dr.jpq.utils import code_type_from_Ks, timer, autodevice
from pyterrier_dr.jpq.index import JPQIndex
from typing import Callable

from .pq import ProductQuantizer, ProductQuantizerFAISS, ProductQuantizerFAISSIndexPQ, ProductQuantizerSKLearn,ProductQuantizerFAISSIndexPQOPQ


#logging.basicConfig(level=logging.INFO, force=True)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S', force=True)

logger = logging.getLogger(__name__)



class JPQTrainer:

    def __init__(
        self,
        backbone_model : BiEncoder, # the backbone biencoder model
        index: FlexIndex, # the index with documents
        device = None,
        pq_impl: Literal['faiss','sklearn','faiss2', 'faiss2opq'] = 'sklearn', # the PQ implementation
        M: int = 8, # number of subquantizers (splits of the vector)
        nbits: int = 8, #  Bits per subquantiser code (e.g., 4, 5, 6, 7, or 8. Max is 16, which would need a very large training size)
        train_query_encoder = True
    ):
        super().__init__()
        self.fitted = False
        self.query_encoder = backbone_model
        self.index = index
        self.d = index.payload()[2]['vec_size']
        self.M = M
        if self.d % M != 0:
            raise ValueError(f"d={self.d} not divisible by M={M}")
        self.nbits = nbits
        if nbits %2 == 1:
            raise ValueError("nbits should be even")
        if nbits < 4 or nbits > 16:
            raise ValueError("nbits should be in range 4-16")
        
        self.Ks = 2** nbits
        self.pq_impl = pq_impl
        self.device = autodevice(device)
        self.train_query_encoder = train_query_encoder
        logger.info(f"[JPQTrainer init] device={self.device} train_query_encoder={train_query_encoder}")

    def _compute_PQ(
        self,
        sample_size: int, # how many doc vectors from docids to use to train PQ centroids
        docids: np.ndarray, # which document indices (into full vecs_mem) to compute codes for
        vecs: np.ndarray, # vector store
        batch_size: int = 10_000, # how many doc vectors from sample to process in a batch when computing PQ codes
        ) -> tuple[np.ndarray, np.ndarray, ProductQuantizer]:

        if self.pq_impl == 'faiss':
            pq_class = ProductQuantizerFAISS
        elif self.pq_impl == 'faiss2':
            pq_class = ProductQuantizerFAISSIndexPQ
        elif self.pq_impl == 'sklearn':
            pq_class = ProductQuantizerSKLearn
        elif self.pq_impl == 'faiss2opq':
            pq_class = ProductQuantizerFAISSIndexPQOPQ
        else:
            raise ValueError(f"Unknown pq_impl {self.pq_impl}, must be 'faiss' or 'sklearn'")

        pq = pq_class(self.M, Ks=self.Ks)

        # train PQ on a random sample of the selected docs
        sample_size = min(sample_size, len(docids))
        logger.info(f"[PQ] training M={self.M} Ks={self.Ks} on {sample_size} documents...")
        # set seed for reproducibility
        np.random.seed(42)
        rng = np.random.default_rng(seed=42)
        sample_docids = rng.choice(docids, size=sample_size, replace=False) # type: ignore
        # vector lookups from np.memmap are quicker when sorted
        # this sort is safe because we just want the vectors
        sample_docids = np.sort(sample_docids)
        logger.info(f"[PQ] First sampled docid: {sample_docids[0]}, last: {sample_docids[-1]}")
        with timer(f"PQ / loading sample vecs (samples={len(sample_docids):,})"):
            sample_vecs = vecs[sample_docids]
        with timer(f"PQ / train (samples={len(sample_docids):,})"):
            pq.fit(sample_vecs)
            sample_vecs = None  # free memory
            
        assert pq.centroids.shape == (self.M, self.Ks, self.d // self.M), \
            f"centroids shape {pq.centroids.shape}, expected {(self.M, self.Ks, self.d // self.M)}"

        # compute codes for all docids in the index, not just sample_size
        logger.info(f"[PQ] computing codes for {len(docids)} docs in chunks of {batch_size}...")
        with timer("PQ / compute codes (all docs)"):
            codes = pq.encode_batch(vecs, docids, batch_size, gpu=self.device if self.device != torch.device("cpu") else None)
        
        assert codes.shape[0] == len(docids), f"codes shape {codes.shape}, expected {len(docids)} rows"
        assert codes.shape[1] == self.M, f"codes shape {codes.shape}, expected {self.M} columns"
        return codes, pq.centroids, pq

    def _training_step(self, batch, loss_f, optimizer) -> float:
        """
        Run one optimisation step and return the loss.
        """
        optimizer.zero_grad()
        loss = loss_f(batch)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def _validation_prep(self, model, selected_docnos, codes, jpq_negs):
        retr, cleanup = self._currentindex(model, selected_docnos, codes)
        if jpq_negs > 0:
            from .data import add_jpq_negs_applier
            batch_decorator = add_jpq_negs_applier(retr, 100, codes) #why 100?
        else:
            batch_decorator = lambda x: x
        return retr, cleanup, batch_decorator

    def _training_loop(
        self,
        model : JPQBiencoder,
        dataset : Dataset,
        lr: float,
        selected_docnos,
        codes : np.ndarray,
        total_steps : int = 1_000_000_000,
        eval_queries: pd.DataFrame | None = None,
        eval_qrels: pd.DataFrame | None = None,
        valid_every: int = 10,
        in_batch : bool = False,
        batch_size : int = 32,
        jpq_negs : int = 0,
        checkpoint_dir: str = None,
        metric: str = "nDCG@10",
        mode: Literal["max", "min"] = "max",
        patience: int = 500,
        save_every_steps: int = 0,   # 0 = only save best/last; >0 also save step snapshots
        resume: bool = False,
        lambda_rank = False
    ):
        def _query_encoder_train():
            if self.train_query_encoder:
                model.query.train()

        # Loss function modes:
        # - lambda_rank: use lambda rank loss, *with* jpq_negs negative, with or without in-batch negatives
        # - in_batch: use in-batch negatives, can also have jpq_negs negatives, CE Loss
        # - jpq_negs only: uses explicit and jpq_negs negatives, CE loss
        # - default: CE loss only on the pairs
        if lambda_rank:
            assert jpq_negs, "lambdarank requires jpqnegs"
            loss_f = JPQCELossJPQNegsLambdaRank(model.query, model.passage, use_inbatch_negatives=in_batch, jpq_negs=jpq_negs)
        elif in_batch:
            loss_f = JPQCELossInBatchNegs(model.query, model.passage)
            # supports with or without jpq_negs
        else:
            if jpq_negs:
                loss_f = JPQCELossWithJPQNegs(model.query, model.passage)
            else:
                loss_f = JPQCELoss(model.query, model.passage)
        loss_f = loss_f.to(self.device)

        if eval_qrels is None or eval_queries is None:
            logger.warning("[JPQTrainer] No eval_queries or eval_qrels provided, validation will be skipped. No early stopping possible.")
            eval_qrels = eval_queries = None

        # Create optimizer for passage encoder AND query encoder
        if self.M in [16, 24]:
            lr = 5e-6
        elif self.M in [32]:
            lr = 2e-5
        elif self.M > 32:
            lr = 1e-4
        if self.train_query_encoder:
            optimizer = torch.optim.AdamW(
                [    # separate learning rates, as per original paper
                    {'params' : model.passage.parameters(), 'lr': lr},
                    {'params' : model.query.dr.model.parameters(), 'lr' : 5e-6}
                ], weight_decay=0.0)
        else:
            # query encoder is frozen
            optimizer = torch.optim.AdamW(
                [
                    {'params' : model.passage.parameters(), 'lr': lr},
                ], weight_decay=0.0)
            for p in model.query.parameters(recurse=True): p.requires_grad = False
        
        # set both models to train
        _query_encoder_train()
        model.passage.to(self.device).train()
        
        # Checkpointing
        assert checkpoint_dir is not None, "directory for checkpoints must be specified"
        best_metric = float("-inf") if mode == "max" else float("+inf")
        better = (lambda new, best: new > best) if mode == "max" else (lambda new, best: new < best)
        valids_since_improve = 0
        best_step = 0


        # Resuming
        if resume:
            last_path = os.path.join(checkpoint_dir, "last.pt")
            if os.path.isfile(last_path):
                st, bm = _load_checkpoint(last_path, model=model, optimizer=optimizer, trainer_self=self)
                best_metric = bm
                logger.info(f"[CKPT] Resumed from {last_path}: step={st}, best={best_metric:.6f}")
        
        data_loader = get_dataloader(dataset, batch_size)
        data_loader_iter = iter(data_loader)

        retr, cleanup, batch_decorator = self._validation_prep(model, selected_docnos, codes, jpq_negs)
        early_stop = False
        running_loss = 0.0
        current_time = time.time()

        if eval_queries is not None and eval_qrels is not None:
            val_stats = self._validation_step(retr, eval_queries, eval_qrels)
            logger.info(f"[JPQ][val] at step 0 {str(val_stats)}")

        for step in range(total_steps):
            # restart the iterator if we have reached the end
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)
            
            # add jpq_negs, if applicable; switch model to eval for retrieval
            model.query.eval()
            batch = batch_decorator(batch)
            _query_encoder_train()
            
            loss = self._training_step(batch=batch, loss_f=loss_f, optimizer=optimizer)
            running_loss += loss
            step += 1

            if step % 100 == 0:
                end_time = time.time()
                logger.info(f"[JPQ] Steps {step} Duration: {int(end_time-current_time)} sec. Training loss: {running_loss/step}")
                running_loss = 0.0
                current_time = end_time
            
            if step % valid_every == 0:
               # remove the previous index
                del(retr)
                cleanup()
                retr, cleanup, batch_decorator = self._validation_prep(model, selected_docnos, codes, jpq_negs)
                if eval_queries is not None and eval_qrels is not None:
                    val_stats = self._validation_step(retr, eval_queries, eval_qrels)
                    logger.info(f"[JPQ][val] at step {step} {str(val_stats)}")

                    # Checkpointing
                    current = float(val_stats[metric])
                    if better(current, best_metric):
                        logger.info(f"[JPQ] New best model at step {step}: {metric} improved from {best_metric:.6f} (at step {best_step}) to {current:.6f} ")
                        best_metric = current
                        valids_since_improve = 0
                        best_step = step
                        if checkpoint_dir:
                            best_path = os.path.join(checkpoint_dir, f"best_step{0:06d}.pt")
                            _save_checkpoint(best_path, model=model, optimizer=optimizer,  step=0, best_metric=best_metric, trainer_self=self)
                            _save_checkpoint(os.path.join(checkpoint_dir, "best.pt"), model=model, optimizer=optimizer, step=0, best_metric=best_metric, trainer_self=self)
                            ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location="cpu", weights_only=False)
                            _export_pq(os.path.join(checkpoint_dir, "pq_best"), ckpt)
                            ckpt = None
                    else:
                        valids_since_improve += 1
                        logger.info(f"[JPQ] No improvement in {valids_since_improve} validations. "
                                    f"Training will terminate in {patience - valids_since_improve} validations "
                                    f"(at step {step+(valid_every*(patience - valids_since_improve))}) if no further improvement.")
                
            # Periodic checkpointing
            if checkpoint_dir and save_every_steps and (step % save_every_steps == 0):
                last_path = os.path.join(checkpoint_dir, f"last_step{0:06d}.pt")
                _save_checkpoint(last_path, model=model, optimizer=optimizer, step=step, best_metric=best_metric, trainer_self=self)
                _save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model=model, optimizer=optimizer, step=step, best_metric=best_metric, trainer_self=self)
                ckpt = torch.load(os.path.join(checkpoint_dir, "last.pt"), map_location="cpu", weights_only=False)
                _export_pq(os.path.join(checkpoint_dir, "pq_last"), ckpt)
                ckpt = None

            if valids_since_improve and valids_since_improve >= patience:
                logger.info(f"[JPQ] Early stopping at step {step}: no improvement in {patience} validations (since step {best_step}).")
                if checkpoint_dir:
                    _save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model=model, optimizer=optimizer, step=step, best_metric=best_metric, trainer_self=self)
                    ckpt = torch.load(os.path.join(checkpoint_dir, "last.pt"), map_location="cpu", weights_only=False)
                    _export_pq(os.path.join(checkpoint_dir, "pq_last"), ckpt)
                    ckpt = None

                best_path = os.path.join(checkpoint_dir, "best.pt")
                if os.path.isfile(best_path):
                    _load_checkpoint(best_path, model=model, optimizer=optimizer, trainer_self=self)
                    logger.info(f"[JPQ] Loaded best model checkpointed so far from {best_path}")

                early_stop = True
                break
            
        # Save end-of-epoch "last"]
        if checkpoint_dir and not early_stop:
            _save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), model=model, optimizer=optimizer, step=step, best_metric=best_metric, trainer_self=self)
            ckpt = torch.load(os.path.join(checkpoint_dir, "last.pt"), map_location="cpu", weights_only=False)
            _export_pq(os.path.join(checkpoint_dir, "pq_last"), ckpt)
            ckpt = None

        del(retr)
        cleanup()

    def _currentindexJPQ(self, model : JPQBiencoder, selected_docnos, codes : np.ndarray, verbose=True) -> tuple[pt.Transformer, Callable]:
        # as the rmtree doesnt work, lets just try to use the same folder each time
        dstindex = "/tmp/valid_index" # tempfile.mkdtemp()

        passage_encoder = model.passage
        def _queryencoder(inp : pd.DataFrame):
            model.query.eval()
            with torch.no_grad():
                Q_t = model.query.forward(inp['query'].tolist(), batch_size=256)
            Q = Q_t.detach().cpu().numpy().astype('float32')
            rtr = inp.copy()
            rtr["query_vec"] = [row for row in Q]
            if self.train_query_encoder:
                model.query.train()
            return rtr
        
        def _cleanup():
            # dest index may still be open
            shutil.rmtree(dstindex, ignore_errors=True)    
            passage_encoder.train()

        # gather the trained sub-id representations
        centroids = torch.stack([ model.passage.sub_embeddings[m].weight for m in range(self.M) ]).detach().cpu().numpy() # type: ignore [M, Ks, dsub]
        # NB: model.query.forward already applies the OPQ rotation, so we dont specify it for this index
        index = self._jpq_index(dstindex, centroids, selected_docnos, codes, opq_R=None)
        return (pt.apply.generic(_queryencoder) >> index.retriever_pq(topk=1000)), _cleanup # type: ignore
        
    def _currentindexFLAT(self, model, selected_docnos, codes : np.ndarray, verbose=True) -> tuple[pt.Transformer, Callable]:
        # as the rmtree doesnt work, lets just try to use the same folder each time
        dstindex = "/tmp/valid_index" # tempfile.mkdtemp()
        flex = FlexIndex(dstindex, verbose=False)

        passage_encoder = model.passage
        def _gen():
            with torch.no_grad():
                iter = range(0, len(codes), 16384) # magic number to replace recon_batch_size
                iter = tqdm(iter, unit='batch', desc='Flat validation index construction')
                for i in iter:
                    chunk = torch.from_numpy(codes[i:i+16384]).long().to(self.device)
                    embs = passage_encoder(chunk).detach().cpu().numpy().astype('float32')
                    for j in range(embs.shape[0]):
                        yield {'docno' : selected_docnos[i+j], 'doc_vec' : embs[j, :]}
        flex.indexer(mode='overwrite').index(_gen())

        def _queryencoder(inp : pd.DataFrame):
            model.query.eval()
            with torch.no_grad():
                Q_t = model.query.forward(inp['query'].tolist(), batch_size=256)
            Q = Q_t.detach().cpu().numpy().astype('float32')
            rtr = inp.copy()
            rtr["query_vec"] = [row for row in Q]
            if self.train_query_encoder:
                model.query.train()
            return rtr
        
        def _cleanup():
            # dest index may still be open
            shutil.rmtree(dstindex, ignore_errors=True)    
            passage_encoder.train()

        return (pt.apply.generic(_queryencoder) >> flex.retriever(num_results=1000)), _cleanup # type: ignore

    _currentindex = _currentindexJPQ
    #_currentindex = _currentindexFLAT

    def _validation_step(
            self, 
            retr_pipe : pt.Transformer, 
            eval_queries : pd.DataFrame, 
            eval_qrels : pd.DataFrame, 
            topk_eval : int = 1000):
        
        start_time = time.time()
        rtr = pt.Evaluate(
            (retr_pipe % topk_eval)(eval_queries),  # type: ignore
            eval_qrels, 
            metrics=[RR@10, Recall@1000, nDCG@10])
        end_time = time.time()
        logger.info(f"Validation query execution took {end_time-start_time} sec")
        return rtr


    def fit(self,
        training_docpairs,
        pq_sample_size: int = 10_000, # how many doc vectors to use to train PQ centroids
        docid_subset: list[int] | list[str] | int | None = None, # how many doc vectors to use to train the sub-id embeddings 
        batch_size: int = 32,
        total_steps: int = 1_000_000_000,
        patience: int = 5,
        lr:float = 2e-5,
        eval_queries : pd.DataFrame | None = None,
        eval_qrels : pd.DataFrame | None = None,
        valid_every : int = 25,
        in_batch : bool = False,
        jpq_negs : int = 0,
        lambda_rank : bool = False,
        checkpoint_dir : str|None = None,
        pq_only: bool = False,
    ):
        start_time = time.time()
        selected_docnos, selected_docids, docno2pos, needs_filtered = get_pq_training_dataset(self.index, docid_subset)
        codes, centroids, pq = self._compute_PQ(pq_sample_size, selected_docids, self.index.payload()[1])
        self.pq = pq
        if hasattr(self.pq, "opq"):
            logger.info(f"[JPQTrainer] using OPQ rotation matrix")
            # TODO: consider if R should be trainable 
            R = torch.Tensor(self.pq.opq).to(self.device) # type: ignore
            query_encoder = OPQQueryEncoder(self.query_encoder, R)
        else:
            query_encoder = QueryEncoder(self.query_encoder)
        model = JPQBiencoder(
            query_encoder, 
            PassageEncoder(self.M, 2**self.nbits, self.d // self.M, centroids)
        ).to(self.device)

        if not pq_only:
            dataset = get_dataset(training_docpairs, selected_docnos, codes, docno2pos, filter_docnos=needs_filtered)
            eval_queries, eval_qrels = prepare_validation_data(eval_queries, eval_qrels, selected_docnos) # type: ignore
        
            # use a temporary directory for checkpoints
            if checkpoint_dir is None:
                import atexit
                checkpoint_dir = tempfile.mkdtemp("jpq_checkpoints")
                atexit.register(shutil.rmtree, checkpoint_dir)
            
            self._training_loop(
                model, dataset, lr, selected_docnos, codes, 
                total_steps=total_steps, 
                eval_queries=eval_queries, 
                eval_qrels=eval_qrels, 
                valid_every=valid_every, 
                in_batch=in_batch, 
                batch_size=batch_size, 
                jpq_negs=jpq_negs, 
                lambda_rank=lambda_rank,
                checkpoint_dir=checkpoint_dir, 
                metric="nDCG@10", 
                mode="max", 
                patience=patience, 
                save_every_steps=0, 
                resume=False)
            
            self.fitted = True

        self.model = model
        self.query_encoder.model.eval()

        if docid_subset is None:
            self.training_setup = "full_index"
            self.codes = codes
        else:
            self.training_setup = "docid_subset"
        end_time = time.time()
        logger.info(f"Total training time {(end_time-start_time)} seconds")

    def _jpq_index(self, 
                   dest : str, 
                   centroids : np.ndarray,
                   docnos : np.ndarray, 
                   codes : np.ndarray,
                   opq_R : np.ndarray = None) -> JPQIndex:
        """
        Builds an index using the provided centroid embeddings, for the provided docnos and codes 
        """
        
        return JPQIndex.build(
            dest,
            docnos,
            codes,
            centroids,
            mode=IndexingMode.overwrite,
            opq=opq_R
        )

    def jpq_index(self, dest : str) -> JPQIndex:
        """Return a JPQIndex using the final fitted JPQ model for all documents in the final index"""
        if not self.fitted:
            raise ValueError("JPQTrainer not fitted")
        
        centroids = torch.stack([ self.model.passage.sub_embeddings[m].weight for m in range(self.M) ]).detach().cpu().numpy() # type: ignore [M, Ks, dsub]
        
        # information from the original index
        docnos, original_embs, _ = self.index.payload(return_docnos=True, return_dvecs=True)
        if self.training_setup != "full_index":
            self.pq.centroids = centroids
            # compute codes for all docids of the original index
            all_codes = self.pq.encode_batch(original_embs, np.arange(len(self.index)), gpu=self.device if self.device != torch.device("cpu") else None)
        else:
            # we can reuse the codes computed during training
            all_codes = self.codes

        opq : None | np.ndarray = None
        if hasattr(self.pq, "opq"):
            opq = self.model.query.R.data.detach().cpu().numpy()
        
        logger.info("Generating final JPQIndex at " + dest)
        return self._jpq_index(dest, centroids, docnos.fwd, all_codes, opq_R=opq)
