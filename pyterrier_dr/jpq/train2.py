import logging
import math
from typing import Callable, Literal
from ir_measures import RR, Recall, nDCG
import numpy as np
import pandas as pd
from pyterrier import tqdm
import tempfile
import torch

import pyterrier as pt
import shutil
from pyterrier_dr import FlexIndex
from pyterrier_dr.biencoder import BiEncoder
from pyterrier_dr.flex.core import IndexingMode
from pyterrier_dr.jpq.data import get_dataloader, get_pq_training_dataset, get_dataset, add_jpq_negs
from pyterrier_dr.jpq.model import JPQBiencoder, JPQCELoss, JPQCELossInBatchNegs, PassageEncoder, QueryEncoder
from pyterrier_dr.jpq.utils import timer, autodevice
from pyterrier_dr.jpq.index import JPQIndex

from .pq import ProductQuantizer, ProductQuantizerFAISS, ProductQuantizerFAISSIndexPQ, ProductQuantizerSKLearn


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def compute_PQ(
    M: int, # number of subquantizers (splits of the vector)
    n_bits: int, # number of centroids per subquantizer
    sample_size: int, # how many doc vectors from docids to use to train PQ centroids
    batch_size: int, # how many doc vectors from sample to process in a batch when computing PQ codes
    docids: np.ndarray, # which document indices (into full vecs_mem) to compute codes for
    vecs: np.ndarray, # vector store
    pq_impl: Literal["faiss", "sklearn", "faiss2"] = "faiss",
    ) -> tuple[np.ndarray, np.ndarray, ProductQuantizer]:

    if pq_impl == 'faiss':
        pq_class = ProductQuantizerFAISS
    elif pq_impl == 'faiss2':
        pq_class = ProductQuantizerFAISSIndexPQ
    elif pq_impl == 'sklearn':
        pq_class = ProductQuantizerSKLearn
    else:
        raise ValueError(f"Unknown pq_impl {pq_impl}, must be 'faiss' or 'sklearn'")

    pq = pq_class(M, Ks=2**n_bits)

    # train PQ on a random sample of the selected docs
    sample_size = min(sample_size, len(docids))
    logger.info(f"[PQ] training M={M} Ks={2**n_bits} on {sample_size} documents...")
    # set seed for reproducibility
    np.random.seed(42)
    sample_docids = np.random.choice(docids, size=sample_size, replace=False) # type: ignore
    # vector lookups from np.memmap are quicker when sorted
    # this sort is safe because we just want the vectors
    sample_docids = np.sort(sample_docids)
    with timer(f"PQ / train (samples={len(sample_docids):,})"):
        pq.fit(vecs[sample_docids])

    logger.info(f"[PQ] computing codes for {len(docids):,} selected docs in chunks of {batch_size:,}...")
    codes = np.empty((len(docids), M), dtype=np.uint8) # not sure this is ok if we return sklearn codes
    with timer("PQ / compute codes (selected)"):
        codes = pq.encode_batch(vecs, docids, batch_size)
    
    # TODO: we should check how the average/min/max codes are observed in sel_indices
    # give that sel_indices is a random sample of sel_indices, it should be fairly uniform
    # as a codes are centroids in the vector space defined in the small sample_size set, which
    # is a subset of the sel_indices set, it should be ok.
    return codes, pq.get_centroids(), pq # type: ignore

def compute_from_pq_index(M, indexpq, docids):

    codes = np.empty((len(docids), M), dtype=np.uint8)

    return codes, pq.get_centroids(), pq # type: ignore




def prepare_validation_data(
    eval_queries: pd.DataFrame | None,
    eval_qrels: pd.DataFrame | None,
    selected_docnos: list[str],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Filter evaluation queries and qrels to include only queries
    that still have relevant documents among `selected_docnos`.
    Note that we remove queries that have no remaining relevant
    documents (label > 0).
    """
    if eval_queries is None:
        return None, None

    assert eval_qrels is not None, "eval_qrels must be provided when eval_queries is not None"

    # Keep qrels with documents used in training
    eval_qrels = eval_qrels[eval_qrels["docno"].isin(selected_docnos)]

    # Find queries that still have relevant docs
    valid_qids = set(eval_qrels.loc[eval_qrels["label"] > 0, "qid"])

    # Filter to only those queries/qrels
    eval_qrels = eval_qrels[eval_qrels["qid"].isin(valid_qids)]
    eval_queries = eval_queries[eval_queries["qid"].isin(valid_qids)]

    logger.info(f"[VAL] using {len(eval_queries)} queries with {len(eval_qrels)} qrels for validation")

    return eval_queries, eval_qrels


class JPQTrainer:

    def _training_step(self, batch, loss_f, optimizer) -> float:
        """
        Run one optimisation step and return the loss.
        """
        optimizer.zero_grad()
        loss = loss_f(batch)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _training_loop(
        self,
        model,
        dataset,
        epochs,
        lr,
        selected_docnos,
        codes,
        max_steps_per_epoch: int = math.inf, # type: ignore
        eval_queries: pd.DataFrame | None = None,
        eval_qrels: pd.DataFrame | None = None,
        valid_every: int = 10,
        in_batch : bool = False,
        batch_size : int = 32,
        jpq_negs : int = 0,
    ):
        lossclz = JPQCELossInBatchNegs if in_batch else JPQCELoss
        loss_f = lossclz(model.query, model.passage).to(self.device)

        # Create optimizer for passage encoder AND query encoder
        if self.M in [16, 24]:
            lr = 5e-6
        elif self.M in [32]:
            lr = 2e-5
        elif self.M > 32:
            lr = 1e-4

        # separate learning rates, as per paper
        optimizer = torch.optim.AdamW(
            [
                {'params' : model.passage.parameters(), 'lr': lr},
                {'params' : model.query.dr.model.parameters(), 'lr' : 5e-6}
            ], weight_decay=0.0)
        
        # set both models to train
        model.query.dr.model.train()
        model.passage.to(self.device).train()
        
        for ep in range(1, epochs + 1):
            step = 0
            running_loss = 0.0

            # add the jpq negs to the dataset; validate if so requested
            if jpq_negs > 0 or eval_queries is not None:
                retr, cleanup = self._currentindex(model, selected_docnos, codes)
                # always evaluate at the _start_ of the epoch (we may need for jpq_negs anyway)
                if eval_queries is not None:
                    val_stats = self._validation_step(retr, eval_queries, eval_qrels)
                    logger.info(f"[JPQ][val] before epoch {ep} {str(val_stats)}")
                if jpq_negs > 0:
                    dataset_jpq = add_jpq_negs(dataset, jpq_negs, retr, codes)
                    data_loader = get_dataloader(dataset_jpq, batch_size)
                # remove the index
                del(retr)
                cleanup()

            # no additional negs to add, apply normal dataset -> dataloader 
            if jpq_negs == 0:
                data_loader = get_dataloader(dataset, batch_size)

            #with timer(f"JPQ / epoch {ep}"):
            total = min(max_steps_per_epoch, len(data_loader)) if max_steps_per_epoch < math.inf else None

            for batch in tqdm(data_loader, unit="batch", total=total, desc="JPQ epoch batches"):
                loss = self._training_step(batch=batch, loss_f=loss_f, optimizer=optimizer)
                running_loss += loss
                step += 1

                if step % 100 == 0:
                    logger.info(f"[JPQ] Training loss: {running_loss/step}")
                    running_loss = 0.0

                if eval_queries is not None and step % valid_every == 0:
                    retr, cleanup = self._currentindex(model, selected_docnos, codes)
                    val_stats = self._validation_step(retr, eval_queries, eval_qrels)
                    cleanup()
                    logger.info(f"[JPQ][val] steps={step} {str(val_stats)}")

                if step >= max_steps_per_epoch:
                    logger.info(f"[JPQ] reached max steps per epoch {max_steps_per_epoch}")
                    step = 0
                    break

            logger.info(f"[JPQ] epoch {ep}/{epochs} steps {step}")
            logger.info(f"[JPQ] Training loss: {running_loss/step}")

    def _currentindex(self, model, selected_docnos, codes, verbose=True) -> tuple[pt.Transformer, Callable]:
        # as the rmtree doesnt work, lets just try to use the same folder each time
        dstindex = "/tmp/valid_index" # tempfile.mkdtemp()
        flex = FlexIndex(dstindex, verbose=False)

        device = next(model.passage.parameters()).device
        passage_encoder = model.passage.to("cpu").eval()
        def _gen():
            with torch.no_grad():
                iter = range(0, len(codes), 16384) # magic number to replace recon_batch_size
                iter = tqdm(iter, unit='batch', desc='Validation index construction')
                for i in iter:
                    chunk = torch.from_numpy(codes[i:i+16384]).long()
                    embs = passage_encoder(chunk).detach().cpu().numpy().astype('float32')
                    for j in range(embs.shape[0]):
                        yield {'docno' : selected_docnos[i+j], 'doc_vec' : embs[j, :]}
        flex.indexer(mode='overwrite').index(_gen())

        def _queryencoder(inp : pd.DataFrame):
            with torch.no_grad():
                Q_t = model.query.encode_texts(inp['query'].tolist(), batch_size=256)
            Q = Q_t.detach().cpu().numpy().astype('float32')
            rtr = inp.copy()
            rtr["query_vec"] = [row for row in Q]
            return rtr
        
        def _cleanup():
            # dest index may still be open
            shutil.rmtree(dstindex, ignore_errors=True)    
            passage_encoder.to(device).train()

        return (pt.apply.generic(_queryencoder) >> flex.retriever(num_results=1000)), _cleanup

    def _validation_step(
            self, 
            retr_pipe : pt.Transformer, 
            eval_queries : pd.DataFrame, 
            eval_qrels : pd.DataFrame, 
            topk_eval : int = 1000):
        
        rtr = pt.Evaluate(
            (retr_pipe % topk_eval)(eval_queries),  # type: ignore
            eval_qrels, 
            metrics=[RR@10, Recall@1000, nDCG@10])
        
        return rtr


    def __init__(
        self,
        model : BiEncoder, # the backbone biencoder model
        index: FlexIndex, # the index with documents
        device = None,
        pq_impl: Literal['faiss','sklearn','faiss2'] = 'sklearn', # the PQ implementation
        M: int = 8, # number of subquantizers (splits of the vector)
        nbits: int = 8, #  Bits per subquantiser code (e.g., 4, 5, 6, 7, or 8)
    ):
        super().__init__()
        self.fitted = False
        self.query_encoder = model
        self.index = index
        self.d = index.payload()[2]['vec_size']
        self.M = M
        self.nbits = nbits
        self.pq_impl = pq_impl
        self.device = autodevice(device)
        logger.info(f"[JPQTrainer] device={self.device}")

    def fit_from_indexpq(self, 
        training_docpairs, 
        index_pq, 
        batch_size: int = 32,
        epochs: int = 3,
        lr:float = 2e-5,
        max_steps_per_epoch: int = math.inf, # type: ignore
        eval_queries : pd.DataFrame | None= None,
        eval_qrels : pd.DataFrame | None = None,
        valid_every : int = 25,
        jpq_negs : int = 0
    ):
        selected_docnos, selected_docids, docno2pos = get_pq_training_dataset(self.index, None)
        codes, centroids, pq = compute_from_pq_index(self.M, index_pq, selected_docids)

        self.pq = pq
        model = JPQBiencoder(
            QueryEncoder(self.query_encoder), 
            PassageEncoder(self.M, 2**self.nbits, self.d // self.M, centroids)
        ).to(self.device)

        dataset = get_dataset(training_docpairs, selected_docnos, codes, docno2pos)
        eval_queries, eval_qrels = prepare_validation_data(eval_queries, eval_qrels, selected_docnos)

        self._training_loop(model, dataset, epochs, lr, selected_docnos, codes, max_steps_per_epoch, eval_queries, eval_qrels, valid_every)
        self.model = model
        self.fitted = True


    def fit(self,
        training_docpairs,
        pq_sample_size: int = 10_000, # how many doc vectors to use to train PQ centroids
        docid_subset: list[int] | list[str] | int | None = None, # how many doc vectors to use to train the sub-id embeddings 
        batch_size: int = 32,
        epochs: int = 3,
        lr:float = 2e-5,
        max_steps_per_epoch: int = math.inf, # type: ignore
        eval_queries : pd.DataFrame | None = None,
        eval_qrels : pd.DataFrame | None = None,
        valid_every : int = 25,
        in_batch : bool = False,
        jpq_negs : int = 0
    ):
        selected_docnos, selected_docids, docno2pos = get_pq_training_dataset(self.index, docid_subset)
        codes, centroids, pq = compute_PQ(self.M, self.nbits, pq_sample_size, 10_000, selected_docids, self.index.payload()[1], pq_impl=self.pq_impl) # type: ignore
        self.pq = pq
        model = JPQBiencoder(
            QueryEncoder(self.query_encoder), 
            PassageEncoder(self.M, 2**self.nbits, self.d // self.M, centroids)
        ).to(self.device)

        dataset = get_dataset(training_docpairs, selected_docnos, codes, docno2pos)
        eval_queries, eval_qrels = prepare_validation_data(eval_queries, eval_qrels, selected_docnos) # type: ignore

        self._training_loop(model, dataset, epochs, lr, selected_docnos, codes, max_steps_per_epoch, eval_queries, eval_qrels, valid_every, in_batch, batch_size=batch_size, jpq_negs=jpq_negs)
        self.model = model
        self.fitted = True

    def jpq_index(self, dest : str) -> JPQIndex:
        if not self.fitted:
            raise ValueError("JPQTrainer not fitted")
        
        # information from the original index
        docnos, original_embs, _ = self.index.payload(return_docnos=True, return_dvecs=True)

        # compute codes for _all_ of the original index
        all_codes = self.pq.encode_batch(original_embs, list(range(len(self.index))))
        assert len(all_codes.shape) == 2, all_codes.shape
        assert all_codes.shape[0] == len(self.index)
        
        # gather the trained sub-id representations
        centroids = torch.stack([ self.model.passage.sub_embeddings[i].weight for i in range(self.M) ]).detach().cpu().numpy() # type: ignore # M x Ks x dsub
        assert len(centroids.shape) == 3, centroids.shape
        
        return JPQIndex.build(
            dest, 
            docnos.fwd,
            all_codes,
            centroids,
            mode=IndexingMode.overwrite
        )
