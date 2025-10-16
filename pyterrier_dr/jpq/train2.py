import logging
import math
from typing import Any, Literal
from ir_measures import RR, Recall, nDCG
import numpy as np
import pandas as pd
from pyterrier import tqdm
import tempfile
import torch

import pyterrier as pt
from pyterrier_dr import FlexIndex
from pyterrier_dr.biencoder import BiEncoder
from pyterrier_dr.jpq.data import get_dataloader, get_pq_training_dataset
from pyterrier_dr.jpq.model import JPQBiencoder, JPQLoss, PassageEncoder, QueryEncoder
from pyterrier_dr.jpq.utils import timer
from pyterrier_dr.jpq.index import JPQIndex

from .pq import ProductQuantizer, ProductQuantizerFAISS, ProductQuantizerSKLearn


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def compute_PQ(
    M: int, # number of subquantizers (splits of the vector)
    n_bits: int, # number of centroids per subquantizer
    sample_size: int, # how many doc vectors from docids to use to train PQ centroids
    batch_size: int, # how many doc vectors from sample to process in a batch when computing PQ codes
    docids: np.ndarray, # which document indices (into full vecs_mem) to compute codes for
    vecs: np.ndarray, # vector store
    pq_impl: Literal["faiss", "sklearn"] = "faiss",
    ) -> tuple[np.ndarray, np.ndarray, ProductQuantizer]:

    if pq_impl == 'faiss':
        pq_class = ProductQuantizerFAISS
    elif pq_impl == 'sklearn':
        pq_class = ProductQuantizerSKLearn
    else:
        raise ValueError(f"Unknown pq_impl {pq_impl}, must be 'faiss' or 'sklearn'")

    pq = pq_class(M, Ks=2**n_bits)

    # train PQ on a random sample of the selected docs
    sample_size = min(sample_size, len(docids))
    print("[PQ] training on %d documents..." % sample_size)
    sample_docids = np.random.choice(docids, size=sample_size, replace=False) # type: ignore
    sample_docids = np.sort(sample_docids) # vector lookups from np.memmap are quicker when sorted
    with timer(f"PQ / train (samples={len(sample_docids):,})"):
        pq.fit(vecs[sample_docids])

    print("[PQ] computing codes for %d selected docs in chunks of %d..." % (len(docids), batch_size))
    codes = np.empty((len(docids), M), dtype=np.uint8) # not sure this is ok if we return sklearn codes
    docids = np.sort(docids) # vector lookups from np.memmap are quicker when sorted
    with timer("PQ / compute codes (selected)"):
        codes = pq.encode_batch(vecs, docids, batch_size)
    
    # TODO: we should check how the average/min/max codes are observed in sel_indices
    # give that sel_indices is a random sample of sel_indices, it should be fairly uniform
    # as a codes are centroids in the vector space defined in the small sample_size set, which
    # is a subset of the sel_indices set, it should be ok.
    return codes, pq.get_centroids(), pq # type: ignore


def autodevice(device) -> Any | Literal['mps'] | Literal['cuda'] | Literal['cpu']:
    return device or ("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


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

    print(f"[VAL] using {len(eval_queries)} queries with {len(eval_qrels)} qrels for validation")

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
        data_loader,
        epochs,
        lr,
        selected_docnos,
        codes,
        max_steps_per_epoch: int = math.inf, # type: ignore
        eval_queries: pd.DataFrame | None = None,
        eval_qrels: pd.DataFrame | None = None,
        valid_every: int = 10,
    ):
        loss_f = JPQLoss(model.query, model.passage).to(self.device)

        # Create optimizer for passage encoder AND query encoder
        optimizer = torch.optim.AdamW(list(model.passage.parameters()) + list(model.query.dr.model.parameters()), lr=lr, weight_decay=0.0)

        # set both models to train
        model.query.dr.model.train()
        model.passage.to(self.device).train()
        for ep in range(1, epochs + 1):
            step = 0
            running_loss = 0.0
            with timer(f"JPQ / epoch {ep}"):
                for batch in tqdm(data_loader, unit="batch", desc="JPQ epoch batches"):
                    loss = self._training_step(batch=batch, loss_f=loss_f, optimizer=optimizer)
                    running_loss += loss
                    step += 1

                    if step % 100 == 0:
                        logger.info(f"[JPQ] Training loss: {running_loss/step}")
                        running_loss = 0.0

                    if eval_queries is not None and step % valid_every == 0:
                        val_stats = self._validation_step(model, eval_queries, eval_qrels, selected_docnos, codes)
                        print(f"[JPQ][val] steps={step} {str(val_stats)}")

                    if step >= max_steps_per_epoch:
                        logger.info(f"[JPQ] reached max steps per epoch {max_steps_per_epoch}")
                        step = 0
                        break

            if step % valid_every == 0:
                logger.info(f"[JPQ] Training loss: {running_loss/step}")
                val_stats = self._validation_step(model, eval_queries, eval_qrels, selected_docnos, codes)
                print(f"[JPQ][val] steps={step} {str(val_stats)}")

            print(f"[JPQ] epoch {ep}/{epochs} steps {step}")


    def _validation_step(self, model, eval_queries, eval_qrels, selected_docnos, codes, topk_eval=100):
        with timer(f"JPQ / validation over {len(eval_queries)} queries"):
            with torch.no_grad():
                Q_t = model.query.encode_texts(eval_queries['query'].tolist(), batch_size=256)
            Q = Q_t.detach().cpu().numpy().astype('float32')

            dstindex = tempfile.mkdtemp()
            flex = FlexIndex(dstindex, verbose=False)

            device = next(model.passage.parameters()).device
            passage_encoder = model.passage.to("cpu").eval()
            def _gen():
                with torch.no_grad():
                    for i in range(0, len(codes), 16384): # magic number to replace recon_batch_size
                        chunk = torch.from_numpy(codes[i:i+16384]).long()
                        embs = passage_encoder(chunk).detach().cpu().numpy().astype('float32')
                        for j in range(embs.shape[0]):
                            yield {'docno' : selected_docnos[i+j], 'doc_vec' : embs[j, :]}
            flex.indexer(mode='overwrite').index(_gen())

            eval_queries = eval_queries.copy()
            eval_queries["query_vec"] = [row for row in Q]
            rtr = pt.Evaluate(
                flex.retriever()(eval_queries),  # type: ignore
                eval_qrels, 
                metrics=[RR@10, Recall@1000, nDCG@10])

            del(flex)

            # TODO: dest index may still be open
            # os.removedirs(dstindex)

            passage_encoder.to(device).train()
            return rtr


    def __init__(
        self,
        model : BiEncoder, # the backbone biencoder model
        index: FlexIndex, # the index with documents
        device = None,
        pq_impl: Literal['faiss'] | Literal['sklearn'] = 'sklearn', # the PQ implementation
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

    def fit(self,
        training_docpairs,
        pq_sample_size: int = 10_000, # how many doc vectors to use to train PQ centroids
        docid_subset: list[int] | list[str] | int | None = None, # how many doc vectors to use to train the sub-id embeddings 
        batch_size: int = 32,
        epochs: int = 3,
        lr:float = 2e-5,
        max_steps_per_epoch: int = math.inf, # type: ignore
        eval_queries : pd.DataFrame = None,
        eval_qrels : pd.DataFrame = None,
        valid_every : int = 25,
    ):
        selected_docnos, selected_docids, docno2pos = get_pq_training_dataset(self.index, docid_subset)
        codes, centroids, pq = compute_PQ(self.M, self.nbits, pq_sample_size, batch_size, selected_docids, self.index.payload()[1], pq_impl=self.pq_impl)
        self.pq = pq
        model = JPQBiencoder(
            QueryEncoder(self.query_encoder), 
            PassageEncoder(self.M, 2**self.nbits, self.d // self.M, centroids)
        ).to(self.device)

        data_loader = get_dataloader(training_docpairs, selected_docnos, codes, docno2pos, batch_size)
        # print(len(data_loader))
        eval_queries, eval_qrels = prepare_validation_data(eval_queries, eval_qrels, selected_docnos)

        self._training_loop(model, data_loader, epochs, lr, selected_docnos, codes, max_steps_per_epoch, eval_queries, eval_qrels, valid_every)
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
        centroids = torch.stack([ self.model.passage.sub_embeddings[i].weight for i in range(self.M) ]).detach().cpu().numpy() # M x Ks x dsub
        assert len(centroids.shape) == 3, centroids.shape
        
        return JPQIndex.build(
            dest, 
            docnos.fwd,
            all_codes,
            centroids
            )
