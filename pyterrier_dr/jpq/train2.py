import logging
import math
from typing import Any, Literal
import numpy as np
from pyterrier import tqdm
import torch

from pyterrier_dr import FlexIndex
from pyterrier_dr.biencoder import BiEncoder
from pyterrier_dr.jpq.data import get_dataloader, get_pq_training_dataset
from pyterrier_dr.jpq.model import JPQBiencoder, JPQLoss, PassageEncoder, QueryEncoder
from pyterrier_dr.jpq.utils import l2_normalize_np, timer

from .pq import ProductQuantizerFAISS, ProductQuantizerFAISSIndexPQ, ProductQuantizerSKLearn


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
    ) -> tuple[np.ndarray, np.ndarray]:

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
    print("[PQ] training on %d documents..." % sample_size)
    sample_docids = np.random.choice(docids, size=sample_size, replace=False) # type: ignore
    with timer(f"PQ / train (samples={len(sample_docids):,})"):
        xb = l2_normalize_np(vecs[sample_docids])
        pq.fit(xb)

    print("[PQ] computing codes for %d selected docs in chunks of %d..." % (len(docids), batch_size))
    codes = np.empty((len(docids), M), dtype=np.uint8) # not sure this is ok if we return sklearn codes
    with timer("PQ / compute codes (selected)"):
        codes = pq.encode_batch(l2_normalize_np(vecs[docids]), batch_size)
    
    # TODO: we should check how the average/min/max codes are observed in sel_indices
    # give that sel_indices is a random sample of sel_indices, it should be fairly uniform
    # as a codes are centroids in the vector space defined in the small sample_size set, which
    # is a subset of the sel_indices set, it should be ok.
    return codes, pq.get_centroids() # type: ignore


def autodevice(device) -> Any | Literal['mps'] | Literal['cuda'] | Literal['cpu']:
    return device or ("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

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
        max_steps_per_epoch: int = math.inf, # type: ignore
    ):
        loss_f = JPQLoss(model.query, model.passage).to(self.device)

        # Freeze query encoder
        model.query.eval()
        for p in model.query.parameters():
            p.requires_grad = False

        # Create optimizer for passage encoder
        optimizer = torch.optim.AdamW(list(model.passage.parameters()), lr=lr, weight_decay=0.0)

        model.passage.to(self.device).train()
        for ep in range(1, epochs + 1):
            step = 0
            running_loss = 0.0
            with timer(f"JPQ / epoch {ep}"):
                for batch in tqdm(data_loader, unit="batch", desc="JPQ epoch batches"):
                    loss = self._training_step(batch=batch, loss_f=loss_f, optimizer=optimizer)
                    running_loss += loss
                    step += 1

                    if step == 100:
                        logger.info(f"[JPQ] Training loss: {running_loss/step}")
                        running_loss = 0.0
                        step = 0

                    # Validation: TODO

                    if step >= max_steps_per_epoch:
                        logger.info(f"[JPQ] reached max steps per epoch {max_steps_per_epoch}")
                        step = 0
                        break


            if step:
                logger.info(f"[JPQ] Training loss: {running_loss/step}")
            print(f"[JPQ] epoch {ep}/{epochs} steps {step}")

        
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

        # self.query_encoder = QueryEncoder(existing_model, normalise=True, batch_size=64)


    def fit(self,
        training_docpairs,
        pq_sample_size: int = 10_000, # how many doc vectors to use to train PQ centroids
        docid_subset: list[int] | list[str] | int | None = None, # how many doc vectors to use to train the sub-id embeddings 
        batch_size: int = 32,
        epochs: int = 3,
        lr:float = 2e-5,
        max_steps_per_epoch: int = math.inf, # type: ignore
    ):
        selected_docnos, selected_docids, docno2pos = get_pq_training_dataset(self.index, docid_subset)
        codes, centroids = compute_PQ(self.M, self.nbits, pq_sample_size, batch_size, selected_docids, self.index.payload()[1])
        model = JPQBiencoder(
            QueryEncoder(self.query_encoder), 
            PassageEncoder(self.M, 2**self.nbits, self.d // self.M, centroids)
        )

        data_loader = get_dataloader(training_docpairs, selected_docnos, codes, docno2pos, batch_size)
        
        self._training_loop(model, data_loader, epochs, lr, max_steps_per_epoch)
        self.fitted = True