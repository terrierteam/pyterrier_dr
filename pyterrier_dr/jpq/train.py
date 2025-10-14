import pyterrier_dr
from typing import Any, Dict, List, Optional, Tuple, Literal, Union, Iterator
import pandas as pd
import pyterrier as pt
import numpy as np
from .utils import timer, l2_normalize_np, NullWanDBRun
import torch
from pyterrier import tqdm
import os
from .index import JPQIndex
import math
from datasets import IterableDataset, DatasetInfo, Dataset
from pyterrier.measures import *
import json
from .pq import ProductQuantizer
from .model import *
from torch.utils.data import DataLoader

def get_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def flex_payload(flex_index: pyterrier_dr.FlexIndex):
    docnos_map, vecs, meta = flex_index.payload()
    N = int(meta.get('doc_count', len(vecs)))
    d = int(meta['vec_size'])
    docnos = [docnos_map[i] for i in range(N)]
    return docnos, vecs, N, d


def inv_map(flex_index: pyterrier_dr.FlexIndex):
    return flex_index.payload()[0].inv  # docno -> position

# ---------- helpers for writing full index ----------
def _unpack_pq_codes_batch(packed: np.ndarray, M: int, nbits: int) -> np.ndarray:
    """FAISS packs codes when nbits != 8. Convert (B, code_size) -> (B, M) uint8."""
    if packed.shape[1] == M:   # nbits == 8 fast path
        return packed.astype(np.uint8, copy=False)
    B = packed.shape[0]
    out = np.empty((B, M), dtype=np.uint8)
    for i in range(B):
        bits = np.unpackbits(packed[i], bitorder='little')
        need = M * nbits
        if bits.size < need:
            bits = np.concatenate([bits, np.zeros(need - bits.size, dtype=np.uint8)], 0)
        vals = np.zeros(M, dtype=np.uint8)
        for m in range(M):
            v = 0; base = m * nbits
            for b in range(nbits): v |= int(bits[base+b]) << b
            vals[m] = v
        out[i] = vals
    return out

def _autodevice(device):
    return device or ("mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

class JPQTrainer:

    def __init__(
            self,
            existing_model : pyterrier_dr.BiEncoder,
            existing_index: pyterrier_dr.FlexIndex,
            device = None,
            pq_impl: Literal['faiss'] | Literal['sklearn'] = 'sklearn', 
            pq_M: int = 8, 
            pq_nbits: int = 8,
            wandb : Union['wandb.Run', NullWanDBRun] = NullWanDBRun()):
        super().__init__()
        self.fitted = False
        self.existing_index = existing_index
        self.d = existing_index.payload()[2]['vec_size']
        self.pq_M = pq_M
        self.pq_nbits = pq_nbits
        self.pq_impl = pq_impl
        self.wandb = wandb or NullWanDBRun()
        self.device = _autodevice(device)
        print("[JPQTrainer] device=", self.device)

        self.query_encoder = QueryEncoder(existing_model, normalise=True, batch_size=64)

    def fit(
            self,
            training_docpairs,
            code_batch_size: int = 200_000, 
            recon_batch_size: int = 20_000,
            pq_sample_size: int = 10_000, # how many doc vectors to use to train PQ centroids
            docid_subset: list[int] | list[str] | int | None = None, # how many doc vectors to use to train the sub-id embeddings 
            epochs: int = 3, 
            batch_size: int = 32, 
            patience : int = 2000,
            extra_neg_pool : int = 0,
            eval_queries : pd.DataFrame = None,
            eval_qrels : pd.DataFrame = None,
            valid_every : int = 25,
            lr: float = 2e-5):
        
        rng = np.random.RandomState(42)
        full_docnos, vecs_mem, N, d = flex_payload(self.existing_index)

        id2idx = inv_map(self.existing_index)
        
        # if extra_neg_pool > 0:
        #     need = extra_neg_pool
        #     while need > 0:
        #         did = full_docnos[int(rng.randint(0, N))]
        #         if did not in selected_doc_ids:
        #             selected_doc_ids.add(did); need -= 1

        print(f"Ingesting docno mapping from index ")
        
        if docid_subset is None:
            selected_doc_ids = self.existing_index.payload()[0].fwd[list(range(len(self.existing_index)))]
            print(f"[SUBSET] using ALL {len(self.existing_index)} docs from index")
        else:
            N = len(self.existing_index)
            if isinstance(docid_subset, int): # select random subset of this size
                if docid_subset > N:
                    raise ValueError(f"docid_subset {docid_subset} > total docs {N}")
                docid_subset = rng.randint(0, N, docid_subset).tolist()
                selected_doc_ids = self.existing_index.payload()[0].fwd[docid_subset]
            elif isinstance(docid_subset, list):  
                if isinstance(docid_subset[0], int): # use the provided list of int docid
                    selected_doc_ids = self.existing_index.payload()[0].rev(docid_subset)  # int -> str
                else:   # use the provided list of str docnos
                    assert isinstance(docid_subset[0], str)
                    selected_doc_ids = self.existing_index.payload()[0].fwd[docid_subset]
            print(f"[SUBSET] using {len(docid_subset)} docs from docid_subset")
            
        selected_doc_ids : List[str] = list(selected_doc_ids)
        # I dont think we need to shuffle, as its likely a random sample. 
        # would be better sorted for faster lookups during PQ training?!
        # rng.shuffle(selected_doc_ids)
        # selected_doc_ids.sort()
        
        sel_indices = np.array([int(id2idx[did]) for did in selected_doc_ids], dtype=np.int64)
        sel_inv = {did: i for i, did in enumerate(selected_doc_ids)}
        
        if eval_queries is not None:
            assert eval_qrels is not None
            # reduce qrels to those with docs in selected_doc_ids
            valid_qrels = eval_qrels[eval_qrels.docno.isin(selected_doc_ids)]

            # identify qid that still have relevant documents
            eval_qids = set(valid_qrels[valid_qrels['label'] > 0]['qid'].tolist())

            # cut queries and qrels to only those that have relevant documents
            valid_qrels = valid_qrels[valid_qrels['qid'].isin(eval_qids)]
            valid_queries = eval_queries[eval_queries['qid'].isin(eval_qids)]
            print(f"[VAL] using {len(eval_qids)} queries with {len(valid_qrels)} qrels for validation")
        else:
            valid_queries = None
            valid_qrels = None

        # ------- PQ -------
        pq, codes_sel, centroids = self._compute_PQ(pq_sample_size, code_batch_size, sel_indices, vecs_mem, rng)

        # ------- initialise the model -------
        # using the centroids from PQ as the starting point for the sub-id embeddings
        self.model = JPQBiencoder(self.query_encoder, PassageEncoder(self.pq_M, 2**self.pq_nbits, self.d // self.pq_M, centroids))

        # --------- initial evaluation on full dataset --------
        if valid_queries is not None:
            from .retriever import build_from_flex
            from tempfile import mkdtemp
            new_flex = build_from_flex(self.existing_index, pq, self.model, mkdtemp())
            from pyterrier.measures import RR, Recall, nDCG
            query_encoder = pt.apply.query_vec(lambda qrow: self.model.query.encode_texts([qrow["query"]])[0])
            jpq_pipe = query_encoder >> new_flex.retriever()
            eval_0 = pt.Experiment(
                [query_encoder >> self.existing_index.retriever(), jpq_pipe],
                eval_queries, 
                eval_qrels, 
                eval_metrics=[RR@10, Recall@50, nDCG@10],
                names=["Baseline flat", "JPQ initial"])
            print(f"Initial JPQ eval\n {eval_0}")

        # ------- dataloader -------
        dl = self._dataloader(training_docpairs, batch_size, selected_doc_ids, sel_inv, codes_sel)
        
        # ------- training the sub-id embeddings -------
        self._training_loop(self.model, centroids, dl, epochs, lr, patience, 
                            selected_doc_ids, codes_sel,
                            valid_queries, valid_qrels, recon_batch_size, 
                            valid_every=valid_every)
        self.fitted = True
        self.pq = pq

        if valid_queries is not None:
            from .retriever import build_from_flex
            from tempfile import mkdtemp
            new_flex = build_from_flex(self.existing_index, pq, self.model, mkdtemp())
            from pyterrier.measures import RR, Recall, nDCG
            retr_pipe = pt.apply.query_vec(lambda qrow: self.model.query.encode_texts([qrow["query"]])[0]) >> new_flex.retriever()
            eval_final = pt.Evaluate(
                retr_pipe(eval_queries), 
                eval_qrels, 
                metrics=[RR@10, Recall@50, nDCG@10])
            print(f"Final JPQ eval {eval_final}")

    
    def _dataloader(self, 
                    training_docpairs, 
                    batch_size : int, # batch size for training
                    selected_doc_ids : list[str], # documents that we're using during training
                    sel_inv : dict[str,int], # map from str docid -> index into codes_sel 
                    codes_sel : np.array) -> DataLoader: # PQ codes for selected documents
       
        # make it a set of easier dataset filtering below
        selected_doc_ids = set(selected_doc_ids)
        # bring codes to torch
        codes_sel = torch.from_numpy(codes_sel).long()

        # making a list smells bad
        dataset = Dataset.from_list([x for x in training_docpairs])

        def filter_in_sel(row):
            return row['doc_id_a'] in selected_doc_ids and row['doc_id_b'] in selected_doc_ids

        def queries_and_codes(x):
            return {
                    'query_text': x["query"],
                    'pos_codes': codes_sel[sel_inv[x["doc_id_a"]]],
                    'neg_codes': codes_sel[sel_inv[x["doc_id_b"]]],
                }
        def collate(batch):
            return {
                'query_text': [b['query_text'] for b in batch],
                'pos_codes': torch.stack([b['pos_codes'] for b in batch]),
                'neg_codes': torch.stack([b['neg_codes'] for b in batch]),
            }
        dataset = dataset.filter(filter_in_sel).map(queries_and_codes)
        dataset.set_format(type='torch', columns=['pos_codes', 'neg_codes', 'query_text'])

        dl = DataLoader(
            # remove train queries for documents that arent in our selection
            dataset, 
            batch_size=batch_size, 
            collate_fn=collate)
        return dl
    
    def _compute_PQ(self, 
                    pq_sample_size: int, # how many doc vectors to use to train PQ centroids
                    code_batch_size: int, # how many doc vectors to process in a batch when computing PQ codes
                    sel_indices : np.array, # which document indices (into full vecs_mem) to compute codes for
                    vecs_mem : np.array, # vector store
                    rng # random state
                    ) -> Tuple[ProductQuantizer, np.ndarray, np.ndarray]:
        from .pq import ProductQuantizerFAISS, ProductQuantizerSKLearn
        pq_class : ProductQuantizer = None
        if self.pq_impl == 'faiss':
            pq_class = ProductQuantizerFAISS
        elif self.pq_impl == 'sklearn':
            pq_class = ProductQuantizerSKLearn
        else:
            raise ValueError(f"Unknown pq_impl {self.pq_impl}, must be 'faiss' or 'sklearn'")
        pq = pq_class(M=self.pq_M, Ks=2**self.pq_nbits)

        # train PQ on a random sample of the selected docs
        sample_size = min(pq_sample_size, len(sel_indices))
        print("[PQ] training on %d documents..." % sample_size)
        sample_idx = rng.choice(sel_indices, size=sample_size, replace=False)
        sample_idx = np.sort(sample_idx) # vector lookups from np.memmap are quicker when sorted
        with timer(f"PQ / train (samples={len(sample_idx):,})"):
            xb = l2_normalize_np(vecs_mem[sample_idx])
            pq.fit(xb)

        print("[PQ] computing codes for %d selected docs in chunks of %d..." % (len(sel_indices), code_batch_size))
        sel_indices = np.sort(sel_indices) # vector lookups from np.memmap are quicker when sorted
        codes_sel = np.empty((len(sel_indices), self.pq_M), dtype=np.uint8)
        with timer("PQ / compute codes (selected)"):
            codes_sel = pq.encode_batch(l2_normalize_np(vecs_mem[sel_indices]), batch_size=code_batch_size)
        
        # TODO: we should check how the average/min/max codes are observed in sel_indices
        # give that sel_indices is a random sample of sel_indices, it should be fairly uniform
        # as a codes are centroids in the vector space defined in the small sample_size set, which
        # is a subset of the sel_indices set, it should be ok.
        return pq, codes_sel, pq.get_centroids()
    
    def _training_loop(self, model, centroids, dl : DataLoader, epochs, lr, patience, 
                       selected_doc_ids : List[str], codes_sel : np.array,
                       eval_queries : pd.DataFrame, eval_qrels : pd.DataFrame, 
                       recon_batch_size : int,
                        max_steps=None, valid_every=25):

        loss_f = JPQLoss(model.query, model.passage).to(self.device)
        model.query.eval()
        for p in model.query.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(list(model.passage.parameters()), lr=lr, weight_decay=0.0)

        import tempfile
        out_dir = tempfile.mkdtemp()
        model_dir = os.path.join(out_dir, "model"); 
        os.makedirs(model_dir, exist_ok=True)

        with torch.no_grad():
            init_w = torch.cat([emb.weight.detach().flatten() for emb in self.model.passage.sub_embeddings]).to(self.device)
        
        lambda_reg = 1e-4
        best_state = None
        step = 0
        best_mrr = 0.0
        bad = 0
        running_loss = 0.0


        for ep in range(1, epochs + 1):
            if step >= (max_steps or math.inf):
                print("[JPQ] reached max_steps"); 
                break
            
            model.passage.to(self.device).train()
            with timer(f"JPQ / epoch {ep}"):
                for batch in tqdm(dl, unit="batch", desc=f"JPQ epoch batches"):
                    optimizer.zero_grad()
                    base = loss_f(batch)
                    cur_w = torch.cat([emb.weight.flatten() for emb in model.passage.sub_embeddings])
                    reg = lambda_reg * torch.mean((cur_w - init_w.to(cur_w.device))**2)
                    loss = base + reg
                    loss.backward()
                    running_loss += float(loss.item())
                    
                    if step % 100 == 0:
                        self.wandb.log({"train/base_loss": running_loss/100, "train/grad_norm": get_grad_norm(model.passage)}, step=step)
                        running_loss = 0.0

                    optimizer.step()
                    

                    step += 1

                    # validation
                    if eval_queries is not None and step % valid_every == 0:
                        val_stats = self._run_validation(model, eval_queries, eval_qrels, selected_doc_ids, codes_sel, recon_batch_size)
                        model.passage.to(self.device).train()
                        self.wandb.log({f"val/{k}": v for k, v in val_stats.items()}, step=step)
                        print(f"[JPQ][val] steps={step} {str(val_stats)}")

                        if val_stats['RR@10'] > best_mrr + 1e-5:
                            best_mrr = val_stats['RR@10']; bad = 0
                            best_state = {k: v.detach().cpu().clone() for k, v in model.passage.state_dict().items()}
                            torch.save(model.passage.state_dict(), os.path.join(model_dir, "jpq_passage.pt"))
                            with open(os.path.join(model_dir, "pq_meta.json"), "w") as f:
                                json.dump({"M": self.pq_M, "nbits": self.pq_nbits, "d": self.d}, f)
                            cents0 = centroids
                            np.save(os.path.join(model_dir, "pq_init_centroids.npy"), cents0)
                        else:
                            bad += 1
                            if bad >= patience:
                                print(f"[JPQ] Early stopping after {step} steps, patience {patience}."); break
                        
                    if step >= (max_steps or math.inf):
                        break
            assert step > 0, "We didnt run any training steps - perhaps data iterator was empty?"
            print(f"[JPQ] epoch {ep}/{epochs} steps {step}")

            # model.passage.eval()
            

        if best_state is not None:
            model.passage.load_state_dict(best_state)

    def _run_validation(self, model, val_queries : pd.DataFrame, cut_qrels : pd.DataFrame, selected_doc_ids : List[str], codes_sel, recon_batch_size : int, topk_eval=100):
        with timer(f"JPQ / validation over {len(val_queries)} queries"):
            with torch.no_grad():
                Q_t = model.query.encode_texts(val_queries['query'].tolist(), batch_size=256)
            Q = Q_t.detach().cpu().numpy().astype('float32')
            Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            dim = model.passage.sub_embeddings[0].embedding_dim * model.passage.M
            from pyterrier_dr import FlexIndex
            import tempfile
            dstindex = tempfile.mkdtemp()
            flex = FlexIndex(dstindex, verbose=False)
            pm = model.passage.to("cpu").eval()
            def _gen():
                with torch.no_grad():
                    for i in range(0, len(codes_sel), recon_batch_size):
                        chunk = torch.from_numpy(codes_sel[i:i+recon_batch_size]).long()
                        embs = pm(chunk)
                        embs = (embs / (embs.norm(dim=1, keepdim=True) + 1e-12)).detach().cpu().numpy().astype('float32')
                        for j in range(embs.shape[0]):
                            yield {'docno' : selected_doc_ids[i+j], 'doc_vec' : embs[j, :]}
            flex.indexer(mode='overwrite').index(_gen())
            val_queries = val_queries.copy()
            val_queries["query_vec"] = [row for row in Q]
            rtr = pt.Evaluate(
                flex.retriever()(val_queries), 
                cut_qrels, 
                metrics=[RR@10, Recall@50, nDCG@10])
            del(flex)
            # TODO: dest index may still be open
            #os.removedirs(dstindex)
            return rtr

    def faiss_index(self, index_out):
        if not self.fitted:
            raise ValueError("JPQTrainer not fitted")
        # return faiss index over FULL set (reconstructed from learned PQ)
        out_file = index_out or os.path.join(out_dir, "full.indexpq")
        with timer("Write FULL IndexPQ (ALL docs)"):
            saved_index_path = write_full_indexpq_from_learned(
                flex_index=self.existing_index,
                passage_model=model.passage,
                d=d, M=pq_M, nbits=pq_nbits,
                out_path=out_file,
                metric=indexpq_metric,
                doc_encoder_for_full=doc_encoder_for_full,   # TCT=None / ANCE=ance_model
                batch_docs=100_000, sub_batch_encode=64,
            )

    def jpq_index(self, dest : str) -> JPQIndex:
        if not self.fitted:
            raise ValueError("JPQTrainer not fitted")
        
        # information from the original index
        docnos, original_embs, _ = self.existing_index.payload(return_docnos=True, return_dvecs=True)

        # compute codes for _all_ of the original index
        all_codes = self.pq.encode_batch(original_embs)
        assert len(all_codes.shape) == 2, all_codes.shape
        assert all_codes.shape[0] == len(self.existing_index)
        
        # gather the trained sub-id representations
        centroids = torch.cat([ self.model.passage.sub_embeddings[i].weight for i in range(self.pq_M) ]).detach().cpu().numpy() # M x Ks x dsub
        assert len(centroids.shape) == 3, centroids.shape
        
        return JPQIndex.build(
            dest, 
            docnos.fwd,
            all_codes,
            centroids
            )


    # # ------- optional -------
    # saved_index_path = None
    # if save_index.lower() == "pq":
        
    # elif save_index.lower() == "flex":
    #     # out_dir_idx = index_out or os.path.join(out_dir, "full.flex")
    #     # with timer("Write FULL FlexIndex (ALL docs, reconstructed)"):
    #     #     saved_index_path = write_full_flexindex_from_learned(
    #     #         flex_index=existing_index,
    #     #         passage_model=model.passage,
    #     #         d=d, M=pq_M, nbits=pq_nbits,
    #     #         out_dir=out_dir_idx,
    #     #         doc_encoder_for_full=doc_encoder_for_full,
    #     #         batch_docs=100_000, sub_batch_encode=64, recon_batch=20_000,
    #     #     )

    # # ------- JPQRetriever -------
    # retriever = JPQRetriever(
    #     biencoder=model,
    #     docnos=selected_doc_ids,
    #     codes=codes_sel,
    #     topk=1000,
    #     name=f"JPQ(M={pq_M},nbits={pq_nbits})"
    # )
    # return model, model_dir, saved_index_path, retriever, selected_doc_ids, {"M": pq_M, "nbits": pq_nbits, "d": d}
