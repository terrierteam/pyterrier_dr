import pyterrier_dr
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import faiss
from .utils import timer, l2_normalize_np
import torch
from pyterrier import tqdm
import os
import json
from .index import JPQIndex
import math
from datasets import IterableDataset


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

# ==== JPQ core (subset-train -> subset retriever) ====
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class QueryEncoderBase(nn.Module):
    def encode_texts(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        raise NotImplementedError
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)

class PTRDRQueryEncoderAsModule(QueryEncoderBase):
    """Treat pyterrier_dr encoders as a frozen QueryEncoderBase."""
    def __init__(self, dr_model, normalize: bool = True, batch_size: int = 64):
        super().__init__()
        self.dr = dr_model
        self.normalize = normalize
        self.batch = batch_size
    def to(self, device):  # frozen
        return self
    def parameters(self, recurse: bool = True):
        return iter(())
    def encode_texts(self, texts: List[str], batch_size: int = 128) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                chunk = texts[i:i+self.batch]
                if hasattr(self.dr, "encode_queries"):
                    arr = self.dr.encode_queries(chunk, batch_size=self.batch)
                else:
                    qe = getattr(self.dr, "query_encoder", lambda **kw: None)(batch_size=self.batch)
                    if qe and hasattr(qe, "encode"):
                        arr = qe.encode(chunk, batch_size=self.batch)
                    else:
                        raise RuntimeError("dr_model must expose encode_queries or query_encoder().encode")
                t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
                if self.normalize:
                    t = t / (t.norm(dim=1, keepdim=True) + 1e-12)
                outs.append(t)
        return torch.cat(outs, dim=0)

class JPQEmbeddingModel(nn.Module):
    def __init__(self, pq: faiss.ProductQuantizer):
        super().__init__()
        self.M, self.k, self.dsub = pq.M, pq.ksub, pq.dsub
        self.sub_embeddings = nn.ModuleList([nn.Embedding(self.k, self.dsub) for _ in range(self.M)])
        cents = faiss.vector_to_array(pq.centroids).reshape(self.M, self.k, self.dsub)
        for i in range(self.M):
            self.sub_embeddings[i].weight.data.copy_(torch.from_numpy(cents[i]).float())
    def forward(self, doc_codes: torch.Tensor) -> torch.Tensor:
        parts = [self.sub_embeddings[i](doc_codes[:, i]) for i in range(self.M)]
        return torch.cat(parts, dim=1)

class JPQLoss(nn.Module):
    def __init__(self, query_encoder: QueryEncoderBase, passage_encoder: JPQEmbeddingModel):
        super().__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
    def forward(self, batch):
        q = self.query_encoder.encode_texts(batch['query_text'])      # CPU
        dev = next(self.passage_encoder.parameters()).device
        q = q.to(dev, non_blocking=True)
        pos = batch['pos_codes'].to(dev, non_blocking=True)
        neg = batch['neg_codes'].to(dev, non_blocking=True)
        pos = self.passage_encoder(pos)
        neg = self.passage_encoder(neg)
        s_pos = self.cos_sim(q, pos); s_neg = self.cos_sim(q, neg)
        scores = torch.stack([s_pos, s_neg], dim=1)
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=dev)
        return self.loss_fct(scores, labels)


class BiEncoder:
    def __init__(self, query_encoder: QueryEncoderBase, passage_encoder: JPQEmbeddingModel):
        self.query = query_encoder
        self.passage = passage_encoder
    def to(self, device: str):
        self.query = self.query.to(device)
        self.passage = self.passage.to(device)
        return self


class JPQTrainer:

    def __init__(
            self,
            existing_model : pyterrier_dr.BiEncoder,
            existing_index: pyterrier_dr.FlexIndex,
            device = None,
            pq_M: int = 8, 
            pq_nbits: int = 8):
        super().__init__()
        self.fitted = False
        self.existing_index = existing_index
        self.d = existing_index.payload()[2]['vec_size']
        self.pq_M = pq_M
        self.pq_nbits = pq_nbits
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ptdr_enc_module_tct = PTRDRQueryEncoderAsModule(existing_model, normalize=True, batch_size=64)
        self.model = BiEncoder(ptdr_enc_module_tct, JPQEmbeddingModel(faiss.ProductQuantizer(self.d, pq_M, pq_nbits)))

    def fit(
            self,
            training_docpairs,
            queries,
            code_batch_size: int = 20_0000, 
            recon_batch_size: int = 20000,
            pq_sample_size: int = 10_000, # how many doc vectors to use to train PQ centroids
            docid_subset: Optional[List[int] | List[str] | int] = None, # how many doc vectors to use to train the sub-id embeddings 
            epochs: int = 3, 
            batch_size: int = 32, 
            patience : int = 2,
            extra_neg_pool : int = 0,
            eval_qrels_df : Optional[pd.DataFrame] = None,
            lr: float = 2e-5):
        
        rng = np.random.RandomState(42)
        full_docnos, vecs_mem, N, d = flex_payload(self.existing_index)

        print(f"Ingesting query mapping ")
        queries = {e.query_id : e.text for e in queries}

        ##### TODO: figure out this stuff

         # ------- split / subset -------
        # qrels_all = defaultdict(set)
        # # I think training_docpairs is a irds iterator
        # for dp in training_docpairs:
        #     qrels_all[dp.query_id].add(dp.doc_id_a)
        # all_qids = [qid for qid in qrels_all.keys() if qid in training_queries]
        # val_size = max(1, int(len(all_qids) * val_ratio))
        # val_qids = set(rng.choice(all_qids, size=val_size, replace=False).tolist())
        # train_qids = set(all_qids) - val_qids
        # train_pairs = [dp for dp in training_docpairs if dp.query_id in train_qids]
        # val_qrels = {qid: qrels_all[qid] for qid in val_qids}
        # print(f"[JPQ] train_qids={len(train_qids)}, val_qids={len(val_qids)}, train_pairs={len(train_pairs)}")

        # eval_pool == "union"
        id2idx = inv_map(self.existing_index)
        # train_doc_ids = {dp.doc_id_a for dp in train_pairs} | {dp.doc_id_b for dp in train_pairs}
        # val_pos_doc_ids = set().union(*[val_qrels[qid] for qid in val_qrels]) if val_qrels else set()
        # selected_doc_ids = (train_doc_ids | val_pos_doc_ids) if eval_pool == "union" else set(train_doc_ids)

        # if eval_qrels_df is not None and len(eval_qrels_df) > 0:
        #     pos_df = eval_qrels_df[eval_qrels_df['label'] >= eval_label_min]
        #     eval_pos = set(map(str, pos_df['docno'].astype(str).tolist()))
        #     eval_pos_in_index = {d for d in eval_pos if d in id2idx}
        #     selected_doc_ids |= eval_pos_in_index
        #     print(f"[EVAL POOL] covered_dev_pos={len(eval_pos_in_index)}, total_dev_pos={len(eval_pos)}")

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
                # docid_subset = rng.choice(full_docnos, size=docid_subset, replace=False).tolist()
                selected_doc_ids = self.existing_index.payload()[0].fwd[docid_subset]
            elif isinstance(docid_subset, list):  
                if isinstance(docid_subset[0], int): # use the provided list of int docid
                    selected_doc_ids = self.existing_index.payload()[0].rev(docid_subset)  # int -> str
                else:   # use the provided list of str docnos
                    assert isinstance(docid_subset[0], str)
                    selected_doc_ids = self.existing_index.payload()[0].fwd[docid_subset]
            print(f"[SUBSET] using {len(docid_subset)} docs from docid_subset")
            
        selected_doc_ids = list(selected_doc_ids)
        # I dont think we need to shuffle, as its likely a random sample. would be better sorted!
        # rng.shuffle(selected_doc_ids)
        # selected_doc_ids.sort()
        
        sel_indices = np.array([int(id2idx[did]) for did in selected_doc_ids], dtype=np.int64)
        sel_inv = {did: i for i, did in enumerate(selected_doc_ids)}
        # make it a set of easier dataset filtering below
        selected_doc_ids = set(selected_doc_ids)

        # ------- PQ -------
        codes_sel, pq = self._compute_PQ(pq_sample_size, code_batch_size, sel_indices, vecs_mem, rng)
        
        # ------- dataloader -------
        dl = self._dataloader(training_docpairs, batch_size, selected_doc_ids, sel_inv, queries, codes_sel)
        
        # ------- training the sub-id embeddings -------
        self._training_loop(self.model, pq, dl, epochs, lr, patience)
        self.fitted = True

    
    def _dataloader(self, training_docpairs, batch_size, selected_doc_ids, sel_inv, queries, codes_sel):
        # make it a set of easier dataset filtering below
        selected_doc_ids = set(selected_doc_ids)

        # bring over the dataloader stuff
        def _gen():
            for dp in training_docpairs.docpairs_iter():
                yield dp._asdict()

        dataset = IterableDataset.from_generator(_gen)

        def queries_and_codes(x):
            return {
                    'query_text': queries[x.query_id],
                    'pos_codes': codes_sel[sel_inv[x.doc_id_a]],#not sure about codes_sel
                    'neg_codes': codes_sel[sel_inv[x.doc_id_b]],#not sure about codes_sel
                }
        def collate(batch):
            return {
                'query_text': [b['query_text'] for b in batch],
                'pos_codes': torch.stack([b['pos_codes'] for b in batch]),
                'neg_codes': torch.stack([b['neg_codes'] for b in batch]),
            }
        dl = DataLoader(
            # remove train queries for documents that arent in our selection
            dataset.filter(lambda row: row['doc_id_a'] in selected_doc_ids and row['doc_id_b'] in selected_doc_ids).map(queries_and_codes), 
            batch_size=batch_size, 
            collate_fn=collate)
        return dl

    def _compute_PQ(self, 
                    pq_sample_size: int, # how many doc vectors to use to train PQ centroids
                    code_batch_size: int, # how many doc vectors to process in a batch when computing PQ codes
                    sel_indices : np.array, # which document indices (into full vecs_mem) to compute codes for
                    vecs_mem : np.array, # vector store
                    rng # random state
                    ) -> Tuple[np.ndarray, faiss.ProductQuantizer]:
        
        # train PQ on a random sample of the selected docs
        pq = faiss.ProductQuantizer(self.d, self.pq_M, self.pq_nbits)
        sample_size = min(pq_sample_size, len(sel_indices))
        print("[PQ] training on %d documents..." % sample_size)
        sample_idx = rng.choice(sel_indices, size=sample_size, replace=False)
        with timer(f"PQ / train (samples={len(sample_idx):,})"):
            xb = l2_normalize_np(vecs_mem[sample_idx])
            pq.train(xb)

        print("[PQ] computing codes for %d selected docs in chunks of %d..." % (len(sel_indices), code_batch_size))
        codes_sel = np.empty((len(sel_indices), self.pq_M), dtype=np.uint8)
        with timer("PQ / compute codes (selected)"):
            for i in tqdm(range(0, len(sel_indices), code_batch_size), desc="PQ compute_codes", leave=False):
                part = sel_indices[i:i+code_batch_size]
                x = l2_normalize_np(vecs_mem[part])
                packed = pq.compute_codes(x)
                if packed.shape[1] != self.pq_M: # not sure what this if is for?
                    unpacked = _unpack_pq_codes_batch(packed, self.pq_M, self.pq_nbits)
                else:
                    unpacked = packed
                codes_sel[i:i+len(part)] = unpacked
        
        # TODO: we should check how the average/min/max codes are observed in sel_indices
        # give that sel_indices is a random sample of sel_indices, it should be fairly uniform
        # as a codes are centroids in the vector space defined in the small sample_size set, which
        # is a subset of the sel_indices set, it should be ok.
        return codes_sel, pq
    
    def _training_loop(self, model, pq, dl : DataLoader, epochs, lr, patience, max_steps=None, valid_every=100):

        loss_f = JPQLoss(model.query, model.passage).to(self.device)
        model.query.eval()
        for p in model.query.parameters(recurse=True): p.requires_grad = False
        optimizer = torch.optim.AdamW(list(model.passage.parameters()), lr=lr, weight_decay=0.0)

        import tempfile
        out_dir = tempfile.mkdtemp()
        model_dir = os.path.join(out_dir, "model"); 
        os.makedirs(model_dir, exist_ok=True)

        with torch.no_grad():
            init_w = torch.cat([emb.weight.detach().flatten() for emb in self.model.passage.sub_embeddings]).to(self.device)
        
        lambda_reg = 1e-4
        best_state = None
        total_steps = 0

        for ep in range(1, epochs + 1):
            if total_steps >= (max_steps or math.inf):
                print("[JPQ] reached max_steps"); 
                break
            
            model.passage.to(self.device).train()
            ep_loss = 0.0
            counter = 0
            with timer(f"JPQ / epoch {ep}"):
                for batch in tqdm(dl):
                    optimizer.zero_grad()
                    base = loss_f(batch)
                    cur_w = torch.cat([emb.weight.flatten() for emb in model.passage.sub_embeddings])
                    reg = lambda_reg * torch.mean((cur_w - init_w.to(cur_w.device))**2)
                    loss = base + reg
                    loss.backward()
                    optimizer.step()
                    ep_loss += float(base.item())

                    # counting stuff
                    counter += dl.batch_size
                    total_steps += dl.batch_size
                    if counter > valid_every:
                        break
                    if counter >= max_steps:
                        break
            print(f"[JPQ] epoch {ep}/{epochs} train_loss={ep_loss/len(dl):.4f}")

            model.passage.eval()
            stats = self.run_validation(model)
            model.passage.to(self.device).train()
            print(f"[JPQ][val] MRR@10={stats['MRR@10']:.4f} Recall@50={stats['Recall@50']:.4f} NDCG@10={stats['NDCG@10']:.4f}")

            if stats['MRR@10'] > best_mrr + 1e-5:
                best_mrr = stats['MRR@10']; bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.passage.state_dict().items()}
                torch.save(model.passage.state_dict(), os.path.join(model_dir, "jpq_passage.pt"))
                with open(os.path.join(model_dir, "pq_meta.json"), "w") as f:
                    json.dump({"M": self.pq_M, "nbits": self.pq_nbits, "d": self.d}, f)
                cents0 = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).astype('float32')
                np.save(os.path.join(model_dir, "pq_init_centroids.npy"), cents0)
            else:
                bad += 1
                if bad >= patience:
                    print("[JPQ] Early stopping."); break

        if best_state is not None:
            model.passage.load_state_dict(best_state)

    def _run_validation(model):
        val_qids = sorted([qid for qid in gt_idx.keys() if qid in training_queries])
        if not val_qids: return {"MRR@10":0.0,"Recall@50":0.0,"NDCG@10":0.0}
        texts = [training_queries[qid] for qid in val_qids]
        with torch.no_grad():
            Q_t = model.query.encode_texts(texts, batch_size=256)
        Q = Q_t.detach().cpu().numpy().astype('float32')
        Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)

        dim = model.passage.sub_embeddings[0].embedding_dim * model.passage.M
        index = faiss.IndexFlatIP(dim)
        pm = model.passage.to("cpu").eval()
        with torch.no_grad():
            for i in range(0, len(codes_sel), recon_batch_size):
                chunk = torch.from_numpy(codes_sel[i:i+recon_batch_size]).long()
                embs = pm(chunk)
                embs = (embs / (embs.norm(dim=1, keepdim=True) + 1e-12)).detach().cpu().numpy().astype('float32')
                index.add(embs)
        D, I = index.search(Q, topk_eval)
        return _val_metrics(I, val_qids, gt_idx)

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
        return JPQIndex.build(dest)


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
