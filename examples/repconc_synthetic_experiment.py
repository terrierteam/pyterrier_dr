"""Small synthetic experiment: dense vs untrained OPQ vs JPQ vs RepCONC.

A self-contained, CPU-friendly comparison of retrieval methods on a tiny
hashed bag-of-words corpus, in the style of the unit tests in
``tests/test_repconc_train.py``. Document ``i`` contains topic words
``w{i}_0``/``w{i}_1`` and query ``i`` shares them, so relevance is learnable.

Compared pipelines (evaluated on 20 held-in-training queries):

* ``dense-pre``       -- the original encoder, exact search (the FlexIndex baseline)
* ``opq-untrained``   -- faiss OPQ+PQ warm-up only (quantisation, no training), ADC
* ``dense-post``      -- the RepCONC-trained encoder, exact search (upper bound)
* ``jpq``             -- JPQ (Li et al. 2023) trained pipeline, ADC
* ``repconc``         -- RepCONC (Zhan et al. 2022) two-stage, ADC
* ``repconc-nocst``   -- RepCONC stage 1+2 with the Sinkhorn constraint
                         disabled (the paper's no-constraint ablation), ADC

Run (from the repository root; ``PYTHONPATH=.`` picks up the local
``pyterrier_dr`` rather than an installed release)::

    PYTHONPATH=. python examples/repconc_synthetic_experiment.py
"""
import shutil
import tempfile
import zlib

import numpy as np
import pandas as pd
import torch
from ir_measures import RR, nDCG, Recall
import pyterrier as pt

from pyterrier_dr import FlexIndex
from pyterrier_dr.biencoder import BiEncoder
from pyterrier_dr.jpq import JPQTrainer
from pyterrier_dr.repconc import RepCONCTrainer, RepCONCIndex

# Workaround for a pre-existing environment incompatibility, unrelated to
# RepCONC: `datasets`' torch formatter runs `from torchvision.io import
# VideoReader` whenever it tensorises a batch, but torchvision >= 0.26 removed
# VideoReader. The JPQ arm builds its dataloader through HF `datasets`, so the
# JPQ fit would crash on CPU in this env. Stub the symbol (it is only ever used
# in an isinstance check against video data, never against our tensors) so the
# JPQ arm can run.
try:
    import torchvision.io as _tvio
    if not hasattr(_tvio, "VideoReader"):
        _tvio.VideoReader = type("VideoReader", (), {})
except Exception:
    pass

# ------------------------------------------------------------------ toy model

D, M, NBITS, VOCAB, NDOCS = 64, 8, 4, 4096, 500
TOPIC = 2
EVAL_I0, EVAL_N = 480, 20  # eval queries (their docs are in the training set)


def _tok_id(tok: str) -> int:
    return zlib.crc32(tok.encode('utf-8')) % VOCAB


class ToyBiEncoder(BiEncoder):
    """A small in-memory bi-encoder: mean of hashed token embeddings."""

    def __init__(self, d=D, seed=0):
        super().__init__(verbose=False)
        g = torch.Generator().manual_seed(seed)
        self.model = torch.nn.Embedding(VOCAB, d)
        with torch.no_grad():
            self.model.weight.copy_(torch.randn(VOCAB, d, generator=g) / np.sqrt(d))

    def _enc(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            toks = [_tok_id(w) for w in str(t).split()]
            out.append(self.model(torch.tensor(toks)).mean(0) if toks else torch.zeros(D))
        return torch.stack(out)

    def encode_queries_torch(self, texts, batch_size=None):
        return self._enc(texts)

    def encode_docs_torch(self, texts, batch_size=None):
        return self._enc(texts)

    def encode_docs(self, texts, batch_size=None):
        with torch.no_grad():
            return self._enc(texts).numpy()

    def __str__(self):
        return "ToyBiEncoder"


def make_corpus():
    rng = np.random.default_rng(7)
    doc_words = []
    doc_texts = []
    for i in range(NDOCS):
        words = [f"w{i}_0", f"w{i}_1"] + [f"filler_{int(x)}" for x in rng.integers(0, 50, size=4)]
        doc_words.append(words)
        doc_texts.append(" ".join(words))
    return doc_texts, doc_words


def query_text(i, doc_words):
    return f"q{i} " + " ".join(doc_words[i][:TOPIC])


def eval_frames(doc_words):
    idxs = list(range(EVAL_I0, EVAL_I0 + EVAL_N))
    q = pd.DataFrame({"qid": [f"q{i}" for i in idxs],
                      "query": [query_text(i, doc_words) for i in idxs]})
    r = pd.DataFrame({"qid": [f"q{i}" for i in idxs],
                      "docno": [f"doc{i}" for i in idxs], "label": [1] * EVAL_N})
    return q, r


def main():
    tmp = tempfile.mkdtemp("repconc_experiment")
    print(f"work dir: {tmp}")
    try:
        doc_texts, doc_words = make_corpus()
        base_bi = ToyBiEncoder()
        flex = FlexIndex(f"{tmp}/base.flex", verbose=False)
        flex.indexer(mode='overwrite').index(
            [{'docno': f"doc{i}", 'doc_vec': base_bi.encode_docs([t])[0]} for i, t in enumerate(doc_texts)])
        eval_queries, eval_qrels = eval_frames(doc_words)
        triples = [{"qid": f"q{i}", "query": query_text(i, doc_words),
                    "doc_id_a": doc_texts[i],
                    "doc_id_b": [doc_texts[(i + 1) % NDOCS], doc_texts[(i + 2) % NDOCS]]}
                   for i in range(NDOCS)]
        pairs = [{"qid": f"q{i}", "query": query_text(i, doc_words), "doc_id_a": f"doc{i}"}
                 for i in range(0, NDOCS, 2)]

        results = {}

        def score(name, pipeline, topk=100):
            res = pipeline(eval_queries)
            r = pt.Evaluate(res, eval_qrels, metrics=[nDCG@10, RR@10, Recall@100])
            results[name] = r
            print(f"{name:16s} nDCG@10={r['nDCG@10']:.4f}  MRR@10={r['RR@10']:.4f}  R@100={r['R@100']:.4f}")

        # 1) dense baseline (original encoder, exact search)
        score("dense-pre", base_bi >> flex.retriever())

        # 2) OPQ+PQ warm-up only, original encoder (quantisation, no training)
        warm = RepCONCTrainer(ToyBiEncoder(), flex, M=M, nbits=NBITS, device='cpu', seed=3)
        warm._warmup()
        opq_index = RepCONCIndex.build(
            f"{tmp}/opq", docnos=list(flex.docnos().fwd), codes=warm._codes,
            centroids=warm.pq.centroids.detach().cpu().numpy(),
            rotation=warm.pq.rotation.detach().cpu().numpy(), mode="overwrite")
        score("opq-untrained", ToyBiEncoder() >> opq_index.retriever_pq(topk=100))

        # 3) RepCONC (stage 1 + stage 2)
        bi_rc = ToyBiEncoder()
        rc = RepCONCTrainer(bi_rc, flex, M=M, nbits=NBITS, device='cpu', seed=3)
        rc.train_stage1(triples, total_steps=40, batch_size=8, lr_encoder=5e-3, lr_centroid=1e-2,
                        lambda_mse=0.2)
        rc.train_stage2(pairs, total_steps=4, batch_size=8, lr_query=5e-4, lr_centroid=1e-3)
        rc_index = rc.repconc_index(f"{tmp}/repconc", corpus_text=doc_texts)
        score("repconc", rc.biencoder.query_encoder() >> rc_index.retriever_pq(topk=100))

        # dense upper bound for the RepCONC-trained encoder
        flex_post = FlexIndex(f"{tmp}/post.flex", verbose=False)
        flex_post.indexer(mode='overwrite').index(
            [{'docno': f"doc{i}", 'doc_vec': bi_rc.encode_docs([t])[0]} for i, t in enumerate(doc_texts)])
        score("dense-post", bi_rc.query_encoder() >> flex_post.retriever())

        # 4) RepCONC, no-constraint ablation (plain argmin codes during training).
        #    The Sinkhorn-constrained assignment is disabled for stage 1; note
        #    that train_stage2 recomputes the fixed codes with use_constraint
        #    toggled internally (and restores the flag in a finally block), so
        #    the flag is re-asserted before stage 2 to keep the ablation clean.
        bi_nc = ToyBiEncoder()
        nc = RepCONCTrainer(bi_nc, flex, M=M, nbits=NBITS, device='cpu', seed=3)
        nc._warmup()
        nc.pq.use_constraint = False
        nc.train_stage1(triples, total_steps=40, batch_size=8, lr_encoder=5e-3, lr_centroid=1e-2,
                        lambda_mse=0.2)
        nc.pq.use_constraint = False
        nc.train_stage2(pairs, total_steps=4, batch_size=8, lr_query=5e-4, lr_centroid=1e-3)
        nc_index = nc.repconc_index(f"{tmp}/repconc_nocst", corpus_text=doc_texts)
        score("repconc-nocst", nc.biencoder.query_encoder() >> nc_index.retriever_pq(topk=100))

        # 5) JPQ
        jpq_pairs = [{"qid": f"q{i}", "query": query_text(i, doc_words),
                      "doc_id_a": f"doc{i}", "doc_id_b": f"doc{(i + 1) % NDOCS}"}
                     for i in range(0, NDOCS, 2)]
        bi_jpq = ToyBiEncoder()
        jpq = JPQTrainer(bi_jpq, flex, M=M, nbits=NBITS, pq_impl='faiss2opq', device='cpu')
        jpq.fit(jpq_pairs, total_steps=40, batch_size=32, lr=1e-4,
                eval_queries=eval_queries, eval_qrels=eval_qrels,
                valid_every=20, patience=3)
        jpq_index = jpq.jpq_index(f"{tmp}/jpq")
        score("jpq", jpq.query_encoder >> jpq_index.retriever_pq(topk=100))

        df = pd.DataFrame(results).T
        df.index.name = "method"
        print("\n" + df.round(4).to_string())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
