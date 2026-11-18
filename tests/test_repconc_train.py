"""End-to-end tests of the RepCONC trainer on a small synthetic corpus (CPU).

Toy setup: a hashed bag-of-words bi-encoder (a single ``nn.Embedding``), and a
320-document corpus in which document ``i`` contains the topic words
``w{i}_0``/``w{i}_1`` and query ``i`` shares those words with document ``i``.
The relevance structure is therefore learnable, and after a short stage-1
training run the encoder's dense retrieval (and the PQ index built from it)
recovers the positives.

All trainers start from the same deterministic initial encoder weights, so a
single FlexIndex (holding the *pre-training* document vectors) is valid for
every fresh trainer.
"""
import tempfile
import unittest
import zlib

import numpy as np
import pandas as pd
import torch

# NB: pyterrier_dr (which loads torch) must be imported before faiss; on macOS
# the two bundled OpenMP runtimes conflict if faiss's is loaded first (see
# pyterrier_dr.util.ensure_faiss_omp_compat).
from pyterrier_dr import FlexIndex
from pyterrier_dr.biencoder import BiEncoder
from pyterrier_dr.repconc import RepCONCTrainer, RepCONCIndex
from pyterrier_dr.repconc.retriever import RepCONCRetriever
from pyterrier_dr.repconc.data import normalize_stage1, get_dataloader

D, M, NBITS, VOCAB, NDOCS = 64, 8, 4, 4096, 320
TOPIC = 2  # topic words shared between each query and its positive document


def _tok_id(tok: str) -> int:
    # stable (process-independent) token -> bucket hash
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


class _Corpus:
    """The shared synthetic corpus (built once)."""

    doc_words = None
    doc_texts = None
    flex = None

    @classmethod
    def build(cls, path):
        if cls.flex is not None:
            return
        rng = np.random.default_rng(7)
        cls.doc_words = []
        cls.doc_texts = []
        for i in range(NDOCS):
            words = [f"w{i}_0", f"w{i}_1"] + [f"filler_{int(x)}" for x in rng.integers(0, 50, size=4)]
            cls.doc_words.append(words)
            cls.doc_texts.append(" ".join(words))
        bi = ToyBiEncoder()
        flex = FlexIndex(path, verbose=False)
        flex.indexer(mode='overwrite').index(
            [{'docno': f"doc{i}", 'doc_vec': bi.encode_docs([t])[0]} for i, t in enumerate(cls.doc_texts)])
        cls.flex = flex

    @classmethod
    def query_text(cls, i):
        return f"q{i} " + " ".join(cls.doc_words[i][:TOPIC])

    @classmethod
    def triples(cls, n=NDOCS):
        return [{"qid": f"q{i}", "query": cls.query_text(i),
                 "doc_id_a": cls.doc_texts[i],
                 "doc_id_b": [cls.doc_texts[(i + 1) % NDOCS], cls.doc_texts[(i + 2) % NDOCS]]}
                for i in range(n)]

    @classmethod
    def pairs(cls, n=16):
        return [{"qid": f"q{i}", "query": cls.query_text(i), "doc_id_a": f"doc{i}"}
                for i in range(0, n * 4, 4)]

    @classmethod
    def eval_frames(cls, i0=300, n=8):
        idxs = list(range(i0, i0 + n))
        q = pd.DataFrame({"qid": [f"q{i}" for i in idxs],
                          "query": [cls.query_text(i) for i in idxs]})
        r = pd.DataFrame({"qid": [f"q{i}" for i in idxs],
                          "docno": [f"doc{i}" for i in idxs], "label": [1] * n})
        return q, r


class TestRepCONCTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp("repconc_train_test")
        _Corpus.build(f"{cls.tmp}/flex")

    def _trainer(self):
        return RepCONCTrainer(ToyBiEncoder(), _Corpus.flex, M=M, nbits=NBITS, device='cpu', seed=3)

    # ------------------------------------------------------------------ stage 1

    def test_stage1_loss_decreases(self):
        """Overfitting a fixed batch of triples must reduce the stage-1 loss."""
        trainer = self._trainer()
        trainer._warmup()
        items = normalize_stage1(_Corpus.triples(8))
        loader = get_dataloader(items, batch_size=8, shuffle=False, seed=3)
        opt = torch.optim.AdamW([
            {"params": trainer.model.encoder_parameters(), "lr": 5e-3},
            {"params": [trainer.pq.centroids], "lr": 1e-2}], weight_decay=0.0)
        trainer.model.set_train(True)
        losses = []
        it = iter(loader)
        for _ in range(40):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            losses.append(trainer._stage1_step(batch, opt, 1.0, 0.2, None, 32))
        self.assertLess(losses[-1], 0.5 * losses[0], f"loss did not decrease: {losses}")

    def test_fit_stage1_stage2_index_retrieval(self):
        """Full two-stage flow: train, build an index (re-encoding the corpus
        with the trained encoder), and retrieve the positives."""
        bi = ToyBiEncoder()
        trainer = RepCONCTrainer(bi, _Corpus.flex, M=M, nbits=NBITS, device='cpu', seed=3)
        eval_queries, eval_qrels = _Corpus.eval_frames()
        trainer.train_stage1(_Corpus.triples(), total_steps=40, batch_size=8,
                             lr_encoder=5e-3, lr_centroid=1e-2, lambda_mse=0.2,
                             eval_queries=eval_queries, eval_qrels=eval_qrels,
                             valid_every=20, patience=3)
        # stage 2 keeps the codes fixed while the centroids keep moving, so
        # (as in the official stage-2 script) it needs a much smaller LR than
        # stage 1: large centroid updates would invalidate the fixed codes
        trainer.train_stage2(_Corpus.pairs(), total_steps=4, batch_size=8,
                             lr_query=5e-4, lr_centroid=1e-3,
                             eval_queries=eval_queries, eval_qrels=eval_qrels,
                             valid_every=2, patience=3)
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.pq)

        idx = trainer.repconc_index(f"{self.tmp}/idx", corpus_text=_Corpus.doc_texts)
        self.assertEqual(len(idx), NDOCS)

        # text-based retrieval: positives should be in the top-3
        res = RepCONCRetriever(bi, idx, topk=5)(eval_queries.copy())
        self.assertEqual(len(res), 8 * 5)
        hits = 0
        for i in range(300, 308):
            sub = res[res["qid"] == f"q{i}"].sort_values("rank")
            self.assertEqual(len(sub), 5)
            self.assertTrue(set(sub["docno"]) <= set(f"doc{j}" for j in range(NDOCS)))
            if f"doc{i}" in sub["docno"].head(3).tolist():
                hits += 1
        self.assertGreaterEqual(hits, 6, f"only {hits}/8 positives in top-3")

        # flat and PQ retrievers agree on the (reconstructed) documents
        topics = eval_queries.copy()
        topics["query_vec"] = bi.encode_queries(topics["query"].tolist()).tolist()
        res_flat = idx.retriever_flat(topk=5)(topics)
        res_pq = idx.retriever_pq(topk=5)(topics)
        self.assertEqual(res_flat["docno"].astype(str).tolist(), res_pq["docno"].astype(str).tolist())
        np.testing.assert_allclose(res_flat["score"].to_numpy(), res_pq["score"].to_numpy(), atol=1e-4)

        # a reloaded index retrieves the same documents
        idx2 = RepCONCIndex(f"{self.tmp}/idx")
        res2 = idx2.retriever_pq(topk=5)(topics)
        self.assertEqual(res_pq["docno"].astype(str).tolist(), res2["docno"].astype(str).tolist())

    def test_repconc_index_corpus_text_iterdict(self):
        """``corpus_text`` may be an iterable of ``{'docno', 'text'}`` dicts in
        any order (the standard PyTerrier ``Indexer.index()`` input format);
        it must produce the same index as the ordered-texts form. Unknown
        docnos are skipped and index docnos without a text fall back to the
        pre-training FlexIndex vectors."""
        trainer = self._trainer()
        trainer.train_stage1(_Corpus.triples(), total_steps=8, batch_size=8,
                             lr_encoder=5e-3, lr_centroid=1e-2, lambda_mse=0.2)
        idx_str = trainer.repconc_index(f"{self.tmp}/idx_str", corpus_text=_Corpus.doc_texts)
        records = [{"docno": f"doc{i}", "text": t} for i, t in enumerate(_Corpus.doc_texts)]
        idx_dict = trainer.repconc_index(f"{self.tmp}/idx_dict", corpus_text=records[::-1])
        self.assertEqual(list(idx_str.docnos.fwd), list(idx_dict.docnos.fwd))
        np.testing.assert_array_equal(np.asarray(idx_str.codes), np.asarray(idx_dict.codes))
        np.testing.assert_array_equal(np.asarray(idx_str.subvecs), np.asarray(idx_dict.subvecs))

        # an unknown docno is skipped; a missing docno falls back (no crash,
        # the full index is still built)
        partial = records[1:] + [{"docno": "doc9999", "text": "not in the index"}]
        idx_part = trainer.repconc_index(f"{self.tmp}/idx_part", corpus_text=partial)
        self.assertEqual(len(idx_part), NDOCS)

        # malformed inputs are rejected
        with self.assertRaises(ValueError):
            trainer.repconc_index(f"{self.tmp}/idx_bad", corpus_text=[{"docno": "doc0"}])
        with self.assertRaises(ValueError):
            trainer.repconc_index(f"{self.tmp}/idx_bad2", corpus_text={"docno": "doc0", "text": "x"})

    # ------------------------------------------------------------------ fit()

    def test_fit(self):
        trainer = self._trainer()
        trainer.fit(_Corpus.triples(), stage2_pairs=None,
                    stage1_kwargs={"total_steps": 4, "batch_size": 8,
                                   "lr_encoder": 5e-3, "lr_centroid": 1e-2, "lambda_mse": 0.2})
        self.assertTrue(trainer.fitted)
        self.assertFalse(trainer.model.training)
        idx = trainer.repconc_index(f"{self.tmp}/idx_fit", corpus_text=_Corpus.doc_texts)
        self.assertEqual(len(idx), NDOCS)
        res = idx.retriever_pq(topk=5)(
            pd.DataFrame({"qid": ["q300"], "query": [_Corpus.query_text(300)],
                          "query_vec": [ToyBiEncoder().encode_queries([_Corpus.query_text(300)])[0]]}))
        self.assertEqual(len(res), 5)

    # -------------------------------------------------------------- checkpoints

    def test_checkpoint_roundtrip(self):
        trainer = self._trainer()
        trainer.train_stage1(_Corpus.triples(), total_steps=10, batch_size=8,
                             lr_encoder=5e-3, lr_centroid=1e-2, lambda_mse=0.2)
        path = f"{self.tmp}/ckpt.pt"
        trainer.save_checkpoint(path)

        t2 = self._trainer()
        t2._warmup()
        step, _ = t2.load_checkpoint(path)
        self.assertEqual(step, 10)
        np.testing.assert_array_equal(
            t2.pq.centroids.detach().cpu().numpy(), trainer.pq.centroids.detach().cpu().numpy())
        np.testing.assert_array_equal(t2.pq.rotation.cpu().numpy(), trainer.pq.rotation.cpu().numpy())
        orig = {name: m.state_dict() for name, m in trainer.model.encoder_modules()}
        for name, m in t2.model.encoder_modules():
            self.assertIn(name, orig, f"encoder module {name!r} missing after load")
            for k, v in m.state_dict().items():
                np.testing.assert_array_equal(v.cpu().numpy(), orig[name][k].cpu().numpy(), err_msg=f"{name}.{k}")

    # ----------------------------------------------------------------- errors

    def test_stage2_unknown_docno(self):
        trainer = self._trainer()
        trainer._warmup()
        with self.assertRaises(ValueError):
            trainer.train_stage2([{"qid": "q", "query": "hello", "doc_id_a": "doc9999"}],
                                 total_steps=1, batch_size=8)


if __name__ == "__main__":
    unittest.main()
