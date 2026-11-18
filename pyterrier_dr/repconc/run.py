"""CLI for RepCONC training and evaluation.

Mirrors the conventions of :mod:`pyterrier_dr.jpq.run`, adapted to RepCONC's
training paradigm: stage one takes *text triples* (query text, positive
document text, negative document text), so in addition to the base FlexIndex
and the training docpair dataset, a *corpus* (docid -> text) is required.

Example (MS MARCO passage)::

    python -m pyterrier_dr.repconc.run \
        --base-index /path/to/msmarco_passage.flex \
        --target-dir /path/to/outputs \
        --model-name tct_colbert \
        --train-ds msmarco-passage/train \
        --corpus msmarco-passage/collection \
        --eval-ds irds:msmarco-passage/dev/small \
        --stage2

Note that the FlexIndex docnos must match the training/corpus docids (the
same convention the JPQ workflow assumes).
"""
import argparse
import itertools
import os
import re
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from ir_measures import RR, Recall, nDCG

import ir_datasets
import pyterrier_dr
import pyterrier as pt
import torch

from pyterrier_dr.repconc import RepCONCTrainer

torch.set_float32_matmul_precision('high')

# The DEFAULT_INIT_BY_NAME dict maps a model name to a python expression that
# instantiates the model (copied from pyterrier_dr.jpq.run).
DEFAULT_INIT_BY_NAME: dict[str, str] = {
    "tct_colbert": "pyterrier_dr.TctColBert.hnp()",
    "tas_b": "pyterrier_dr.TasB()",
    "star": "pyterrier_dr.STAR('/root/nfs/jpq/provided_models/star/')",
    "adore_star": "pyterrier_dr.STAR('/root/nfs/jpq/provided_models/adore_star/')",
    "repllama": "pyterrier_dr.RepLLama.v1_7b()",
    "e5": "pyterrier_dr.E5()",
    "dragon": 'pyterrier_dr.Dragon()',
    "lion": 'pyterrier_dr.LionLlamaDense()',
}


def instantiate_model(model_name: str):
    """Instantiate the model by evaluating the resolved expression in a
    restricted namespace. Only whitelisted globals are visible to eval()."""
    allowed_globals = {
        "pyterrier_dr": pyterrier_dr,
    }
    expr = DEFAULT_INIT_BY_NAME[model_name]
    try:
        obj = eval(expr, allowed_globals, {})   # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate model with expression: {expr}\n"
            f"Error: {e}"
        ) from e
    return obj


_slug_re = re.compile(r"[^a-zA-Z0-9._-]+")


def _slug(s: str, maxlen: int = 64) -> str:
    s = str(s).strip().replace(" ", "-")
    s = _slug_re.sub("-", s)
    return s[:maxlen].strip("-")


def compute_index_name(data_cfg, train_cfg) -> str:
    # Compact, human-readable summary of the run
    parts = [
        _slug(data_cfg.train_ds),
        _slug(data_cfg.model_name),
        f"M{train_cfg.M}_nbits{train_cfg.nbits}",
        f"ps{train_cfg.pq_sample_size}",
        "s2" if train_cfg.stage2 else "s1",
    ]
    return "__".join(parts)


@dataclass
class DataConfig:
    base_index: str
    target_dir: str
    model_name: Literal["tct_colbert", "tas_b", "star", "adore_star", "e5", "repllama", "dragon", "lion"]
    train_ds: str
    corpus: str
    eval_ds: str
    eval_split: str | None
    test_ds: str
    test_split: str


@dataclass
class TrainingConfig:
    # PQ
    M: int
    nbits: int
    pq_sample_size: int
    # stage 1
    stage1_epochs: int
    stage1_steps: int | None
    stage1_batch_size: int
    lr_encoder: float
    lr_centroid: float
    lambda_mse: float | None
    hardneg_topk: int | None
    valid_every: int
    patience: int
    metric: str
    pairs_cap: int | None
    # stage 2
    stage2: bool
    stage2_epochs: int
    stage2_steps: int | None
    stage2_batch_size: int
    lr_query: float
    lr_centroid_s2: float
    dynamic_negs: int
    # index
    nlist: int
    nprobe: int


def add_data_args(parser: argparse.ArgumentParser):
    p = parser.add_argument_group("Data")
    p.add_argument("--base-index", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--model-name", choices=list(DEFAULT_INIT_BY_NAME), default="tct_colbert")
    p.add_argument("--train-ds", default="msmarco-passage/train")
    p.add_argument("--corpus", default="msmarco-passage/collection",
                   help="ir_datasets corpus providing docid -> text (stage 1 trains on text)")
    p.add_argument("--eval-ds", default="irds:msmarco-passage/dev/small")
    p.add_argument("--eval-split", default=None)
    p.add_argument("--test-ds", default="msmarco_passage")
    p.add_argument("--test-split", default="test-2019,test-2020")


def add_training_args(parser: argparse.ArgumentParser):
    p = parser.add_argument_group("Training")
    p.add_argument("--M", type=int, default=48)
    p.add_argument("--nbits", type=int, default=8)
    p.add_argument("--pq-sample-size", type=int, default=159_744)
    p.add_argument("--pairs-cap", type=int, default=2_000_000)
    p.add_argument("--stage1-epochs", type=int, default=1)
    p.add_argument("--stage1-steps", type=int, default=None)
    p.add_argument("--stage1-batch-size", type=int, default=256)
    p.add_argument("--lr-encoder", type=float, default=5e-6,
                   help="stage 1 encoder LR (paper default; the official script uses 2e-5)")
    p.add_argument("--lr-centroid", type=float, default=2e-4,
                   help="stage 1 centroid LR (paper default; the official script uses 5e-4)")
    p.add_argument("--lambda-mse", type=float, default=None,
                   help="clustering loss weight (default: the paper's value for M; the official script uses 1e-4)")
    p.add_argument("--hardneg-topk", type=int, default=None,
                   help="keep only the top-k non-positive candidates per query as negatives (official dynamic_topk_hard_negative)")
    p.add_argument("--valid-every", type=int, default=500)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--metric", default="nDCG@10", choices=["nDCG@10", "RR@10", "R@1000"])
    p.add_argument("--stage2", action="store_true", default=False,
                   help="also run stage 2 (the paper's optional second stage; marginal gains)")
    p.add_argument("--stage2-epochs", type=int, default=1)
    p.add_argument("--stage2-steps", type=int, default=None)
    p.add_argument("--stage2-batch-size", type=int, default=128)
    p.add_argument("--lr-query", type=float, default=2e-6, help="stage 2 encoder LR (official)")
    p.add_argument("--lr-centroid-s2", type=float, default=2e-5, help="stage 2 centroid LR (official)")
    p.add_argument("--dynamic-negs", type=int, default=10,
                   help="dynamic negatives per query in stage 2 (official uses 200)")
    p.add_argument("--nlist", type=int, default=1, help="IVF clusters for the final index (1 = flat PQ)")
    p.add_argument("--nprobe", type=int, default=10, help="IVF probes for the final index")


def parse_args():
    parser = argparse.ArgumentParser(description="RepCONC CLI")
    add_data_args(parser)
    add_training_args(parser)

    args = parser.parse_args()

    data = DataConfig(
        base_index=args.base_index,
        target_dir=args.target_dir,
        model_name=args.model_name,
        train_ds=args.train_ds,
        corpus=args.corpus,
        eval_ds=args.eval_ds,
        eval_split=args.eval_split,
        test_ds=args.test_ds,
        test_split=args.test_split,
    )
    train = TrainingConfig(
        M=args.M,
        nbits=args.nbits,
        pq_sample_size=args.pq_sample_size,
        pairs_cap=args.pairs_cap,
        stage1_epochs=args.stage1_epochs,
        stage1_steps=args.stage1_steps,
        stage1_batch_size=args.stage1_batch_size,
        lr_encoder=args.lr_encoder,
        lr_centroid=args.lr_centroid,
        lambda_mse=args.lambda_mse,
        hardneg_topk=args.hardneg_topk,
        valid_every=args.valid_every,
        patience=args.patience,
        metric=args.metric,
        stage2=args.stage2,
        stage2_epochs=args.stage2_epochs,
        stage2_steps=args.stage2_steps,
        stage2_batch_size=args.stage2_batch_size,
        lr_query=args.lr_query,
        lr_centroid_s2=args.lr_centroid_s2,
        dynamic_negs=args.dynamic_negs,
        nlist=args.nlist,
        nprobe=args.nprobe,
    )
    return data, train


def _load_corpus_texts(corpus_ds: str, docnos: list[str]) -> dict[str, str]:
    """docno -> text map for the documents in the base index."""
    corpus = ir_datasets.load(corpus_ds)
    wanted = set(docnos)
    texts: dict[str, str] = {}
    for doc_id, doc in corpus.docs_iter():
        if doc_id in wanted:
            texts[doc_id] = doc.text
    missing = len(wanted) - len(texts)
    if missing:
        print(f"WARNING: {missing} of {len(wanted)} index docnos not found in corpus {corpus_ds}")
    return texts


def _triples_iter(queries, docpairs, texts: dict[str, str], pairs_cap: int | None) -> Iterator[dict]:
    """Stage-1 text triples from docpairs + corpus texts."""
    qdict: dict[str, str] = {}
    for query_id, doc in queries:
        qdict[query_id] = doc.text
    count = 0
    for dp in docpairs:
        pos, neg = texts.get(dp.doc_id_a), texts.get(dp.doc_id_b)
        query = qdict.get(dp.query_id)
        if pos is None or neg is None or query is None:
            continue
        yield {'qid': dp.query_id, 'query': query, 'doc_id_a': pos, 'doc_id_b': neg}
        count += 1
        if pairs_cap is not None and count >= pairs_cap:
            break
    if count == 0:
        raise ValueError("no stage-1 triples could be produced "
                         "(check that the corpus docids match the training docpair docids)")


def _pairs_iter(queries, docpairs, pairs_cap: int | None) -> Iterator[dict]:
    """Stage-2 (query, docno) pairs from docpairs."""
    qdict: dict[str, str] = {}
    for query_id, doc in queries:
        qdict[query_id] = doc.text
    count = 0
    for dp in docpairs:
        query = qdict.get(dp.query_id)
        if query is None:
            continue
        yield {'qid': dp.query_id, 'query': query, 'doc_id_a': dp.doc_id_a}
        count += 1
        if pairs_cap is not None and count >= pairs_cap:
            break


def _rebuild_iters(train_ds: str, pairs_cap: int | None):
    """(queries, docpairs) iterators for the training dataset (fresh per use)."""
    ds = ir_datasets.load(train_ds)
    docpairs = ds.docpairs_iter()
    if pairs_cap is not None:
        docpairs = itertools.islice(docpairs, pairs_cap)
    return ds.queries_iter(), docpairs


if __name__ == "__main__":
    data, train = parse_args()

    print("DATA CONFIG:", data)
    print("TRAIN CONFIG:", train)

    target = data.target_dir + "/" + compute_index_name(data, train)
    print("DESTINATION:", target)

    model = instantiate_model(data.model_name)
    print("Instantiated model:", model)

    index = pyterrier_dr.FlexIndex(data.base_index)
    print(f"Index: {data.base_index} has {len(index)} docs")

    docnos = [str(x) for x in index.docnos().fwd]
    print("loading corpus texts ...")
    texts = _load_corpus_texts(data.corpus, docnos)
    print(f"corpus texts loaded: {len(texts)}")

    eval_dataset = pt.get_dataset(data.eval_ds)
    print("eval dataset loaded")

    # reproducibility
    torch.manual_seed(0)

    t = RepCONCTrainer(model, index, M=train.M, nbits=train.nbits)

    stage1_kwargs: dict[str, Any] = dict(
        batch_size=train.stage1_batch_size,
        epochs=train.stage1_epochs,
        total_steps=train.stage1_steps,
        lr_encoder=train.lr_encoder,
        lr_centroid=train.lr_centroid,
        lambda_mse=train.lambda_mse,
        hardneg_topk=train.hardneg_topk,
        valid_every=train.valid_every,
        patience=train.patience,
        metric=train.metric,
    )
    stage2_kwargs: dict[str, Any] = dict(
        batch_size=train.stage2_batch_size,
        epochs=train.stage2_epochs,
        total_steps=train.stage2_steps,
        lr_query=train.lr_query,
        lr_centroid=train.lr_centroid_s2,
        dynamic_negs=train.dynamic_negs,
        valid_every=train.valid_every,
        patience=train.patience,
        metric=train.metric,
    )

    if train.stage2:
        t.fit(
            _triples_iter(*_rebuild_iters(data.train_ds, train.pairs_cap), texts, train.pairs_cap),
            stage2_pairs=_pairs_iter(*_rebuild_iters(data.train_ds, train.pairs_cap), train.pairs_cap),
            pq_sample_size=train.pq_sample_size,
            stage1_kwargs=stage1_kwargs,
            stage2_kwargs=stage2_kwargs,
        )
    else:
        t.fit(
            _triples_iter(*_rebuild_iters(data.train_ds, train.pairs_cap), texts, train.pairs_cap),
            pq_sample_size=train.pq_sample_size,
            stage1_kwargs=stage1_kwargs,
        )

    os.makedirs(target, exist_ok=True)
    t.save_checkpoint(target + "/checkpoint.pt")

    # re-encode the corpus with the trained encoder (official convention)
    missing = [d for d in docnos if d not in texts]
    if missing:
        raise ValueError(f"cannot build the RepCONC index with corpus_text: "
                         f"{len(missing)} index docnos have no corpus text, e.g. {missing[:3]}")
    corpus_texts = [texts[d] for d in docnos]
    newindex = t.repconc_index(target, nlist=train.nlist, nprobe=train.nprobe,
                               corpus_text=corpus_texts)

    oldmodel = instantiate_model(data.model_name)

    p = [
        oldmodel >> index.retriever(),   # type: ignore
        t.biencoder.query_encoder() >> newindex.retriever_pq(),
    ]
    save_dir = target + "/" + 'runs'

    def _do_run(dataset, split, name):
        print(split, name)
        split_savedir = save_dir + "/" + name.replace(":", "_").replace("/", "_")
        os.makedirs(split_savedir, exist_ok=True)
        df = pt.Experiment(
            p,
            dataset.get_topics(split),
            dataset.get_qrels(split),
            eval_metrics=[RR@10, Recall(rel=2)@100, Recall@100, nDCG@10, "mrt"],
            names=["baseline", "RepCONC pq"],
            save_dir=split_savedir,
            baseline=0
        )
        df.to_csv(split_savedir + "/metrics.csv")
        print(df)

    for split in data.test_split.split(","):
        _do_run(pt.get_dataset(data.test_ds), split, split)
    if data.eval_ds is not None:
        _do_run(eval_dataset, data.eval_split, "dev")
