from contextlib import contextmanager
import time
from typing import Generator, Tuple
import numpy as np
import os
import pandas as pd

@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[TIMER] {name}: {dt/60:.2f} min ({dt:.1f} s)")

def l2_normalize_np(arr: np.ndarray) -> np.ndarray:
    X = np.array(arr, dtype=np.float32, copy=True)
    X /= (np.linalg.norm(X, axis=1, keepdims=True).astype(np.float32) + 1e-12)
    return X

def bytes_to_gb(nbytes: int) -> float:
    return nbytes / (1024**3)

def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total

def queries_qrels_to_pairsiter(queries: pd.DataFrame, qrels: pd.DataFrame) -> Generator[Tuple[str, str, str, str]]:
    """
    Given a set of queries and qrels, yield (queryid, querytext, posdocid, negdocid) tuples
    suitable for training a bi-encoder with pairwise loss.

    Note that this does not ensure that the negative document is not also a positive document.
    """
    qdict = dict(zip(queries['qid'], queries['query']))
    qrels_grouped = qrels.groupby('qid')
    for qid, group in qrels_grouped:

        pos_docs = group[group['label'] > 0]['docno'].tolist()
        neg_docs = group[group['label'] <= 0]['docno'].tolist()
        if len(pos_docs) == 0 or len(neg_docs) == 0:
            continue
        query_text = qdict[qid]
        for pos_doc in pos_docs:
            for neg_doc in neg_docs:
                yield (qid, query_text, pos_doc, neg_doc)