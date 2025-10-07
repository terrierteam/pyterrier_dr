from contextlib import contextmanager
import time
from typing import Iterator, Tuple, Any, Dict
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

# def merge_queries_into_docpairs(queries: pd.DataFrame, docpairs : Iterator[Any]):
#     queries = {e.query_id : e.text for e in queries}
#     for e in docpairs:
#         e = e._asdict()
#         e['query'] = queries[e['query_id']]
#         yield e

class _MergeQueriesIterator:
    def __init__(self, queries: pd.DataFrame, docpairs: Iterator[Any]):
        # Build lookup dict for queries
        self.queries = {e.query_id: e.text for e in queries}
        self.docpairs = iter(docpairs)

    def __iter__(self):
        return self

    def __next__(self):
        e = next(self.docpairs)  # raises StopIteration automatically when done
        e = e._asdict()
        e['query'] = self.queries[e['query_id']]
        return e

def merge_queries_into_docpairs(queries: pd.DataFrame | Iterator[Any], docpairs: Iterator[Any]) -> Iterator[dict]:
    """Return an iterator that merges queries into docpairs."""
    if not isinstance(queries, pd.DataFrame):
        queries = pd.DataFrame(queries)
    return _MergeQueriesIterator(queries, docpairs)

def queries_qrels_to_pairsiter(queries: pd.DataFrame, qrels: pd.DataFrame, max_neg=None) -> Iterator[Tuple[str, str, str, str]]:
    """
    Given a set of queries and qrels, yield (qid, querytext, posdocno, negdocno) tuples
    suitable for training a bi-encoder with pairwise loss.

    Note that this does not ensure that the negative document is not also a positive document.
    """
    qdict = dict(zip(queries['qid'], queries['query']))
    qrels_grouped = qrels.groupby('qid')
    for qid, group in qrels_grouped:

        pos_docs = group[group['label'] > 0]['docno'].tolist()
        neg_docs = group[group['label'] <= 0]['docno'].tolist() #.sample(frac=1)
        if len(pos_docs) == 0 or len(neg_docs) == 0:
            continue
        query_text = qdict[qid]
        for pos_doc in pos_docs:
            for count_neg, neg_doc in enumerate(neg_docs):
                yield (qid, query_text, pos_doc, neg_doc)
                if max_neg is not None and count_neg + 1 >= max_neg:
                    break

class NullWanDBRun:
    def log(self, *a, **kw): pass
    def watch(self, *a, **kw): pass
    def finish(self): pass