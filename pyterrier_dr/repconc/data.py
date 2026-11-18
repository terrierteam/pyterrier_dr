"""Training data contract for RepCONC (text triples).

RepCONC training is a different paradigm from JPQ: rather than document-id
pairs (with document vectors looked up from an index), stage one takes *text
triples* in which the documents are encoded on the fly (so that gradients can
flow through the document encoder, as in the paper)::

    {"qid"?: str, "query": str, "doc_id_a": <positive document text>,
     "doc_id_b": <negative document text> | [<negative text>, ...]}

Stage two takes ``(query, docno)`` pairs; the positive is a document *docno*
of the index (its code is fixed, paper §3.6.2) and negatives are retrieved
dynamically at each step from the current PQ index.

The field names (``query``, ``doc_id_a``, ``doc_id_b``) follow the JPQ
convention; in stage one the ``doc_id_*`` values are document *texts*, not
identifiers.
"""

from typing import Any, Iterable, List, Union

import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ["RepCONCDataset", "normalize_stage1", "normalize_stage2", "get_dataloader"]

_REQUIRED_STAGE1 = ("query", "doc_id_a")
_REQUIRED_STAGE2 = ("query", "doc_id_a")


def _check_item(item: Any, stage: int) -> dict:
    if not isinstance(item, dict):
        raise ValueError(f"training item must be a dict, got {type(item)}: {item!r}")
    for field in _REQUIRED_STAGE1 if stage == 1 else _REQUIRED_STAGE2:
        if field not in item:
            raise ValueError(f"stage {stage} training item missing field {field!r}: {item!r}")
    if not isinstance(item["query"], str) or not item["query"]:
        raise ValueError(f"stage {stage} training item 'query' must be a non-empty string: {item!r}")
    if stage == 1:
        pos = item["doc_id_a"]
        if not isinstance(pos, str) or not pos:
            raise ValueError(f"stage 1 'doc_id_a' (positive text) must be a non-empty string: {item!r}")
        negs = item.get("doc_id_b", [])
        if isinstance(negs, str):
            negs = [negs]
        elif negs is None:
            negs = []
        if not isinstance(negs, (list, tuple)):
            raise ValueError(f"stage 1 'doc_id_b' must be a string or a list of strings: {item!r}")
        for neg in negs:
            if not isinstance(neg, str) or not neg:
                raise ValueError(f"stage 1 'doc_id_b' entries must be non-empty strings: {item!r}")
        return {"qid": item.get("qid"), "query": item["query"],
                "pos": pos, "negs": list(negs)}
    # stage 2: doc_id_a is a docno
    if not isinstance(item["doc_id_a"], (str, int)):
        raise ValueError(f"stage 2 'doc_id_a' (docno) must be a string or int: {item!r}")
    return {"qid": item.get("qid"), "query": item["query"], "docno": item["doc_id_a"]}


def normalize_stage1(triples: Iterable[dict]) -> List[dict]:
    """Validate and normalise stage-one text triples.

    Returns a list of ``{"qid", "query", "pos": str, "negs": list[str]}``.
    """
    return [_check_item(t, 1) for t in triples]


def normalize_stage2(pairs: Iterable[dict]) -> List[dict]:
    """Validate and normalise stage-two (query, docno) pairs.

    Returns a list of ``{"qid", "query", "docno"}``.
    """
    return [_check_item(t, 2) for t in pairs]


class RepCONCDataset(Dataset):
    """A dataset of normalised training items (see :func:`normalize_stage1`
    and :func:`normalize_stage2`)."""

    def __init__(self, items: List[dict]):
        if not items:
            raise ValueError("training dataset is empty")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def get_dataloader(items: List[dict], batch_size: int, *,
                   shuffle: bool = True, seed: int | None = 42) -> DataLoader:
    """A dataloader yielding batches (lists) of training items.

    Each batch is a plain Python list of item dicts; document encoding is
    performed by the trainer (the BiEncoder encodes in its own internal
    batches).
    """
    ds = RepCONCDataset(items)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=lambda x: x, drop_last=False,
                      generator=generator)
