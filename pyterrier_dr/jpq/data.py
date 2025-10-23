from typing import Any
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import pyterrier as pt, pandas as pd
from pyterrier_dr.flex.core import FlexIndex


def get_pq_training_dataset(
        flex_index: FlexIndex,
        docid_subset: list[int] | list[str] | int | None = None, # how many doc vectors to use to train the sub-id embeddings 
) ->tuple:
    """
    Build the (docno, docid) subset used to train PQ sub-embeddings.

    Args:
        flex_index: The dense index providing doc mappings and vectors.
        docid_subset:
            - None or empty: use all documents.
            - int: sample that many random *internal* doc ids from [0, N).
            - Sequence[int]: explicit list of *internal* doc ids.
            - Sequence[str]: explicit list of docnos.
        rng: Optional NumPy Generator for reproducible sampling.

    Returns:
        (selected_docnos, selected_docids, docnos2pos)
        where docnos are strings and docids are internal integer ids.
    """
    print(f"Ingesting docno mapping from index ")
    doc_map = flex_index.payload()[0]
    N = len(flex_index)

    if not docid_subset: # use all documents if not used or empty list
        docid_subset = list(range(N))
    if isinstance(docid_subset, int): # use a random subset of given size
        if docid_subset > N:
            raise ValueError(f"docid_subset {docid_subset} > total docs {N}")
        selected_docids = np.random.choice(N, size=docid_subset, replace=False) # type: ignore
        selected_docids = np.sort(selected_docids)
        selected_docnos = doc_map.fwd[selected_docids]
        print(f"[SUBSET] using {len(selected_docnos)} random docs from index")        
    elif isinstance(docid_subset, list):  
        if isinstance(docid_subset[0], int): # use the provided list of int docid
            selected_docids = np.sort(docid_subset)
            selected_docnos = doc_map.fwd[docid_subset]
            print(f"[SUBSET] using {len(selected_docnos)} provided docs from index")
        elif isinstance(docid_subset[0], str): # use the provided list of str docnos
            selected_docnos = docid_subset
            selected_docids = doc_map.inv[selected_docnos] # do we need this?
            selected_docids = np.sort(selected_docids)
            print(f"[SUBSET] using {len(selected_docnos)} provided docs from index")
        else:
            raise ValueError(f"list on integers or strings must be provided")
        
    selected_docnos = list(selected_docnos) # do we need this conversion?    
    #print(f"-----{selected_docnos}")
    docnos2pos = {docno: i for i, docno in enumerate(selected_docnos)} # map from docno to position in selected_docnos (for aligning with codes_sel)

    return selected_docnos, selected_docids, docnos2pos

def get_dataloader(
        ds : Dataset,
        batch_size: int
) -> Dataset: 
    
    def collate(batch) -> dict[str, Any]:
        return {
            'query_text': [b['query_text'] for b in batch],
            'pos_codes': torch.stack([b['pos_codes'] for b in batch]),
            'neg_codes': torch.stack([b['neg_codes'] for b in batch]),
        }
    print(f"[DATA] Collating")
    return DataLoader(
        ds,  # type: ignore
        batch_size=batch_size, 
        collate_fn=collate)

def get_dataset(
    docpairs: list[dict[str, Any]], # training docpairs to use
    docnos: list[str], # documents we use during training
    codes: np.ndarray, # PQ codes for used documents
    docno2pos, # map from code position to docno
) -> DataLoader:       

    # we discard training pair where pos or neg documents were not used during PQ training
    docnos_set = set(docnos)
    def filter_in_sel(docpair) -> bool:
        return docpair['doc_id_a'] in docnos_set and docpair['doc_id_b'] in docnos_set

    # Convert codes once to torch (long) so mapped samples are tensors already
    codes_t = torch.as_tensor(codes, dtype=torch.long)

    # we convert (query, pos_docno, neg_docno) into (query, pos_pq_codes, neg_pq_codes)
    def queries_and_codes(docpair: dict[str, Any]) -> dict[str, Any]:
        return {
            'query_text': docpair["query"],
            'pos_docno': docpair["doc_id_a"],
            'pos_codes': codes_t[docno2pos[docpair["doc_id_a"]]],
            'neg_docno' : docpair["doc_id_b"],
            'neg_codes': codes_t[docno2pos[docpair["doc_id_b"]]],
        }

    print("[DATA] Preparing training data")
    docpairs = list(docpairs) # in case docpairs is an iterator...

    ds = Dataset.from_list(docpairs)
    ds = ds.filter(filter_in_sel).shuffle()
    if not len(ds):
        raise ValueError(f"After filtering {len(docpairs)} in the training dataset down to the sampled {len(docnos_set)}, we have 0 pairs left. \n"
                         "Try increasing size of training dataset, or value of docid_subset")
    print(f"[DATA] After filtering, we have {len(ds)} remaining from {len(docpairs)} pairs")
    ds = ds.map(
        queries_and_codes,
        remove_columns=[c for c in ds.column_names if c not in ("query_text", 'pos_docno', "pos_codes", 'neg_docno', "neg_codes")],
    )
    ds.set_format(type="torch", columns=["query_text", "pos_codes", "neg_codes"])
    return ds
    
def add_jpq_negs(ds : Dataset, top_k : int, retr_pipe : pt.Transformer, codes: np.ndarray) -> Dataset:

    retr_pipe = (retr_pipe % top_k).compile()
    codes_t = torch.as_tensor(codes, dtype=torch.long)
    def _add_neg(docpair: dict[str, Any]) -> dict[str, Any]:
        res : pd.DataFrame = retr_pipe.search(docpair['query_text'])
        res = res[~res["docno"].isin([docpair['pos_docno'], docpair['neg_docno']])]
        codes_negs = codes_t[res["docid"].to_list()]
        docpair["neg_jpq_codes"] = codes_negs
        return docpair

    print("[DATA] Adding %d top_k negs to %d training samples" % (top_k, len(ds)))
    ds = ds.map(
        _add_neg,
        remove_columns=[c for c in ds.column_names if c not in ("query_text", "pos_codes", "neg_codes", "neg_jpq_codes")]
    )
    ds.set_format(type="torch", columns=["query_text", "pos_codes", "neg_codes", "neg_jpq_codes"])
    return ds