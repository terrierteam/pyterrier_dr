import logging

from typing import Any
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import pyterrier as pt, pandas as pd
from pyterrier_dr.flex.core import FlexIndex

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

def get_pq_training_dataset(
        flex_index: FlexIndex,
        docid_subset: list[int] | list[str] | int | None = None, # how many doc vectors to use to train the sub-id embeddings 
) -> tuple:
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
    logger.info(f"Ingesting docno mapping from index ")
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
        logger.info(f"[SUBSET] using {len(selected_docnos)} random docs from index")        
    elif isinstance(docid_subset, list):  
        if isinstance(docid_subset[0], int): # use the provided list of int docid
            selected_docids = np.sort(docid_subset)
            selected_docnos = doc_map.fwd[docid_subset]
            logger.info(f"[SUBSET] using {len(selected_docnos)} provided docs from index")
        elif isinstance(docid_subset[0], str): # use the provided list of str docnos
            selected_docnos = docid_subset
            selected_docids = doc_map.inv[selected_docnos] # do we need this?
            selected_docids = np.sort(selected_docids)
            logger.info(f"[SUBSET] using {len(selected_docnos)} provided docs from index")
        else:
            raise ValueError(f"list on integers or strings must be provided")
        
    selected_docnos = list(selected_docnos) # do we need this conversion?    
    docnos2pos = {docno: i for i, docno in enumerate(selected_docnos)} # map from docno to position in selected_docnos (for aligning with codes_sel)

    return selected_docnos, selected_docids, docnos2pos


def get_dataloader(
        ds: Dataset,
        batch_size: int
) -> DataLoader: 
    
    def collate(batch) -> dict[str, Any]:
        rtr = {
            'query_text': [b['query_text'] for b in batch],
            'pos_codes': torch.stack([b['pos_codes'] for b in batch]),
            'neg_codes': torch.stack([b['neg_codes'] for b in batch])
        }
        if 'neg_jpq_codes' in batch[0]:
            rtr['neg_jpq_codes'] = torch.stack([b['neg_jpq_codes'] for b in batch])
            rtr['neg_jpq_ranks'] = torch.stack([b['neg_jpq_ranks'] for b in batch])
            rtr['pos_ranks' ] = torch.stack([b['pos_ranks'] for b in batch])
            rtr['neg_ranks'] = torch.stack([b['neg_ranks'] for b in batch])
        return rtr
    # logger.info(f"[DATA] Collating")

    return DataLoader(
        ds, # type: ignore
        batch_size=batch_size, 
        collate_fn=collate, shuffle=False)


def get_dataset(
    docpairs: list[dict[str, Any]], # training docpairs to use
    docnos: list[str], # documents we use during training
    codes: np.ndarray, # PQ codes for used documents
    docno2pos, # map from code position to docno
    *,
    shuffle: bool = False,
    seed: int | None = None,
) -> Dataset:       

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

    logger.info("[DATA] Preparing training data")
    docpairs = list(docpairs) # in case docpairs is an iterator...

    ds = Dataset.from_list(docpairs)
    ds = ds.filter(filter_in_sel)
    if not len(ds):
        raise ValueError(f"After filtering {len(docpairs)} in the training dataset down to the sampled {len(docnos_set)}, we have 0 pairs left. \n"
                         "Try increasing size of training dataset, or value of docid_subset")
    logger.info(f"[DATA] After filtering, we have {len(ds)} remaining from {len(docpairs)} pairs")

    if shuffle:
        ds = ds.shuffle(seed=seed)

    ds = ds.map(
        queries_and_codes,
        remove_columns=[c for c in ds.column_names if c not in ("query_text", 'pos_docno', "pos_codes", 'neg_docno', "neg_codes")],
    )
    ds.set_format(type="torch", columns=["query_text", "pos_codes", "neg_codes"])
    return ds


def add_jpq_negs(
    ds: Dataset, 
    top_k: int,
    retr_pipe: pt.Transformer,
    codes: np.ndarray,
    cache : bool = False
) -> Dataset:
    retr_pipe = (retr_pipe % (top_k + 2)).compile() # +2 to account for pos and neg docs already in the index
    
    if cache:
        # many queries will be repeated due to the nature of some pairs datasets
        # (e.g. if they are instantiated from a qrels file), so cache results to speed up
        from pyterrier_caching import RetrieverCache
        retr_pipe = RetrieverCache(None, retr_pipe, on='query')
    
    codes_t = torch.as_tensor(codes, dtype=torch.long)

    def _add_neg_batches(docpairs: dict[str, list[Any]]) -> dict[str, list[Any]]:
        queries = pd.DataFrame([  {"qid" : f"q{i}", "query": qtext} for i, qtext in enumerate(docpairs['query_text'])])
        res : pd.DataFrame = retr_pipe.search(queries)
        docpairs = docpairs.copy()
        docpairs['neg_jpq_codes'] = []
        docpairs['neg_jpq_ranks'] = []
        docpairs['pos_ranks'] = []
        docpairs['neg_ranks'] = []

        res_grouped = dict(tuple(res.groupby("qid")))
        for i, _ in enumerate(docpairs["query_text"]):
            res_i = res_grouped.get(f"q{i}", pd.DataFrame())
            res_i = res_i[~res_i["docno"].isin([docpairs['pos_docno'][i], docpairs['neg_docno'][i]])]
            res_i = res_i.head(top_k) # take top_k only
            codes_negs = codes_t[res_i["docid"].to_list()]
            rank_negs = res_i["rank"].to_list()
            docpairs["neg_jpq_codes"].append(codes_negs)
            docpairs["neg_jpq_ranks"].append(rank_negs)
            for t in ["pos", "neg"]:
                t_res = res_i["docno"] == docpairs['pos_docno'][i]
                if t_res.any():
                    docpairs[f"{t}_ranks"].append( res_i[t_res]["rank"].values[0] )
                else:
                    docpairs[f"{t}_ranks"].append(100) # a deep enough rank
        return docpairs

    logger.info("[DATA] Adding %d top_k negs to %d training examples" % (top_k, len(ds)))

    ds = ds.map(
        _add_neg_batches, batched=True, batch_size=32,
        remove_columns=[c for c in ds.column_names if c not in ("query_text", "pos_codes", "neg_codes", "neg_jpq_codes", "pos_ranks", "neg_ranks", "neg_jpq_ranks")]
    )
    ds.set_format(type="torch", columns=["query_text", "pos_codes", "neg_codes", "neg_jpq_codes", "pos_ranks", "neg_ranks", "neg_jpq_ranks"])
    
    return ds