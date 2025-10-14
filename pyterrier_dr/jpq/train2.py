from typing import Literal
import numpy as np

from pyterrier_dr import FlexIndex
from pyterrier_dr.jpq.utils import l2_normalize_np, timer

from .pq import ProductQuantizer, ProductQuantizerFAISS, ProductQuantizerSKLearn


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
        (selected_docnos, selected_docids)
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
        selected_docnos = doc_map.fwd[selected_docids]
        print(f"[SUBSET] using {len(selected_docnos)} random docs from index")        
    elif isinstance(docid_subset, list):  
        if isinstance(docid_subset[0], int): # use the provided list of int docid
            selected_docids = docid_subset
            selected_docnos = doc_map.rev(docid_subset)
            print(f"[SUBSET] using {len(selected_docnos)} provided docs from index")
        elif isinstance(docid_subset[0], int): # use the provided list of str docnos
            selected_docnos = doc_map.fwd[docid_subset]
            print(f"[SUBSET] using {len(selected_docnos)} provided docs from index")
        else:
            raise ValueError(f"list on integers or strings must be provided")
        
    selected_docnos = list(selected_docnos) # do we need this conversion?    
    selected_docids = doc_map.inv[selected_docnos] # do we need this?
    
    return selected_docnos, selected_docids


def compute_PQ(
    M: int, # number of subquantizers (splits of the vector)
    n_bits: int, # number of centroids per subquantizer
    sample_size: int, # how many doc vectors from docids to use to train PQ centroids
    batch_size: int, # how many doc vectors from sample to process in a batch when computing PQ codes
    docids: np.ndarray, # which document indices (into full vecs_mem) to compute codes for
    vecs: np.ndarray, # vector store
    pq_impl: Literal["faiss", "sklearn"] = "faiss",
    ) -> tuple[np.ndarray, np.ndarray]:

    if pq_impl == 'faiss':
        pq_class = ProductQuantizerFAISS
    elif pq_impl == 'sklearn':
        pq_class = ProductQuantizerSKLearn
    else:
        raise ValueError(f"Unknown pq_impl {pq_impl}, must be 'faiss' or 'sklearn'")

    pq = pq_class(M, Ks=2**n_bits)

    # train PQ on a random sample of the selected docs
    sample_size = min(sample_size, len(docids))
    print("[PQ] training on %d documents..." % sample_size)
    sample_docids = np.random.choice(docids, size=sample_size, replace=False) # type: ignore
    with timer(f"PQ / train (samples={len(sample_docids):,})"):
        xb = l2_normalize_np(vecs[sample_docids])
        pq.fit(xb)

    print("[PQ] computing codes for %d selected docs in chunks of %d..." % (len(docids), batch_size))
    codes = np.empty((len(docids), M), dtype=np.uint8) # not sure this is ok if we return sklearn codes
    with timer("PQ / compute codes (selected)"):
        codes = pq.encode_batch(l2_normalize_np(vecs[docids]), batch_size)
    
    # TODO: we should check how the average/min/max codes are observed in sel_indices
    # give that sel_indices is a random sample of sel_indices, it should be fairly uniform
    # as a codes are centroids in the vector space defined in the small sample_size set, which
    # is a subset of the sel_indices set, it should be ok.
    return codes, pq.get_centroids() # type: ignore












def get_dataloader(
    docpairs, 
    docnos,
):       

    docnos_set = set(docnos)
    def filter_in_sel(docpair) -> bool:
        return docpair['doc_id_a'] in docnos_set and docpair['doc_id_b'] in docnos_set

    def queries_and_codes(x):# -> dict[str, Any]:
        return {
            'query_text': x["query"],
            'pos_codes': codes_sel[sel_inv[x["doc_id_a"]]],
            'neg_codes': codes_sel[sel_inv[x["doc_id_b"]]],
        }


    selected_doc_ids = set(selected_doc_ids)
    # bring codes to torch
    codes_sel = torch.from_numpy(codes_sel).long()

    # making a list smells bad
    dataset = Dataset.from_list([x for x in training_docpairs])


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