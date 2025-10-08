import unittest
import pyterrier as pt
import pyterrier_dr
import torch

class TestJPQ(unittest.TestCase):
    def test_jpq(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        print("loading model")
        tct = pyterrier_dr.TctColBert(device=device)

        print("loading index")
        index = pyterrier_dr.FlexIndex("./msmarco-passage.tct-hnp.flex")
        print(f"index has has {len(index)} docs")

        import ir_datasets
        train_dataset = ir_datasets.load("msmarco-passage/train")
        print("training dataset loaded")

        from pyterrier_dr.jpq.utils import merge_queries_into_docpairs
        from pyterrier_dr.jpq import JPQTrainer
        t = JPQTrainer(tct, index, pq_M=16, pq_nbits=8, pq_impl='faiss')

        t.fit(
            merge_queries_into_docpairs(train_dataset.queries_iter(), train_dataset.docpairs_iter()[:100000]), 
            docid_subset=100_000, 
            pq_sample_size=50_000,
            code_batch_size=5_000,
            valid_every=100,
            patience=20_000, epochs=100_000,
            eval_queries = pt.get_dataset('msmarco_passage').get_topics('test-2019'),
            eval_qrels = pt.get_dataset('msmarco_passage').get_qrels('test-2019'),
        ) 