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
        t = JPQTrainer(tct, index, M=96, nbits=8, pq_impl='faiss2')

        t.fit(
            merge_queries_into_docpairs(train_dataset.queries_iter(), train_dataset.docpairs_iter()[:2_000_000]), 
            # docid_subset=1_000_000, 
            pq_sample_size=159_744,
            epochs=10,
            batch_size=512,
            eval_queries = pt.get_dataset('msmarco_passage').get_topics('test-2019'),
            eval_qrels = pt.get_dataset('msmarco_passage').get_qrels('test-2019'), valid_every=500,
            jpq_negs=100,
            lambda_rank=True            
        ) 
