import unittest
from pyterrier_dr import FlexIndex
import pyterrier_dr, torch, pyterrier as pt


class TestJPQ(unittest.TestCase):
    def test_jpq(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")

        tct = pyterrier_dr.TctColBert(device=device)#device=torch.device("mps"))
        index = FlexIndex("./tests/fixtures/vaswani_tct.flex")
        from pyterrier_dr.jpq import JPQTrainer
        # t = JPQTrainer(tct, index, pq_impl='sklearn', pq_M=4, pq_nbits=4)
        t = JPQTrainer(tct, index, pq_impl='faiss', pq_M=4, pq_nbits=4)
        
        dataset = pt.get_dataset("vaswani")
        doc_pairs = pyterrier_dr.jpq.utils.queries_qrels_to_pairsiter(
            dataset.get_topics(), 
            # vaswani has no negative judgements in qrels, add a random doc from corpus as negative for each query
            pyterrier_dr.jpq.utils.sample_random_negatives( 
                dataset.get_qrels(), 
                1, 
                index.payload(return_dvecs=False)[0].fwd),
            max_neg=1
            )
        
        t.fit(
            doc_pairs, 
            epochs=10, patience=10000, 
            pq_sample_size=500, 
            eval_queries=dataset.get_topics(), 
            eval_qrels= dataset.get_qrels(), valid_every=64
        )