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

        tct = pyterrier_dr.TctColBert(device=device)
        index = FlexIndex("./tests/fixtures/vaswani_tct.flex")

        from pyterrier_dr.jpq import JPQTrainer
        t = JPQTrainer(tct, index, pq_impl='faiss', M=4, nbits=4)

        dataset = pt.get_dataset("vaswani")
        from pyterrier_dr.jpq import utils
        doc_pairs = utils.queries_qrels_to_pairsiter(
            dataset.get_topics(), 
            # vaswani has no negative judgements in qrels, add a random doc from corpus as negative for each query
            utils.sample_random_negatives( 
                dataset.get_qrels(), 
                1, 
                index.payload(return_dvecs=False)[0].fwd),
            max_neg=1
            )
        
        t.fit(
            doc_pairs, 
            epochs=100, 
            pq_sample_size=750, 
            eval_queries=dataset.get_topics(), 
            eval_qrels= dataset.get_qrels(), valid_every=25
        )