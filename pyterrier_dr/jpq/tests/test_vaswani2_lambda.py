import unittest

from ir_measures import RR, Recall, nDCG
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
        t = JPQTrainer(tct, index, pq_impl='faiss2', M=96)

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
            epochs=10, 
            pq_sample_size=10_000, 
            batch_size=512,
            eval_queries=dataset.get_topics(), 
            eval_qrels= dataset.get_qrels(), valid_every=25,
            jpq_negs=100,
            lambda_rank=True
        )

        # newindex = t.jpq_index("/root/nfs/jpq/20_epochs_M96")
        newindex = t.jpq_index("/data/data-nicola/jpq/vaswani_10_epochs_M96")
        
        # t.query_encoder.model.save_pretrained("/root/nfs/jpq/20_epochs_M96", from_pt=True)
        t.query_encoder.model.save_pretrained("/data/data-nicola/jpq/vaswani_10_epochs_M96", from_pt=True)  # type: ignore
 
        oldmodel = pyterrier_dr.TctColBert()
        dataset = pt.get_dataset('vaswani')
        print(pt.Experiment(
            [
                oldmodel >> index.retriever(), # type: ignore
                t.query_encoder >> newindex.retriever_flat()
            ],
            topics = dataset.get_topics(),
            qrels = dataset.get_qrels(),
            eval_metrics=[RR@10, Recall(rel=2)@100, Recall@100, nDCG@10],
            batch_size=10,
            verbose=True
        )) # type: ignore