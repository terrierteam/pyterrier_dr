import unittest
import pyterrier as pt
import pyterrier_dr
import torch

from pyterrier.measures import RR, Recall, nDCG


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
        tct = pyterrier_dr.TctColBert.hnp(device=device)

        print("loading index")
        index = pyterrier_dr.FlexIndex("./msmarco-passage.tct-hnp.flex")
        print(f"index has has {len(index)} docs")

        import ir_datasets
        train_dataset = ir_datasets.load("msmarco-passage/train")
        print("training dataset loaded")

        from pyterrier_dr.jpq.utils import merge_queries_into_docpairs
        from pyterrier_dr.jpq import JPQTrainer
        t = JPQTrainer(tct, index, M=96, nbits=8, pq_impl='faiss2opq')

        t.fit(
            merge_queries_into_docpairs(train_dataset.queries_iter(), train_dataset.docpairs_iter()[:2_000_000]), 
            # docid_subset=1_000_000,
            docid_subset=None,
            pq_sample_size=159_744,
            valid_every=2000,
            # epochs=10,
            eval_queries = pt.get_dataset('msmarco_passage').get_topics('test-2019'),
            eval_qrels = pt.get_dataset('msmarco_passage').get_qrels('test-2019'),
            # max_steps_per_epoch=20000,
            in_batch=True,
            lambda_rank=True,
            jpq_negs=200
        ) 

        TARGET="/data/data-nicola/jpq/10_epochs_M96_ibn_10jpqnegs_lambdarank"
        newindex = t.jpq_index(TARGET)
        t.query_encoder.model.save_pretrained(TARGET, from_pt=True)
 
        oldmodel = pyterrier_dr.TctColBert.hnp()
        test_dataset = pt.get_dataset('msmarco_passage')

        p = [
            oldmodel >> index.retriever(), # type: ignore
            t.query_encoder >> newindex.retriever_flat(),
            t.query_encoder >> newindex.retriever_pq()
        ]
        print(pt.Experiment(
            p,
            test_dataset.get_topics('test-2019'),
            test_dataset.get_qrels('test-2019'),
            eval_metrics=[RR@10, Recall(rel=2)@100, Recall@100, nDCG@10, "mrt"],
            names=["baseline", "JPQ flat", "JPQ pq"]
        ))
 
