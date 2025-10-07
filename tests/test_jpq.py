import unittest
import tempfile
import numpy as np
import pandas as pd
from pyterrier_dr import FlexIndex
import pyterrier_dr, torch, pyterrier as pt


class TestJPQ(unittest.TestCase):
    def test_jpq(self):
        tct = pyterrier_dr.TctColBert(device=torch.device("mps"))
        index = FlexIndex("./tests/fixtures/vaswani_tct.flex")
        from pyterrier_dr.jpq import JPQTrainer
        t = JPQTrainer(tct, index, pq_impl='sklearn', pq_M=4, pq_nbits=4)
        
        dataset = pt.get_dataset("vaswani")
        doc_pairs = pyterrier_dr.jpq.utils.queries_qrels_to_pairsiter(dataset.get_topics(), dataset.get_qrels())
        t.fit(
            doc_pairs, 
            epochs=1, 
            pq_sample_size=200, 
            eval_queries=dataset.get_topics(), 
            eval_qrels=dataset.get_qrels())