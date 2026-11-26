import unittest
import tempfile
import numpy as np
import pandas as pd
from pyterrier_dr import FlexIndex
import pyterrier_dr, torch, pyterrier as pt
from pyterrier.measures import *
import torch

class TestJPQ(unittest.TestCase):

    def test_vec_loader(self):
        from pyterrier_dr.jpq import JPQIndex

        centroids = np.array([
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[10, 20], [30, 40], [50, 60], [70, 80]],
        ], dtype=np.float32)
        codes = np.array([[0, 3], [2, 1], [1, 0]], dtype=np.uint8)
        rotation = np.array([
            [0, -1, 0, 0], [1, 0, 0, 0],
            [0, 0, 0, -1], [0, 0, 1, 0],
        ], dtype=np.float32)
        reconstructed = np.array([[1, 2, 70, 80], [5, 6, 30, 40], [3, 4, 10, 20]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as destdir:
            for opq in (None, rotation):
                with self.subTest(opq=opq is not None):
                    path = destdir + ('/opq' if opq is not None else '/pq')
                    index = JPQIndex.build(path, ['a', 'b', 'c'], codes, centroids, opq=opq)
                    loader = index.vec_loader()
                    expected = reconstructed if opq is None else reconstructed @ opq.T

                    for column, values in [('docid', [2, 0, 2]), ('docno', ['c', 'a', 'c'])]:
                        inp = pd.DataFrame({column: values, 'score': [0.1, 0.2, 0.3]}, index=[4, 1, 4])
                        result = loader(inp)
                        self.assertEqual(result.index.tolist(), [4, 1, 4])
                        self.assertEqual(result['score'].tolist(), [0.1, 0.2, 0.3])
                        np.testing.assert_array_equal(np.stack(result['codes']), codes[[2, 0, 2]])
                        np.testing.assert_array_equal(np.stack(result['doc_vec']), expected[[2, 0, 2]])

                    result = loader(pd.DataFrame({'docid': [1], 'docno': ['a']}))
                    np.testing.assert_array_equal(result.iloc[0]['codes'], codes[1])
                    for column in ('docid', 'docno'):
                        inp = pd.DataFrame({column: pd.Series(dtype=object), 'score': pd.Series(dtype=float)})
                        empty = loader(inp)
                        pt.validate.columns(empty, includes=[column, 'score', 'doc_vec', 'codes'])
                        self.assertEqual(list(empty.columns), [column, 'score', 'doc_vec', 'codes'])
                        pd.testing.assert_frame_equal(empty[[column, 'score']], inp)
                        self.assertTrue(empty.empty)
                    with self.assertRaises(pt.validate.InputValidationError):
                        loader(pd.DataFrame({'score': [1]}))
                    with self.assertRaises(pt.validate.InputValidationError):
                        loader(pd.DataFrame({'score': pd.Series(dtype=float)}))
                    self.assertEqual(loader, JPQIndex(path).vec_loader())
                    self.assertEqual(hash(loader), hash(JPQIndex(path).vec_loader()))

            empty_index = JPQIndex.build(
                destdir + '/empty', [], np.empty((0, 2), dtype=np.uint8), centroids, opq=None,
            )
            for column in ('docid', 'docno'):
                empty = empty_index.vec_loader()(pd.DataFrame({column: pd.Series(dtype=object)}))
                pt.validate.columns(empty, includes=[column, 'doc_vec', 'codes'])
                self.assertTrue(empty.empty)

    @unittest.skipIf(not torch.cuda.is_available(), "No CUDA")
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
        #t = JPQTrainer(tct, index, pq_impl='sklearn', M=4, nbits=4)
        t = JPQTrainer(tct, index, pq_impl='faiss2opq', M=96, nbits=7)
        
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
        
        doc_pairs = list(doc_pairs)[:256] # limit for test speed
        t.fit(
            doc_pairs, 
            total_steps=1_000_000, 
            patience=2, 
            pq_sample_size=10000, 
            eval_queries=dataset.get_topics(), 
            eval_qrels= dataset.get_qrels(), 
            valid_every=64,
            jpq_negs = 10,
            lambda_rank=True,
            in_batch=True
        )
        from tempfile import mkdtemp
        import os, shutil
        dest = mkdtemp()
        os.rmdir(dest)
        jpqindex = t.jpq_index(dest)
        oldmodel = pyterrier_dr.TctColBert()

        p = [
            (tct >> jpqindex.retriever_flat()),
            (tct >> jpqindex.retriever_prune()),
            (tct >> jpqindex.retriever_pq())
        ]
        for r in p:
            res = r.search("chemical reactions")
            print(res.head())
            print(len(res))
        print(pt.Experiment(
            [oldmodel >> index] + p,
            dataset.get_topics(),
            dataset.get_qrels(),
            eval_metrics=[RR@100, MAP, nDCG@10, "mrt"],
            names=["baseline", "JPQ flat", "JPQ prune", "JPQ PQ"]
        ))
        del(jpqindex)
        shutil.rmtree(dest)
