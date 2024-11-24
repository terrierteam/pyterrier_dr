import tempfile
import unittest
import numpy as np
import pandas as pd
import pyterrier_dr
from pyterrier_dr import FlexIndex


class TestFlexIndex(unittest.TestCase):

    def _generate_data(self, count=2000, dim=100):
        def random_unit_vec():
            v = np.random.rand(dim).astype(np.float32)
            return v / np.linalg.norm(v)
        return [
            {'docno': str(i), 'doc_vec': random_unit_vec()}
            for i in range(count)
        ]

    def test_index_typical(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')

            self.assertFalse(index.built())

            dataset = self._generate_data()

            index.index(dataset)

            self.assertTrue(index.built())

            self.assertEqual(len(index), len(dataset))

            stored_dataset = list(index.get_corpus_iter())
            self.assertEqual(len(stored_dataset), len(dataset))
            for a, b in zip(stored_dataset, dataset):
                self.assertEqual(a['docno'], b['docno'])
                self.assertTrue((a['doc_vec'] == b['doc_vec']).all())

    def test_corpus_graph(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data()
            index.index(dataset)
            
            graph = index.corpus_graph(16)
            self.assertEqual(graph.neighbours(4).shape, (16,))

    @unittest.skipIf(not pyterrier_dr.util.faiss_available(), "faiss not available")
    def test_faiss_hnsw_graph(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data()
            index.index(dataset)

            graph = index.faiss_hnsw_graph(16)
            self.assertEqual(graph.neighbours(4).shape, (16,))

    def _test_retr(self, Retr, exact=True, test_smaller=True):
        with self.subTest('basic'):
            with tempfile.TemporaryDirectory() as destdir:
                index = FlexIndex(destdir+'/index')
                dataset = self._generate_data(count=2000)
                index.index(dataset)
                
                retr = Retr(index)
                res = retr(pd.DataFrame([
                    {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
                    {'qid': '1', 'query_vec': dataset[1]['doc_vec']},
                ]))
                self.assertTrue(all(c in res.columns) for c in ['qid', 'docno', 'rank', 'score', 'query_vec'])
                if exact:
                    self.assertEqual(len(res), 2000)
                    self.assertEqual(len(res[res.qid=='0']), 1000)
                    self.assertEqual(len(res[res.qid=='1']), 1000)
                    self.assertEqual(res[(res.qid=='0')&((res['rank']==0))].iloc[0]['docno'], '0')
                    self.assertEqual(res[(res.qid=='1')&((res['rank']==0))].iloc[0]['docno'], '1')
                else:
                    self.assertTrue(len(res) <= 2000)
                    self.assertTrue(len(res[res.qid=='0']) <= 1000)
                    self.assertTrue(len(res[res.qid=='1']) <= 1000)

        with self.subTest('drop_query_vec=True'):
            with tempfile.TemporaryDirectory() as destdir:
                index = FlexIndex(destdir+'/index')
                dataset = self._generate_data(count=2000)
                index.index(dataset)
                
                retr = Retr(index, drop_query_vec=True)
                res = retr(pd.DataFrame([
                    {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
                    {'qid': '1', 'query_vec': dataset[1]['doc_vec']},
                ]))
                self.assertTrue(all(c in res.columns) for c in ['qid', 'docno', 'rank', 'score'])
                self.assertTrue(all(c not in res.columns) for c in ['query_vec'])

        if test_smaller:
            with self.subTest('smaller'):
                with tempfile.TemporaryDirectory() as destdir:
                    index = FlexIndex(destdir+'/index')
                    dataset = self._generate_data(count=100)
                    index.index(dataset)
                    
                    retr = Retr(index)
                    res = retr(pd.DataFrame([
                        {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
                        {'qid': '1', 'query_vec': dataset[1]['doc_vec']},
                    ]))
                    self.assertTrue(all(c in res.columns) for c in ['qid', 'docno', 'rank', 'score'])
                    if exact:
                        self.assertEqual(len(res), 200)
                        self.assertEqual(len(res[res.qid=='0']), 100)
                        self.assertEqual(len(res[res.qid=='1']), 100)
                        self.assertEqual(res[(res.qid=='0')&((res['rank']==0))].iloc[0]['docno'], '0')
                        self.assertEqual(res[(res.qid=='1')&((res['rank']==0))].iloc[0]['docno'], '1')
                    else:
                        self.assertTrue(len(res) <= 200)
                        self.assertTrue(len(res[res.qid=='0']) <= 100)
                        self.assertTrue(len(res[res.qid=='1']) <= 100)

    @unittest.skipIf(not pyterrier_dr.util.faiss_available(), "faiss not available")
    def test_faiss_flat_retriever(self):
        self._test_retr(FlexIndex.faiss_flat_retriever)

    @unittest.skipIf(not pyterrier_dr.util.faiss_available(), "faiss not available")
    def test_faiss_hnsw_retriever(self):
        self._test_retr(FlexIndex.faiss_hnsw_retriever, exact=False)

    @unittest.skipIf(not pyterrier_dr.util.faiss_available(), "faiss not available")
    def test_faiss_ivf_retriever(self):
        self._test_retr(FlexIndex.faiss_ivf_retriever, exact=False)

    @unittest.skipIf(not pyterrier_dr.util.scann_available(), "scann not available")
    def test_scann_retriever(self):
        self._test_retr(FlexIndex.scann_retriever, exact=False)

    def test_np_retriever(self):
        self._test_retr(FlexIndex.np_retriever)

    def test_torch_retriever(self):
        self._test_retr(FlexIndex.torch_retriever)

    @unittest.skipIf(not pyterrier_dr.util.voyager_available(), "voyager not available")
    def test_voyager_retriever(self):
        # Voyager doesn't support requesting more results than are availabe in the index
        # (the "smaller" case), so disable that test case here.
        self._test_retr(FlexIndex.voyager_retriever, exact=False, test_smaller=False)

    def test_np_vec_loader(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data()
            index.index(dataset)
            
            vec_loader = index.np_vec_loader()
            with self.subTest('docid'):
                res = vec_loader(pd.DataFrame({
                    'docid': [5, 1, 100, 198],
                }))
                self.assertTrue(all(c in res.columns) for c in ['docid', 'doc_vec'])
                self.assertEqual(len(res), 4)
                self.assertTrue((res.iloc[0]['doc_vec'] == dataset[5]['doc_vec']).all())
                self.assertTrue((res.iloc[1]['doc_vec'] == dataset[1]['doc_vec']).all())
                self.assertTrue((res.iloc[2]['doc_vec'] == dataset[100]['doc_vec']).all())
                self.assertTrue((res.iloc[3]['doc_vec'] == dataset[198]['doc_vec']).all())
            with self.subTest('docno'):
                res = vec_loader(pd.DataFrame({
                    'docno': ['20', '0', '100', '198'],
                    'query': 'ignored',
                }))
                self.assertTrue(all(c in res.columns) for c in ['docno', 'doc_vec', 'query'])
                self.assertEqual(len(res), 4)
                self.assertTrue((res.iloc[0]['doc_vec'] == dataset[20]['doc_vec']).all())
                self.assertTrue((res.iloc[1]['doc_vec'] == dataset[0]['doc_vec']).all())
                self.assertTrue((res.iloc[2]['doc_vec'] == dataset[100]['doc_vec']).all())
                self.assertTrue((res.iloc[3]['doc_vec'] == dataset[198]['doc_vec']).all())

    def _test_reranker(self, Reranker, expands=False):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=2000)
            index.index(dataset)
            
            retr = Reranker(index)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec'], 'docno': '0', 'score': -100.},
                {'qid': '0', 'query_vec': dataset[0]['doc_vec'], 'docno': '50', 'score': -100.},
                {'qid': '1', 'query_vec': dataset[1]['doc_vec'], 'docno': '100', 'score': -100.},
                {'qid': '1', 'query_vec': dataset[1]['doc_vec'], 'docno': '0', 'score': -100.},
                {'qid': '1', 'query_vec': dataset[1]['doc_vec'], 'docno': '40', 'score': -100.},
            ]))
            self.assertTrue(all(c in res.columns) for c in ['qid', 'docno', 'rank', 'score'])
            if expands:
                self.assertTrue(len(res) >= 5)
            else:
                self.assertEqual(len(res), 5)
            self.assertEqual(res[(res.qid=='0')&((res['rank']==0))].iloc[0]['docno'], '0')
            self.assertEqual(len(res[res.score==-100.]), 0) # all scores re-assigned

    def test_np_scorer(self):
        self._test_reranker(FlexIndex.np_scorer)

    def test_torch_scorer(self):
        self._test_reranker(FlexIndex.torch_scorer)

    def test_pre_ladr(self):
        self._test_reranker(FlexIndex.pre_ladr, expands=True)

    def test_ada_ladr(self):
        self._test_reranker(FlexIndex.ada_ladr, expands=True)

    def test_gar(self):
        self._test_reranker(FlexIndex.gar, expands=True)

    def test_torch_vecs(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data()
            index.index(dataset)
            
            torch_vecs = index.torch_vecs()
            self.assertEqual(torch_vecs.shape, (2000, 100))


if __name__ == '__main__':
    unittest.main()
