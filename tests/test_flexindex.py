import tempfile
import unittest
import numpy as np
import pandas as pd
import pyterrier as pt
import pyterrier_dr
from pyterrier_dr import FlexIndex


class TestFlexIndex(unittest.TestCase):

    def _generate_data(self, count=2000, dim=100):
        return [
            {'docno': str(i), 'doc_vec': np.random.rand(dim).astype(np.float32)}
            for i in range(count)
        ]

    def test_index_typical(self):
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
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
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir+'/index')
        dataset = self._generate_data()
        index.index(dataset)
        
        graph = index.corpus_graph(16)
        self.assertEqual(graph.neighbours(4).shape, (16,))

    @unittest.skipIf(not pyterrier_dr.util.package_available('faiss'), "faiss not available")
    def test_faiss_hnsw_graph(self):
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir+'/index')
        dataset = self._generate_data()
        index.index(dataset)

        graph = index.faiss_hnsw_graph(16)
        self.assertEqual(graph.neighbours(4).shape, (16,))

    def _test_retr(self, Retr, exact=True):
        with self.subTest('basic'):
            destdir = tempfile.mkdtemp()
            self.test_dirs.append(destdir)
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=2000)
            index.index(dataset)
            
            retr = Retr(index)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
                {'qid': '1', 'query_vec': dataset[1]['doc_vec']},
            ]))
            self.assertTrue(all(c in res.columns) for c in ['qid', 'docno', 'rank', 'score'])
            if exact:
                self.assertEqual(len(res), 2000)
                self.assertEqual(len(res[res.qid=='0']), 1000)
                self.assertEqual(len(res[res.qid=='1']), 1000)
                self.assertEqual(res[(res.qid=='0')&((res['rank']==1))].iloc[0]['docno'], '0')
                self.assertEqual(res[(res.qid=='1')&((res['rank']==1))].iloc[0]['docno'], '1')
            else:
                self.assertTrue(len(res) <= 2000)
                self.assertTrue(len(res[res.qid=='0']) <= 1000)
                self.assertTrue(len(res[res.qid=='1']) <= 1000)

        with self.subTest('smaller'):
            destdir = tempfile.mkdtemp()
            self.test_dirs.append(destdir)
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
                self.assertEqual(res[(res.qid=='0')&((res['rank']==1))].iloc[0]['docno'], '0')
                self.assertEqual(res[(res.qid=='1')&((res['rank']==1))].iloc[0]['docno'], '1')
            else:
                self.assertTrue(len(res) <= 200)
                self.assertTrue(len(res[res.qid=='0']) <= 100)
                self.assertTrue(len(res[res.qid=='1']) <= 100)

    @unittest.skipIf(not pyterrier_dr.util.package_available('faiss'), "faiss not available")
    def test_faiss_flat_retriever(self):
        self._test_retr(FlexIndex.faiss_flat_retriever)

    @unittest.skipIf(not pyterrier_dr.util.package_available('faiss'), "faiss not available")
    def test_faiss_hnsw_retriever(self):
        self._test_retr(FlexIndex.faiss_hnsw_retriever, exact=False)

    @unittest.skipIf(not pyterrier_dr.util.package_available('faiss'), "faiss not available")
    def test_faiss_ivf_retriever(self):
        self._test_retr(FlexIndex.faiss_ivf_retriever, exact=False)

    def test_np_retriever(self):
        self._test_retr(FlexIndex.np_retriever)

    def test_torch_retriever(self):
        self._test_retr(FlexIndex.torch_retriever)

    def test_np_vec_loader(self):
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir+'/index')
        dataset = self._generate_data()
        index.index(dataset)
        
        vec_loader = index.np_vec_loader()
        with self.subTest('docid'):
            res = vec_loader(pd.DataFrame({
                'docid': [5, 1, 100, 198],
            }))
            self.assertTrue((res.iloc[0]['doc_vec'] == dataset[5]['doc_vec']).all())
            self.assertTrue((res.iloc[1]['doc_vec'] == dataset[1]['doc_vec']).all())
            self.assertTrue((res.iloc[2]['doc_vec'] == dataset[100]['doc_vec']).all())
            self.assertTrue((res.iloc[3]['doc_vec'] == dataset[198]['doc_vec']).all())
        with self.subTest('docno'):
            res = vec_loader(pd.DataFrame({
                'docno': ['20', '0', '100', '198'],
            }))
            self.assertTrue((res.iloc[0]['doc_vec'] == dataset[20]['doc_vec']).all())
            self.assertTrue((res.iloc[1]['doc_vec'] == dataset[0]['doc_vec']).all())
            self.assertTrue((res.iloc[2]['doc_vec'] == dataset[100]['doc_vec']).all())
            self.assertTrue((res.iloc[3]['doc_vec'] == dataset[198]['doc_vec']).all())

    # TODO: tests for:
    #  - pre_ladr
    #  - ada_ladr
    #  - gar
    #  - np_scorer
    #  - scann_retriever
    #  - torch_vecs
    #  - torch_scorer

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        self.test_dirs = []

    def tearDown(self):
        import shutil
        for d in self.test_dirs: 
            try:
                shutil.rmtree(d)
            except:
                pass


if __name__ == '__main__':
    unittest.main()
