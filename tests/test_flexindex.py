import unittest
import pandas as pd
import tempfile
import pyterrier as pt
import numpy as np


class TestFlexIndex(unittest.TestCase):

    def _generate_data(self, count=1000, dim=100):
        return [
            {'docno': str(i), 'doc_vec': np.random.rand(dim).astype(np.float32)}
            for i in range(count)
        ]

    def test_index_typical(self):
        from pyterrier_dr import FlexIndex
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
        from pyterrier_dr import FlexIndex
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir+'/index')

        self.assertFalse(index.built())

        dataset = self._generate_data()

        index.index(dataset)
        graph = index.corpus_graph(16)
        self.assertEqual(graph.neighbours(4).shape, (16,))

    def test_faiss_hnsw_graph(self):
        from pyterrier_dr import FlexIndex
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir+'/index')

        self.assertFalse(index.built())

        dataset = self._generate_data()

        index.index(dataset)
        graph = index.faiss_hnsw_graph(16)
        self.assertEqual(graph.neighbours(4).shape, (16,))

    # TODO: tests for:
    #  - faiss_flat_retriever
    #  - faiss_hnsw_retriever
    #  - faiss_ivf_retriever
    #  - pre_ladr
    #  - ada_ladr
    #  - np_retriever
    #  - np_vec_loader
    #  - np_scorer
    #  - scann_retriever
    #  - torch_vecs
    #  - torch_scorer
    #  - torch_retriever

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
