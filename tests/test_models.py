import unittest
import pandas as pd
import tempfile
import pyterrier as pt
import itertools


class TestModels(unittest.TestCase):

    def _base_test(self, model):
        from pyterrier_dr import FlexIndex
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir)

        dataset = pt.get_dataset('irds:vaswani')

        docs = list(itertools.islice(pt.get_dataset('irds:vaswani').get_corpus_iter(), 200))

        with self.subTest('query_encoder'):
            query_res = model(dataset.get_topics())

        with self.subTest('doc_encoder'):
            doc_res = model(pd.DataFrame(docs))

        with self.subTest('scorer'):
            score_res = model(pd.DataFrame([
                {'qid': '0', 'query': 'test query', 'docno': '0', 'text': 'test documemnt'},
            ]))

        with self.subTest('indexer'):
            pipeline = model >> index
            pipeline.index(docs)

        with self.subTest('retriever'):
            retr_res = pipeline(dataset.get_topics())

    def test_tct(self):
        from pyterrier_dr import TctColBert
        self._base_test(TctColBert())
    
    def test_ance(self):
        from pyterrier_dr import Ance
        self._base_test(Ance.firstp())

    def test_tasb(self):
        from pyterrier_dr import TasB
        self._base_test(TasB.dot())

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
