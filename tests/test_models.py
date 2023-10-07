import unittest
import pandas as pd
import tempfile
import pyterrier as pt
import itertools


class TestModels(unittest.TestCase):

    def _base_test(self, model, test_query_encoder=True, test_doc_encoder=True, test_scorer=True, test_indexer=True, test_retriever=True):
        from pyterrier_dr import FlexIndex
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir+'/index')

        dataset = pt.get_dataset('irds:vaswani')

        docs = list(itertools.islice(pt.get_dataset('irds:vaswani').get_corpus_iter(), 200))

        if test_query_encoder:
            with self.subTest('query_encoder'):
                query_res = model(dataset.get_topics())
                # TODO: what to assert about query_res?

        if test_doc_encoder:
            with self.subTest('doc_encoder'):
                doc_res = model(pd.DataFrame(docs))
                # TODO: what to assert about doc_res?

        if test_scorer:
            with self.subTest('scorer'):
                # TODO: more comprehensive test case
                score_res = model(pd.DataFrame([
                    {'qid': '0', 'query': 'test query', 'docno': '0', 'text': 'test documemnt'},
                ]))
                # TODO: what to assert about score_res?

        if test_indexer:
            with self.subTest('indexer'):
                pipeline = model >> index
                pipeline.index(docs)
                # TODO: what to assert?

        if test_retriever:
            with self.subTest('retriever'):
                retr_res = pipeline(dataset.get_topics())
                # TODO: what to assert about retr_res?

    def test_tct(self):
        from pyterrier_dr import TctColBert
        self._base_test(TctColBert())
    
    def test_ance(self):
        from pyterrier_dr import Ance
        self._base_test(Ance.firstp())

    def test_tasb(self):
        from pyterrier_dr import TasB
        self._base_test(TasB.dot())

    def test_retromae(self):
        from pyterrier_dr import RetroMAE
        self._base_test(RetroMAE.msmarco_finetune())

    def test_query2query(self):
        from pyterrier_dr import Query2Query
        self._base_test(Query2Query(), test_doc_encoder=False, test_scorer=False, test_indexer=False, test_retriever=False)

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
