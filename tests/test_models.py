import unittest
import tempfile
import itertools
import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier_dr import FlexIndex


class TestModels(unittest.TestCase):

    def _base_test(self, model, test_query_encoder=True, test_doc_encoder=True, test_scorer=True, test_indexer=True, test_retriever=True):
        dataset = pt.get_dataset('irds:vaswani')

        docs = list(itertools.islice(pt.get_dataset('irds:vaswani').get_corpus_iter(), 200))
        docs_df = pd.DataFrame(docs)

        if test_query_encoder:
            with self.subTest('query_encoder'):
                topics = dataset.get_topics()
                enc_topics = model(topics)
                self.assertEqual(len(enc_topics), len(topics))
                self.assertTrue('query_vec' in enc_topics.columns)
                self.assertTrue(all(c in enc_topics.columns for c in topics.columns))
                self.assertEqual(enc_topics.query_vec.dtype, object)
                self.assertEqual(enc_topics.query_vec[0].dtype, np.float32)
                self.assertTrue(all(enc_topics.query_vec[0].shape == v.shape for v in enc_topics.query_vec))

            with self.subTest('query_encoder empty'):
                enc_topics_empty = model(pd.DataFrame(columns=['qid', 'query']))
                self.assertEqual(len(enc_topics_empty), 0)
                self.assertTrue('query_vec' in enc_topics.columns)

        if test_doc_encoder:
            with self.subTest('doc_encoder'):
                enc_docs = model(pd.DataFrame(docs_df))
                self.assertEqual(len(enc_docs), len(docs_df))
                self.assertTrue('doc_vec' in enc_docs.columns)
                self.assertTrue(all(c in enc_docs.columns for c in docs_df.columns))
                self.assertEqual(enc_docs.doc_vec.dtype, object)
                self.assertEqual(enc_docs.doc_vec[0].dtype, np.float32)
                self.assertTrue(all(enc_docs.doc_vec[0].shape == v.shape for v in enc_docs.doc_vec))

            with self.subTest('doc_encoder empty'):
                enc_docs_empty = model(pd.DataFrame(columns=['docno', 'text']))
                self.assertEqual(len(enc_docs_empty), 0)
                self.assertTrue('doc_vec' in enc_docs_empty.columns)

        if test_scorer:
            with self.subTest('scorer_qtext_dtext'):
                res_qtext_dtext = topics.head(2).merge(docs_df, how='cross')
                scored_res_qtext_dtext = model(res_qtext_dtext)
                self.assertTrue('score' in scored_res_qtext_dtext.columns)
                self.assertTrue('rank' in scored_res_qtext_dtext.columns)
                self.assertTrue(all(c in scored_res_qtext_dtext.columns for c in res_qtext_dtext.columns))

            with self.subTest('scorer_qvec_dtext'):
                res_qvec_dtext = enc_topics.drop(columns=['query']).head(2).merge(docs_df, how='cross')
                scored_res_qvec_dtext = model(res_qvec_dtext)
                self.assertTrue('score' in scored_res_qvec_dtext.columns)
                self.assertTrue('rank' in scored_res_qvec_dtext.columns)
                self.assertTrue(all(c in scored_res_qvec_dtext.columns for c in res_qvec_dtext.columns))

            with self.subTest('scorer_qtext_dvec'):
                res_qtext_dvec = topics.head(2).merge(enc_docs.drop(columns=['text']), how='cross')
                scored_res_qtext_dvec = model(res_qtext_dvec)
                self.assertTrue('score' in scored_res_qtext_dvec.columns)
                self.assertTrue('rank' in scored_res_qtext_dvec.columns)
                self.assertTrue(all(c in scored_res_qtext_dvec.columns for c in res_qtext_dvec.columns))

            with self.subTest('scorer_qvec_dvec'):
                res_qvec_dvec = enc_topics.drop(columns=['query']).head(2).merge(enc_docs.drop(columns=['text']), how='cross')
                scored_res_qvec_dvec = model(res_qvec_dvec)
                self.assertTrue('score' in scored_res_qvec_dvec.columns)
                self.assertTrue('rank' in scored_res_qvec_dvec.columns)
                self.assertTrue(all(c in scored_res_qvec_dvec.columns for c in res_qvec_dvec.columns))

            with self.subTest('scorer empty'):
                enc_res_empty = model(pd.DataFrame(columns=['qid', 'query', 'docno', 'text']))
                self.assertEqual(len(enc_res_empty), 0)
                self.assertTrue('score' in enc_res_empty.columns)
                self.assertTrue('rank' in enc_res_empty.columns)

        with tempfile.TemporaryDirectory() as destdir:
            if test_indexer:
                with self.subTest('indexer'):
                    # Make sure this model can index properly
                    # More extensive testing of FlexIndex is done in test_flexindex
                    index = FlexIndex(destdir+'/index')
                    pipeline = model >> index
                    pipeline.index(docs)
                    self.assertTrue(index.built())
                    self.assertEqual(len(index), len(docs))

            if test_retriever:
                with self.subTest('retriever'):
                    assert test_indexer, "test_retriever requires test_indexer"
                    # Make sure this model can retrieve properly
                    # More extensive testing of FlexIndex is done in test_flexindex
                    retr_res = pipeline(dataset.get_topics())
                    self.assertTrue('qid' in retr_res.columns)
                    self.assertTrue('query' in retr_res.columns)
                    self.assertTrue('docno' in retr_res.columns)
                    self.assertTrue('score' in retr_res.columns)
                    self.assertTrue('rank' in retr_res.columns)
    
    def _test_bgem3_multi(self, model, test_query_multivec_encoder=False, test_doc_multivec_encoder=False):
        dataset = pt.get_dataset('irds:vaswani')

        docs = list(itertools.islice(pt.get_dataset('irds:vaswani').get_corpus_iter(), 200))
        docs_df = pd.DataFrame(docs)

        if test_query_multivec_encoder:
            with self.subTest('query_multivec_encoder'):
                topics = dataset.get_topics()
                enc_topics = model(topics)
                self.assertEqual(len(enc_topics), len(topics))
                self.assertTrue('query_toks' in enc_topics.columns)
                self.assertTrue('query_embs' in enc_topics.columns)
                self.assertTrue(all(c in enc_topics.columns for c in topics.columns))
                self.assertEqual(enc_topics.query_toks.dtype, object)
                self.assertTrue(all(isinstance(v, dict) for v in enc_topics.query_toks))
                self.assertEqual(enc_topics.query_embs.dtype, object)
                self.assertTrue(all(v.dtype == np.float32 for v in enc_topics.query_embs))
            with self.subTest('query_multivec_encoder empty'):
                enc_topics_empty = model(pd.DataFrame(columns=['qid', 'query']))
                self.assertEqual(len(enc_topics_empty), 0)
                self.assertTrue('query_toks' in enc_topics_empty.columns)
                self.assertTrue('query_embs' in enc_topics_empty.columns)
        if test_doc_multivec_encoder:
            with self.subTest('doc_multi_encoder'):
                enc_docs = model(pd.DataFrame(docs_df))
                self.assertEqual(len(enc_docs), len(docs_df))
                self.assertTrue('toks' in enc_docs.columns)
                self.assertTrue('doc_embs' in enc_docs.columns)
                self.assertTrue(all(c in enc_docs.columns for c in docs_df.columns))
                self.assertEqual(enc_docs.toks.dtype, object)
                self.assertTrue(all(isinstance(v, dict) for v in enc_docs.toks))
                self.assertEqual(enc_docs.doc_embs.dtype, object)
                self.assertTrue(all(v.dtype == np.float32 for v in enc_docs.doc_embs))
            with self.subTest('doc_multi_encoder empty'):
                enc_docs_empty = model(pd.DataFrame(columns=['docno', 'text']))
                self.assertEqual(len(enc_docs_empty), 0)
                self.assertTrue('toks' in enc_docs_empty.columns)
                self.assertTrue('doc_embs' in enc_docs_empty.columns)

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

    def test_gtr(self):
        from pyterrier_dr import GTR
        self._base_test(GTR.base())

    def test_query2query(self):
        from pyterrier_dr import Query2Query
        self._base_test(Query2Query(), test_doc_encoder=False, test_scorer=False, test_indexer=False, test_retriever=False)

    def test_bgem3(self):
        from pyterrier_dr import BGEM3
        # create BGEM3 instance
        bgem3 = BGEM3(max_length=1024)
        
        self._base_test(bgem3.query_multi_encoder(), test_doc_encoder=False, test_scorer=False, test_indexer=False, test_retriever=False)
        self._base_test(bgem3.doc_multi_encoder(), test_query_encoder=False, test_scorer=False, test_indexer=False, test_retriever=False)

        self._test_bgem3_multi(bgem3.query_multi_encoder(), test_query_multivec_encoder=True)
        self._test_bgem3_multi(bgem3.doc_multi_encoder(), test_doc_multivec_encoder=True)


if __name__ == '__main__':
    unittest.main()
