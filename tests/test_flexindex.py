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

    def _test_retr(self, Retr, exact=True, test_smaller=True, post_test_fn=None):
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
                if post_test_fn is not None:
                    post_test_fn(index)

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

    @unittest.skipIf(not pyterrier_dr.util.kannolo_available(), "kannolo not available")
    def test_kannolo_hnsw_retriever(self):
        def _check_cache_fn(index):
            m=32
            ef_construction=200
            self.assertIn(f'kannolo_hnsw-{m}_ef-{ef_construction}', index._cache)
            retr2 = index.kannolo_hnsw_retriever()

        self._test_retr(FlexIndex.kannolo_hnsw_retriever, exact=False, post_test_fn=_check_cache_fn)

    @unittest.skipIf(not pyterrier_dr.util.scann_available(), "scann not available")
    def test_scann_retriever(self):
        self._test_retr(FlexIndex.scann_retriever, exact=False)

    @unittest.skipIf(not pyterrier_dr.util.flatnav_available(), "flatnav not available")
    def test_flatnav_retriever(self):
        self._test_retr(FlexIndex.flatnav_retriever, exact=False)

    def test_np_retriever(self):
        self._test_retr(FlexIndex.np_retriever)

    def test_torch_retriever(self):
        self._test_retr(FlexIndex.torch_retriever)

    # --- Group A: mask tests for NumpyRetriever ---

    # Mask of all 1s should produce identical results to no mask
    def test_np_retriever_mask_all_ones(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            mask = np.ones(200, dtype=np.float32)
            retr_nomask = index.np_retriever(num_results=100)
            retr_mask = index.np_retriever(num_results=100, mask=mask)

            inp = pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
                {'qid': '1', 'query_vec': dataset[1]['doc_vec']},
            ])
            res_nomask = retr_nomask(inp)
            res_mask = retr_mask(inp)

            self.assertEqual(len(res_nomask), len(res_mask))
            for qid, expected_docno in [('0', '0'), ('1', '1')]:
                r0_nomask = res_nomask[(res_nomask.qid == qid) & (res_nomask['rank'] == 0)].iloc[0]
                r0_mask = res_mask[(res_mask.qid == qid) & (res_mask['rank'] == 0)].iloc[0]
                self.assertEqual(r0_nomask['docno'], expected_docno)
                self.assertEqual(r0_mask['docno'], expected_docno)
                np.testing.assert_almost_equal(r0_mask['score'], r0_nomask['score'], decimal=5)

    # Mask of all 0s should zero out all scores
    def test_np_retriever_mask_zeros(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            mask = np.zeros(200, dtype=np.float32)
            retr = index.np_retriever(num_results=100, mask=mask)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
            ]))
            np.testing.assert_allclose(res['score'].values, 0.0, atol=1e-7)

    # Zeroing out a doc's mask should remove it from rank 0
    def test_np_retriever_mask_exclude_rank0(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            mask = np.ones(200, dtype=np.float32)
            mask[0] = 0.0
            retr = index.np_retriever(num_results=100, mask=mask)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
            ]))
            res_q0 = res[res.qid == '0']
            self.assertNotEqual(res_q0[res_q0['rank'] == 0].iloc[0]['docno'], '0')
            doc0_row = res_q0[res_q0['docno'] == '0']
            if len(doc0_row) > 0:
                np.testing.assert_almost_equal(doc0_row.iloc[0]['score'], 0.0, decimal=5)
            self.assertGreater(res_q0[res_q0['rank'] == 0].iloc[0]['score'], 0.0)

    # Masking out multiple specific docs should zero their scores and leave others unchanged
    def test_np_retriever_mask_multiple_docs(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            mask = np.ones(200, dtype=np.float32)
            mask[0] = 0
            mask[5] = 0
            mask[10] = 0

            inp = pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
            ])
            res_nomask = index.np_retriever(num_results=200)(inp)
            res_mask = index.np_retriever(num_results=200, mask=mask)(inp)

            for docno in ['0', '5', '10']:
                row = res_mask[res_mask['docno'] == docno]
                if len(row) > 0:
                    np.testing.assert_almost_equal(row.iloc[0]['score'], 0.0, decimal=5)

            score_orig = res_nomask[res_nomask['docno'] == '1'].iloc[0]['score']
            score_masked = res_mask[res_mask['docno'] == '1'].iloc[0]['score']
            np.testing.assert_almost_equal(score_masked, score_orig, decimal=5)

    # Non-binary mask values should raise ValueError
    def test_np_retriever_mask_rejects_non_binary(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            mask = np.ones(200, dtype=np.float32)
            mask[0] = 0.5
            with self.assertRaises(ValueError):
                index.np_retriever(num_results=100, mask=mask)

    # fuse_rank_cutoff should propagate the mask to the new retriever
    def test_np_retriever_mask_fuse_rank_cutoff(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            mask = np.ones(200, dtype=np.float32)
            mask[0] = 0.0
            retr = index.np_retriever(num_results=100, mask=mask)
            fused = retr.fuse_rank_cutoff(10)

            self.assertIsNotNone(fused)
            self.assertEqual(fused.num_results, 10)
            self.assertIs(fused.mask, retr.mask)

            res = fused(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
            ]))
            self.assertLessEqual(len(res[res.qid == '0']), 10)
            self.assertNotEqual(res[res['rank'] == 0].iloc[0]['docno'], '0')

    # Empty query DataFrame with mask should return empty results
    def test_np_retriever_mask_empty_queries(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            mask = np.ones(200, dtype=np.float32)
            retr = index.np_retriever(num_results=100, mask=mask)
            res = retr(pd.DataFrame(columns=['qid', 'query_vec']))
            self.assertEqual(len(res), 0)

    # --- Group B: index_select tests for TorchRetriever ---

    # Only selected document IDs should appear in results
    def test_torch_retriever_index_select_subset(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            index_select = np.array([0, 1, 2, 3, 4], dtype=np.int64)
            retr = index.torch_retriever(num_results=100, index_select=index_select)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
            ]))
            res_q0 = res[res.qid == '0']
            self.assertEqual(len(res_q0), 5)
            self.assertTrue(set(res_q0['docno'].values).issubset({'0', '1', '2', '3', '4'}))
            self.assertEqual(res_q0[res_q0['rank'] == 0].iloc[0]['docno'], '0')

    # Excluding the best-matching doc should remove it from results
    def test_torch_retriever_index_select_exclude_match(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            index_select = np.array([2, 3, 4, 5, 6], dtype=np.int64)
            retr = index.torch_retriever(num_results=100, index_select=index_select)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
            ]))
            res_q0 = res[res.qid == '0']
            self.assertNotIn('0', res_q0['docno'].values)
            self.assertTrue(set(res_q0['docno'].values).issubset({'2', '3', '4', '5', '6'}))
            self.assertEqual(len(res_q0), 5)

    # Selecting all docs should produce identical results to no index_select
    def test_torch_retriever_index_select_all(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            index_select = np.arange(200, dtype=np.int64)
            inp = pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
                {'qid': '1', 'query_vec': dataset[1]['doc_vec']},
            ])
            res_plain = index.torch_retriever(num_results=100)(inp)
            res_sel = index.torch_retriever(num_results=100, index_select=index_select)(inp)

            self.assertEqual(len(res_plain), len(res_sel))
            for qid, expected_docno in [('0', '0'), ('1', '1')]:
                r0_plain = res_plain[(res_plain.qid == qid) & (res_plain['rank'] == 0)].iloc[0]
                r0_sel = res_sel[(res_sel.qid == qid) & (res_sel['rank'] == 0)].iloc[0]
                self.assertEqual(r0_plain['docno'], expected_docno)
                self.assertEqual(r0_sel['docno'], expected_docno)
                np.testing.assert_almost_equal(r0_sel['score'], r0_plain['score'], decimal=5)

    # When fewer docs are selected than num_results, return all selected docs
    def test_torch_retriever_index_select_fewer_than_num_results(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            index_select = np.array([10, 20, 30], dtype=np.int64)
            retr = index.torch_retriever(num_results=100, index_select=index_select)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[10]['doc_vec']},
            ]))
            res_q0 = res[res.qid == '0']
            self.assertEqual(len(res_q0), 3)
            self.assertTrue(set(res_q0['docno'].values).issubset({'10', '20', '30'}))
            self.assertEqual(res_q0[res_q0['rank'] == 0].iloc[0]['docno'], '10')

    # fuse_rank_cutoff should propagate index_select to the new retriever
    def test_torch_retriever_index_select_fuse_rank_cutoff(self):
        import torch
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            index_select = np.array([0, 1, 2, 3, 4], dtype=np.int64)
            retr = index.torch_retriever(num_results=100, index_select=index_select)
            fused = retr.fuse_rank_cutoff(3)

            self.assertIsNotNone(fused)
            self.assertEqual(fused.num_results, 3)
            self.assertIsNotNone(fused.index_select)
            self.assertTrue(torch.equal(fused.index_select, retr.index_select))

            res = fused(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
            ]))
            res_q0 = res[res.qid == '0']
            self.assertLessEqual(len(res_q0), 3)
            self.assertTrue(set(res_q0['docno'].values).issubset({'0', '1', '2', '3', '4'}))
            self.assertEqual(res_q0[res_q0['rank'] == 0].iloc[0]['docno'], '0')

    # Returned docnos and docids should map to original index positions, not subset positions
    def test_torch_retriever_index_select_docno_mapping(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            index_select = np.array([50, 100, 150], dtype=np.int64)
            retr = index.torch_retriever(num_results=10, index_select=index_select)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[50]['doc_vec']},
            ]))
            res_q0 = res[res.qid == '0']
            self.assertTrue(set(res_q0['docno'].values).issubset({'50', '100', '150'}))
            self.assertTrue(set(res_q0['docid'].values).issubset({50, 100, 150}))
            self.assertEqual(res_q0[res_q0['rank'] == 0].iloc[0]['docno'], '50')

    # Selecting a single document should return exactly one result
    def test_torch_retriever_index_select_single_doc(self):
        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=200, dim=50)
            index.index(dataset)

            index_select = np.array([42], dtype=np.int64)
            retr = index.torch_retriever(num_results=100, index_select=index_select)
            res = retr(pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[42]['doc_vec']},
            ]))
            res_q0 = res[res.qid == '0']
            self.assertEqual(len(res_q0), 1)
            self.assertEqual(res_q0.iloc[0]['docno'], '42')
            self.assertEqual(res_q0.iloc[0]['docid'], 42)

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
