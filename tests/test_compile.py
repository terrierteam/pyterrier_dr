import unittest
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

    def test_compilation_with_rank_and_averageprf(self):
        self._test_compilation_with_rank_and_prf(pyterrier_dr.AveragePrf)

    def test_compilation_with_rank_and_vectorprf(self):
        self._test_compilation_with_rank_and_prf(pyterrier_dr.VectorPrf)
    
    def _test_compilation_with_rank_and_prf(self, prf_clz):

        with tempfile.TemporaryDirectory() as destdir:
            index = FlexIndex(destdir+'/index')
            dataset = self._generate_data(count=2000)
            index.index(dataset)
            
            retr = index.retriever()
            queries = pd.DataFrame([
                {'qid': '0', 'query_vec': dataset[0]['doc_vec']},
                {'qid': '1', 'query_vec': dataset[1]['doc_vec']},
            ])

            pipe1 = retr >> prf_clz(k=3) >> retr
            pipe1_opt = pipe1.compile()
            self.assertEqual(3, pipe1_opt[0].num_results)
            self.assertEqual(1000, pipe1_opt[-1].num_results)
            #NB: pipe1 wouldnt actually work for PRF, as doc_vecs are not present. however compilation is valid

            pipe2 = retr >> index.vec_loader() >> pyterrier_dr.AveragePrf(k=3) >> (retr % 2)
            pipe2_opt = pipe2.compile()
            self.assertEqual(3, pipe2_opt[0].num_results)
            self.assertEqual(2, pipe2_opt[-1].num_results)

            res2 = pipe2(queries)
            res2_opt = pipe2_opt(queries)
            
            pd.testing.assert_frame_equal(res2, res2_opt)
            
            