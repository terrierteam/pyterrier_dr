import unittest
import pandas as pd
import tempfile
import pyterrier as pt


class TestIndexingTCT(unittest.TestCase):

    def _indexing_100doc(self, model : pt.Transformer, indexer_clz, dim=None):
        import pyterrier_dr
        
        # a memoryindex doesnt need a directory
        if indexer_clz == pyterrier_dr.MemIndex:
            index = indexer_clz()
        else:
            destdir = tempfile.mkdtemp()
            self.test_dirs.append(destdir)
            index = indexer_clz(destdir)

        # create an indexing pipelne
        idx_pipeline = model >> index
        
        iter = pt.get_dataset("vaswani").get_corpus_iter()
        # only index 200 docs
        idx_pipeline.index([ next(iter) for i in range(200) ])

        retr_pipeline = model >> index
        df1 = retr_pipeline.search("analogue computer")
        self.assertTrue(len(df1) > 0)
        
    
    def test_indexing_tct_numpy(self):
        import pyterrier_dr
        self._indexing_100doc(
            pyterrier_dr.TctColBert(),
            pyterrier_dr.NumpyIndex
        )

    def test_indexing_tct_torch(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("no cuda available")
        import pyterrier_dr
        self._indexing_100doc(
            pyterrier_dr.TctColBert(),
            pyterrier_dr.TorchIndex
        )
    
    def test_indexing_tct_faisshnsw(self):
        import pyterrier_dr
        self._indexing_100doc(
            pyterrier_dr.TctColBert(),
            pyterrier_dr.FaissHnsw
        )

    def test_indexing_tct_faissflat(self):
        import pyterrier_dr
        self._indexing_100doc(
            pyterrier_dr.TctColBert(),
            pyterrier_dr.FaissFlat
        )

    def test_indexing_tct_mem(self):
        import pyterrier_dr
        self._indexing_100doc(
            pyterrier_dr.TctColBert(),
            pyterrier_dr.MemIndex
        )

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        self.test_dirs = []

    def tearDown(self):
        import shutil
        try:
            for d in self.test_dirs: 
                shutil.rmtree(d)
        except:
            pass