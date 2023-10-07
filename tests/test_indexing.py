import unittest
import pandas as pd
import tempfile
import pyterrier as pt


class TestIndexingTCT(unittest.TestCase):

    def test_indexing_tct(self):
        from pyterrier_dr import FlexIndex, TctColBert
        destdir = tempfile.mkdtemp()
        self.test_dirs.append(destdir)
        index = FlexIndex(destdir, overwrite=True)

        model = TctColBert()

        # create an indexing pipelne
        idx_pipeline = model >> index
        
        iter = pt.get_dataset("vaswani").get_corpus_iter()
        # only index 200 docs
        idx_pipeline.index([ next(iter) for i in range(200) ])

        retr_pipeline = model >> index
        df1 = retr_pipeline.search("analogue computer")
        self.assertTrue(len(df1) > 0)

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
