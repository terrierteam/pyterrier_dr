Evaluation
=======================================================

``pyterrier-dr`` provides an evaluation measure, :func:`pyterrier_dr.ILS` (Intra-List Similarity),
which can be used to evaluate the diversity of search results based on the dense vectors of a :class:`~pyterrier_dr.FlexIndex`.

This measure can be used alongside PyTerrier's built-in evaluation measures in a ``pt.Experiment``.

.. code-block:: python
    :caption: Compare the relevance and ILS of lexical and dense retrieval with a PyTerrier Experiment

    import pyterrier as pt
    from pyterrier.measures import nDCG, R
    from pyterrier_dr import FlexIndex, TasB
    from pyterrier_pisa import PisaIndex

    dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
    index = FlexIndex.from_hf('macavaney/msmarco-passage.tasb.flex')
    bm25 = PisaIndex.from_hf('macavaney/msmarco-passage.bm25.pisa').bm25()
    model = TasB.dot()

    pt.Experiment(
        [
            bm25,
            model >> index,
        ],
        dataset.get_topics(),
        dataset.get_qrels(),
        [nDCG@10, R(rel=2)@1000, index.ILS@10, index.ILS@1000]
    )
    #  name   nDCG@10  R(rel=2)@1000    ILS@10  ILS@1000
    # BM25   0.498902       0.755495  0.852248  0.754691
    # TAS-B  0.716068       0.841756  0.889112  0.775415

.. autofunction:: pyterrier_dr.ILS
.. autofunction:: pyterrier_dr.ils
