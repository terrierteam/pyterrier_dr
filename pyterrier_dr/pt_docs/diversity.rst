Diversity
=======================================================

Search Result Diversification
-------------------------------------------------------

``pyterrier-dr`` provides one diversification algorithm, :class:`~pyterrier_dr.MmrScorer` (Maximal Marginal Relevance).
The transformer works over input dataframes that contain the dense vectors of the documents and the query. You can also
use :meth:`~pyterrier_dr.FlexIndex.mmr` to first load vectors from an index and then apply MMR.

.. autoclass:: pyterrier_dr.MmrScorer

Diversity Evaluation
-------------------------------------------------------

``pyterrier-dr`` provides one diversity evaluation measure, :func:`~pyterrier_dr.ILS` (Intra-List Similarity),
which can be used to evaluate the diversity of search results based on the dense vectors of a :class:`~pyterrier_dr.FlexIndex`.

This measure can be used alongside PyTerrier's built-in evaluation measures in a :func:`pyterrier.Experiment`.

.. code-block:: python
    :caption: Compare the relevance and ILS of lexical and dense retrieval with a PyTerrier Experiment

    import pyterrier as pt
    from pyterrier.measures import nDCG, R
    from pyterrier_dr import FlexIndex, TasB
    from pyterrier_pisa import PisaIndex

    dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')
    index = FlexIndex.from_hf('macavaney/msmarco-passage.tasb.flex')
    bm25 = PisaIndex.from_hf('macavaney/msmarco-passage.pisa').bm25()
    model = TasB.dot()

    pt.Experiment(
        [
            bm25,
            model >> index.retriever(),
            model >> index.retriever() >> index.mmr(),
        ],
        dataset.get_topics(),
        dataset.get_qrels(),
        [nDCG@10, R(rel=2)@1000, index.ILS@10, index.ILS@1000]
    )
    #        name   nDCG@10  R(rel=2)@1000    ILS@10  ILS@1000
    # BM25            0.498          0.755     0.852     0.754
    # TasB            0.716          0.841     0.889     0.775
    # TasB w/ MMR     0.714          0.841     0.888     0.775

.. autofunction:: pyterrier_dr.ILS
.. autofunction:: pyterrier_dr.ils
