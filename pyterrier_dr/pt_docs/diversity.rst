Diversity using Dense Vectors
=======================================================

Dense vectors can be used for both diversifying search results, and measuring the diversity of search results.
``pyterrier-dr`` provides functionality for both these use cases.

Search Result Diversification
-------------------------------------------------------

.. related:: pyterrier_dr.FlexIndex.mmr
.. related:: pyterrier_dr.MmrScorer

Maximal Marginal Relevance (MMR) is a technique to diversify search results by balancing relevance and novelty.
It is available using :class:`pyterrier_dr.MmrScorer`, which uses document vector similarities to measure novelty,
and the value in the ``score`` input column to measure relevance.

The transformer requires ``doc_vec`` columns to be present in the input data frame. Therefore, you
will usually want to load vectors from a :class:`~pyterrier_dr.FlexIndex` first using :meth:`~pyterrier_dr.FlexIndex.vec_loader`,
then apply :class:`~pyterrier_dr.MmrScorer`. :meth:`FlexIndex.mmr() <pyterrier_dr.FlexIndex.mmr>` is a shorthand to return both
these steps. Alternatively, you could include an encoder beforehand to compute document vectors on-the-fly.

The example below applies BM25 retrieval over a sparse index, then applies search result diversification to the results using MMR.

.. schematic::
    :show_code:

    import pyterrier_dr
    sparse_index = pt.terrier.TerrierIndex.example()
    dense_index = pyterrier_dr.FlexIndex.example()
    # FOLD
    sparse_index.bm25() >> dense_index.mmr()

.. cite.dblp:: conf/sigir/CarbonellG98

Diversity Evaluation
-------------------------------------------------------

.. related:: pyterrier_dr.ILS
.. related:: pyterrier_dr.ils
.. related:: pyterrier_dr.FlexIndex.ILS

Intra-List Similarity (ILS) is a diversity evaluation measure that quantifies the similarity between documents in a ranked list.
It is available using :func:`pyterrier_dr.ILS` or :meth:`FlexIndex.ILS <pyterrier_dr.FlexIndex.ILS>`.

This measure can be used alongside PyTerrier's built-in evaluation measures in a :func:`pt.Experiment <pyterrier.Experiment>`.

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

.. cite.dblp:: conf/www/ZieglerMKL05
