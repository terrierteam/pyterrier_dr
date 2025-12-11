Dense Retrieval API Reference
---------------------------------------------------------

Encoding
=====================================================

.. autoclass:: pyterrier_dr.BiEncoder
   :members:

.. autoclass:: pyterrier_dr.SBertBiEncoder
    :members:

Indexing and Retrieval
=====================================================

.. autoclass:: pyterrier_dr.FlexIndex

    Indexing
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. automethod:: index
    .. automethod:: indexer

    Retrieval
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. py:method:: retriever(*, num_results=1000)

        Returns a transformer that performs basic exact retrieval over indexed vectors using a brute force search. An alias to :meth:`np_retriever`.

    .. automethod:: np_retriever
    .. automethod:: torch_retriever
    .. automethod:: faiss_flat_retriever
    .. automethod:: faiss_hnsw_retriever
    .. automethod:: faiss_ivf_retriever
    .. automethod:: flatnav_retriever
    .. automethod:: scann_retriever
    .. automethod:: voyager_retriever

    Re-Ranking
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. py:method:: scorer

        An alias to :meth:`np_scorer`.

    .. automethod:: np_scorer
    .. automethod:: torch_scorer
    .. automethod:: gar
    .. automethod:: ladr_proactive
    .. automethod:: ladr_adaptive
    .. automethod:: mmr

    Evaluation
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. autoproperty:: ILS

    Index Data Access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. automethod:: built
    .. automethod:: vec_loader
    .. automethod:: get_corpus_iter
    .. automethod:: np_vecs
    .. automethod:: torch_vecs
    .. automethod:: docnos
    .. automethod:: corpus_graph
    .. automethod:: faiss_hnsw_graph

    Sharing
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. seealso::
       You can share Flex indices using the Artifacts API:

       - HuggingFace: :meth:`~pyterrier.Artifact.from_hf` and :meth:`~pyterrier.Artifact.to_hf`
       - Zenodo: :meth:`~pyterrier.Artifact.from_zenodo` and :meth:`~pyterrier.Artifact.to_zenodo`
       - Peer-to-peer: :meth:`~pyterrier.Artifact.from_p2p` and :meth:`~pyterrier.Artifact.to_p2p`
       - URLs: :meth:`~pyterrier.Artifact.from_url`


Pseudo-Relevance Feedback
=====================================================


.. autoclass:: pyterrier_dr.AveragePrf
    :members:

.. autoclass:: pyterrier_dr.VectorPrf
    :members:

Diversity
=====================================================

.. autoclass:: pyterrier_dr.MmrScorer
.. autofunction:: pyterrier_dr.ILS
.. autofunction:: pyterrier_dr.ils

Deprecated
====================================================

.. warning::
   The following classes are deprecated and will be removed in future releases.

.. autoclass:: pyterrier_dr.DocnoFile
.. autoclass:: pyterrier_dr.NilIndex
.. autoclass:: pyterrier_dr.NumpyIndex
.. autoclass:: pyterrier_dr.MemIndex
.. autoclass:: pyterrier_dr.FaissFlat
.. autoclass:: pyterrier_dr.FaissHnsw
.. autoclass:: pyterrier_dr.TorchIndex
