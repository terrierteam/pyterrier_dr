Indexing & Retrieval
=====================================================

This page covers the indexing and retrieval functionality provided by ``pyterrier_dr``.

:class:`~pyterrier_dr.FlexIndex` provides a flexible way to index and retrieve documents
using dense vectors, and is the main class for indexing and retrieval.

API Documentation
-----------------------------------------------------

.. autoclass:: pyterrier_dr.FlexIndex
    :show-inheritance:

    Indexing
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    Basic indexing functionality is provided through :meth:`index`. For more advanced options, use :meth:`indexer`.

    .. automethod:: index
    .. automethod:: indexer

    Retrieval
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    ``FlexIndex`` provides a variety of retriever backends. Each one expects ``qid`` and ``query_vec`` columns
    as input, and outputs a result frame. When you do not care about which backend you want, you can use
    :meth:`retriever` (an alias to :meth:`np_retriever`), which preforms exact retrieval using a brute force search
    over all vectors.

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

    Results can be re-ranked using indexed vectors using :meth:`scorer`. (:meth:`np_scorer` and :meth:`torch_scorer` are
    available as specific implementations, if needed.)

    :meth:`gar`, :meth:`ladr_proactive`, and :meth:`ladr_adaptive` are *adaptive* re-ranking approaches that pull in other
    documents from the corpus that may be relevant.

    .. py:method:: scorer

        An alias to :meth:`np_scorer`.

    .. automethod:: np_scorer
    .. automethod:: torch_scorer
    .. automethod:: gar
    .. automethod:: ladr_proactive
    .. automethod:: ladr_adaptive

    Index Data Access
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    These methods are for low-level index data access.

    .. automethod:: vec_loader
    .. automethod:: get_corpus_iter
    .. automethod:: np_vecs
    .. automethod:: torch_vecs
    .. automethod:: docnos
    .. automethod:: corpus_graph
    .. automethod:: faiss_hnsw_graph

    Extras
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. automethod:: built
    .. py:classmethod:: from_hf(repo)

            Loads the index from HuggingFace Hub.

            :param repo: The repository name download from.

            :returns: A :class:`~pyterrier_dr.FlexIndex` object.

    .. py:method:: to_hf(repo)

            Uploads the index to HuggingFace Hub.

            :param repo: The repository name to upload to.
