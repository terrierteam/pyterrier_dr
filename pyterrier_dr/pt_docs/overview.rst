Overview
=======================================================

Installation
-------------------------------------------------------

``pyterrier-dr`` can be installed with ``pip``.

.. code-block:: console
    :caption: Install ``pyterrier-dr`` with ``pip``

    $ pip install pyterrier-dr

.. hint::

    Some functionality requires the installation ot other software packages. For instance, to retrieve using
    `FAISS <https://github.com/facebookresearch/faiss>`__ (e.g., using :meth:`~pyterrier_dr.FlexIndex.faiss_hnsw_retriever`),
    you will need to install the FAISS package:

    .. code-block:: bash
        :caption: Install FAISS with ``pip`` or ``conda``

        pip install faiss-cpu
        # or with conda:
        conda install -c pytorch faiss-cpu
        # or with GPU support:
        conda install -c pytorch faiss-gpu

Basic Usage
-------------------------------------------------------

Dense Retrieval consists of two main components: (1) a model that encodes content as dense vectors,
and (2) algorithms and data structures to index and retrieve documents using these dense vectors.

Encoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(More information can be found at :doc:`encoding`.)

Let's start by loading a dense model: `RetroMAE <https://arxiv.org/abs/2205.12035>`__. The model has several
checkpoints available on huggingface, including ``Shitao/RetroMAE_MSMARCO_distill``.
``pyterrier_dr`` provides an alias to this checkpoint with :meth:`RetroMAE.msmarco_distill() <pyterrier_dr.RetroMAE.msmarco_distill>`:[#]_

.. code-block:: python
    :caption: Loading a dense model with  ``pyterrier_dr``

    >>> from pyterrier_dr import RetroMAE
    >>> model = RetroMAE.msmarco_distill()

Dense models model acts as transformers that can encode queries and documents into dense vectors. For example:

.. code-block:: python
    :caption: Encode queries and documents with a dense model

    >>> import pandas as pd
    >>> model(pd.DataFrame([
    ...   {"qid": "0", "query": "hello terrier"},
    ...   {"qid": "1", "query": "information retrieval"},
    ...   {"qid": "2", "query": "chemical reactions"},
    ... ]))
    qid                query                          query_vec
    0          hello terrier  [ 0.26, -0.17,  0.49, -0.12, ...]
    1  information retrieval  [-0.49,  0.16,  0.24,  0.38, ...]
    2     chemical reactions  [ 0.19,  0.11, -0.08, -0.00, ...]

    >>> model(pd.DataFrame([                                                                                                                               
    ...   {"docno": "1161848_2", "text": "Cutest breed of dog is a PBGV (look up on Internet) they are a little hound that looks like a shaggy terrier."},
    ...   {"docno": "686980_0",  "text": "Golden retriever has longer hair and is a little heavier."},                                                                                                                              
    ...   {"docno": "4189224_1", "text": "The onion releases a chemical that makes your eyes water up. I mean, no way short of wearing a mask or just avoiding the sting."},
    ... ]))
        docno                              text                          doc_vec
    1161848_2  Cutest breed of dog is a PBGV...  [0.03, -0.17, 0.18, -0.03, ...]
     686980_0  Golden retriever has longer h...  [0.14, -0.20, 0.00,  0.34, ...]
    4189224_1  The onion releases a chemical...  [0.16,  0.03, 0.49, -0.41, ...]

``query_vec`` and ``doc_vec`` are dense vectors that represent the query and document, respectively. In the
next section, we will use these vectors to perform retrieval.

Indexing and Retrieval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(More information can be found at :doc:`indexing-retrieval`.)

:class:`pyterrier_dr.FlexIndex` provides dense indexing and retrieval capabilities. Here's how you can index
a collection of documents:

.. code-block:: python
    :caption: Indexing documents with ``pyterrier_dr``

    >>> from pyterrier_dr import FlexIndex, RetroMAE
    >>> model = RetroMAE.msmarco_distill()
    >>> index = FlexIndex('my-index.flex')
    # build an indexing pipeline that first applies RetroMAE to get dense vectors, then indexes them into the FlexIndex
    >>> pipeline = model >> index.indexer()
    # run the indexing pipeline over a set of documents
    >>> pipeline.index([
    ...   {"docno": "1161848_2", "text": "Cutest breed of dog is a PBGV (look up on Internet) they are a little hound that looks like a shaggy terrier."},
    ...   {"docno": "686980_0",  "text": "Golden retriever has longer hair and is a little heavier."},
    ...   {"docno": "4189224_1", "text": "The onion releases a chemical that makes your eyes water up. I mean, no way short of wearing a mask or just avoiding the sting."},
    ... ])

Now that the documents are indexed, you can retrieve over them:

.. code-block:: python
    :caption: Retrieving with ``pyterrier_dr``

    >>> from pyterrier_dr import FlexIndex, RetroMAE
    >>> model = RetroMAE.msmarco_distill()
    >>> index = FlexIndex('my-index.flex')
    # build a retrieval pipeline that first applies RetroMAE to encode the query, then retrieves using those vectors over the FlexIndex
    >>> pipeline = model >> index.retriever()
    # run the indexing pipeline over a set of documents
    >>> pipeline.search('golden retrievers')
      qid              query      docno  docid      score  rank
    0   1  golden retrievers   686980_0      1  77.125557     0
    1   1  golden retrievers  1161848_2      0  61.379417     1
    2   1  golden retrievers  4189224_1      2  54.269958     2

Extras
-------------------------------------------------------

#. You can load models from the wonderful `Sentence Transformers <https://sbert.net/>`__ library directly
   using :class:`~pyterrier_dr.SBertBiEncoder`.

#. Dense indexing is the most common way to use dense models. But you can also score
   any pair of text using a dense model using :meth:`BiEncoder.text_scorer() <pyterrier_dr.BiEncoder.text_scorer>`.

#. Re-ranking can often yield better trade-offs between effectiveness and efficiency than doing dense retrieval.
   You can build a re-ranking pipeline with :meth:`FlexIndex.scorer() <pyterrier_dr.FlexIndex.scorer>`.

#. Dense Pseudo-Relevance Feedback (PRF) is a technique to improve the performance of a retrieval system by expanding
   the original query vector with the vectors from the top-ranked documents. Check out more :doc:`here <prf>`.

-------------------------------------------------------

.. [#] You can also load the model from HuggingFace with :class:`~pyterrier_dr.HgfBiEncoder`:
   ``HgfBiEncoder("Shitao/RetroMAE_MSMARCO_distill")``. Using the alias will ensure that all settings for
   the model are assigned properly.
