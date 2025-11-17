Dense Retrieval Overview
=======================================================

`pyterrier-dr <https://github.com/terrierteam/pyterrier_dr>`__ lets you construct single-vector dense indexing and retrieval pipelines.
These methods allow for retrieving based on *semantic matching* instead of the lexical matching used in traditional retrieval methods like BM25.

These processes involve two main components: **Dense Models** and **Dense Indexes**. These components are typically combined to create
**Indexing Pipelines** and **Retrieval Pipelines**.

Dense Models
------------------------------------------

**Dense Models** are transformers that encode text (queries and documents) into dense vectors. This package provides
various pretrained dense models, as well as the ability to load models from `Sentence Transformers <https://sbert.net/>`__ and `HuggingFace <https://huggingface.co/>`__.

.. code-block:: python
   :caption: Loading a dense model

   from pyterrier_dr import SBertBiEncoder
   model = SBertBiEncoder('sentence-transformers/all-MiniLM-L6-v2') # :footnote: You can replace this with any Sentence Transformer model name or path.

A dense model can perform a several of operations:

- Encode queries into dense vectors using its :meth:`~pyterrier_dr.BiEncoder.query_encoder`.
- Encode documents into dense vectors using its :meth:`~pyterrier_dr.BiEncoder.doc_encoder`.
- Re-rank results by encoding queries and documents and computing their similarity scores using its :meth:`~pyterrier_dr.BiEncoder.text_scorer`.

.. seealso::
    More information about dense encoders is available on the :doc:`encoding` page.

Dense Indexes
------------------------------------------

**Dense Indexes** are data structures and algorithms to index and retrieve documents using dense vectors. This package provides
:class:`~pyterrier_dr.FlexIndex`, which stores document vectors on disk and provides various retrieval backends.

.. code-block:: python
   :caption: Creating a dense index

   from pyterrier_dr import FlexIndex
   index = FlexIndex('path/to/index.flex') # :footnote: You can specify any path where you want the index to be stored. By convention, we use the ``.flex`` extension, but this is not required.

A dense index can perform several operations:

- Index documents using its :meth:`~pyterrier_dr.FlexIndex.indexer`.
- Retrieve documents using methods like :meth:`~pyterrier_dr.FlexIndex.retriever`, :meth:`~pyterrier_dr.FlexIndex.faiss_hnsw_retriever`, and more.
- Re-rank results using its stored vectors using methods like :meth:`~pyterrier_dr.FlexIndex.scorer`, :meth:`~pyterrier_dr.FlexIndex.ladr_adaptive`, and more.

.. seealso::
    More information about dense indexes is available on the :doc:`indexing-retrieval` page.

Pipelines
------------------------------------------

In most cases, you will want to combine dense models and dense indexes into pipelines.

**Indexing Pipelines** encode documents into dense vectors and stores them in an index.

.. schematic::
    :show_code:

    import pyterrier_dr
    index = pyterrier_dr.FlexIndex.example()
    model = pyterrier_dr.BiEncoder.example()
    # FOLD
    model.doc_encoder() >> index.indexer()

**Retrieval Pipelines** encode queries into dense vectors and retrieves documents from an index.

.. schematic::
    :show_code:

    import pyterrier_dr
    index = pyterrier_dr.FlexIndex.example()
    model = pyterrier_dr.BiEncoder.example()
    # FOLD
    model.query_encoder() >> index.retriever()


Putting it all Together
------------------------------------------

Here's an example of a complete dense retrieval pipeline that indexes documents and retrieves them using dense vectors.

.. code-block:: python
    :caption: Dense Indexing and Retrieval

    import pyterrier_dr
    index = pyterrier_dr.FlexIndex('my_index.flex')
    model = pyterrier_dr.SBertBiEncoder('sentence-transformers/all-MiniLM-L6-v2')
    indexer = model.doc_encoder() >> index.indexer() # :footnote: This is an indexing pipeline that encodes documents using the model's document encoder and indexes them into the flex index.
    indexer.index([ # :footnote: In this example, we only index four documents. In most cases, you'll index much larger collections.
        {'docno': 'doc1', 'text': 'A dog is a domesticated carnivorous mammal.'},
        {'docno': 'doc2', 'text': 'Dogs are known for their loyalty and companionship.'},
        {'docno': 'doc3', 'text': 'The domestic dog is a subspecies of the gray wolf.'},
        {'docno': 'doc4', 'text': 'Scottish Terriers are dogs that are known for their independent nature and distinctive appearance.'},
    ])

    retriever = model.query_encoder() >> index.retriever() # :footnote: We construct a retrievel pipeline using the default exact retriever. Other retrievers like FAISS HNSW can also be used.
    results = retriever.search('scottie dog')
    # qid        query  docno     score  rank
    #   1  scottie dog   doc4  0.428387     0 # :footnote: The top result is doc4, which is the most relevant document about Scottish Terriers.
    #   1  scottie dog   doc1  0.390875     1
    #   1  scottie dog   doc2  0.353309     2
    #   1  scottie dog   doc3  0.331017     3
