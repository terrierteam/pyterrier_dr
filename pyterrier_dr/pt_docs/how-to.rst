Dense Retrieval How-To Guides
============================================================


.. how-to:: How do I retrieve over a dense index?

    .. code-block:: python
        :caption: Retrieving over a dense index

        import pyterrier as pt
        import pyterrier_dr

        index = pyterrier_dr.FlexIndex('my_index.flex') # :footnote: Specify the path where you want the index to be stored.
        model = pyterrier_dr.SBertBiEncoder('sentence-transformers/all-MiniLM-L6-v2') # :footnote: Specify the model used to create the index.
        retr = model.query_encoder() >> index.retriever() # :footnote: Create a retrieval pipeline by chaining the query encoder and the retriever.

        results = retr.search('a single query')
        # or
        results = retr([
            {'qid': '1', 'query': 'multiple queries'},
            {'qid': '2', 'query': 'can be passed as a list of dicts'},
        ])


.. how-to:: How do I index documents into a dense index?

    .. code-block:: python
        :caption: Indexing documents into a FlexIndex

        import pyterrier as pt
        import pyterrier_dr

        index = pyterrier_dr.FlexIndex('my_index.flex') # :footnote: Specify the path where you want the index to be stored.
        model = pyterrier_dr.SBertBiEncoder('sentence-transformers/all-MiniLM-L6-v2') # :footnote: Specify the model used to create the index.
        indexer = model.doc_encoder() >> index.indexer() # :footnote: Create an indexing pipeline by chaining the document encoder and the indexer.

        docs = [ # :footnote: ``docs`` can be any *iterable* of documents, including generators. This allows you to index collections that are too large to fit in memory at once.
            {'docid': 'doc1', 'text': 'This is the first document.'},
            {'docid': 'doc2', 'text': 'This is the second document.'},
            # Add more documents as needed
        ]

        indexer.index(docs)


.. how-to:: How do I perform re-ranking using a dense index?
    
    .. _pyterrier-dr:how-to:dense-index-reranking:

    This example assumes that you already built a dense index for your collection. If you want to perform re-ranking "on-the-fly"
    for a dense model, check out :ref:`the next guide <pyterrier-dr:how-to:dense-model-reranking>`.

    .. code-block:: python
        :caption: Re-ranking BM25 results using a FlexIndex

        import pyterrier as pt
        import pyterrier_dr

        sparse_index = pt.terrier.TerrierIndex('my_index.terrier') # :footnote: In this example, we use a sample sparse index with for initial retrieval.
        dense_index = pyterrier_dr.FlexIndex('my_index.flex') # :footnote: Specify the path where you want the index to be stored.
        model = pyterrier_dr.SBertBiEncoder('sentence-transformers/all-MiniLM-L6-v2') # :footnote: Specify the model used to create the index.
        retr = sparse_index.bm25() >> model.query_encoder() >> dense_index.scorer() # :footnote: Create a re-ranking pipeline by chaining an initial retriever, a query encoder, and a scorer.

        retr.search('my query')


.. how-to:: How do I perform re-ranking using a dense model?
    
    .. _pyterrier-dr:how-to:dense-model-reranking:

    This example performs re-ranking "on-the-fly" using a dense model without requiring a dense index. If you want to perform re-ranking
    using a dense index, check out :ref:`the previous guide <pyterrier-dr:how-to:dense-index-reranking>`.

    .. code-block:: python
        :caption: Re-ranking BM25 results using a dense model

        import pyterrier as pt
        import pyterrier_dr

        sparse_index = pt.terrier.TerrierIndex('my_index.terrier') # :footnote: In this example, we use a sample sparse index with for initial retrieval.
        model = pyterrier_dr.SBertBiEncoder('sentence-transformers/all-MiniLM-L6-v2') # :footnote: Specify the model you want to use as a re-ranker.
        retr = sparse_index.bm25(include_fields=['text']) >> model.text_scorer() # :footnote: Create a re-ranking pipeline by chaining an initial retriever and a text scorer.

        retr.search('my query')
