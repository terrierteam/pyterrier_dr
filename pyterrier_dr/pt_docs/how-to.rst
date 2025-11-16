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
