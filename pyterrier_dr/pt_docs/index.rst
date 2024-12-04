Dense Retrieval for PyTerrier
=======================================================

`pyterrier-dr <https://github.com/terrierteam/pyterrier_dr>`__ is a PyTerrier plugin
that provides functionality for Dense Retrieval.

It provides this functionality primarily through:

1. Transformers for :doc:`encoding queries/documents <./encoding>` into dense vectors (e.g., :class:`~pyterrier_dr.SBertBiEncoder`)

2. Transformers for :doc:`indexing and retrieval <./indexing-retrieval>` using these dense vectors (e.g., :class:`~pyterrier_dr.FlexIndex`)

This functionality is covered in more detail in the following pages:

.. toctree::
    :maxdepth: 1

    overview
    encoding
    indexing-retrieval
    prf
