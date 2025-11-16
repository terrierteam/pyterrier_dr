Dense Retrieval for PyTerrier
=======================================================

`pyterrier-dr <https://github.com/terrierteam/pyterrier_dr>`__ is a PyTerrier extension that provides single-vector [#]_
dense retrieval functionality through components for dense encoding, indexing, and retrieval.

You can install this extension with pip:

.. code-block:: bash

    pip install pyterrier-dr

Some functionality requires the installation ot other software packages. For instance, to retrieve using
`FAISS <https://github.com/facebookresearch/faiss>`__ (e.g., using :meth:`~pyterrier_dr.FlexIndex.faiss_hnsw_retriever`),
you will need to install the FAISS package:

.. tabs::
    .. tab:: With pip
        .. code-block:: bash

            pip install faiss-cpu

    .. tab:: With conda
        .. code-block:: bash

            conda install -c pytorch faiss-cpu

    .. tab:: With GPU support
        .. code-block:: bash

            conda install -c pytorch faiss-gpu


.. toctree::
    :caption: Contents
    :maxdepth: 1

    Overview <overview>
    Encoding <encoding>
    Indexing & Retrieval <indexing-retrieval>
    Pseudo-Relevance Feedback <prf>
    Diversity <diversity>
    How-To Guides <how-to>
    API Reference <api>

----

.. [#] If you are interested in multi-vector dense retrieval, check out `pyterrier-colbert <https://github.com/terrierteam/pyterrier_colbert>`__.
