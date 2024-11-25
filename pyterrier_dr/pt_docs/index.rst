Dense Retrieval for PyTerrier
=======================================================

Features to support Dense Retrieval in `PyTerrier <https://github.com/terrier-org/pyterrier>`__.

.. rubric:: Getting Started

.. code-block:: console
    :caption: Install ``pyterrier-dr`` with ``pip``

    $ pip install pyterrier-dr

Import ``pyterrier_dr``, load a pre-built index and model, and retrieve:

.. code-block:: python
    :caption: Basic example of using ``pyterrier_dr``

    >>> from pyterrier_dr import FlexIndex, TasB

    >>> index = FlexIndex.from_hf('macavaney/vaswani.tasb.flex')
    >>> model = TasB('sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco')
    >>> pipeline = model.query_encoder() >> index.np_retriever()
    >>> pipeline.search('chemical reactions')

             score  docno  docid  rank qid               query
    0    95.841721   7049   7048     0   1  chemical reactions
    1    94.669395   9374   9373     1   1  chemical reactions
    2    93.520027   3101   3100     2   1  chemical reactions
    3    92.809227   6480   6479     3   1  chemical reactions
    4    92.376190   3452   3451     4   1  chemical reactions
    ..         ...    ...    ...   ...  ..                 ...
    995  82.554390   7701   7700   995   1  chemical reactions
    996  82.552139   1553   1552   996   1  chemical reactions
    997  82.551933  10064  10063   997   1  chemical reactions
    998  82.546890   4417   4416   998   1  chemical reactions
    999  82.545776   7120   7119   999   1  chemical reactions


.. rubric:: Table of Contents

.. toctree::
    :maxdepth: 1

    prf
