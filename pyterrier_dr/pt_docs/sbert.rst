Using Sentence Transformer models for Dense retrieval in PyTerrier
==================================================================

With PyTerrier_DR, its easy to support Sentence Transformer models, e.g. from HuggingFace, for dense retrieval.

The base class is ``SBertBiEncoder('huggingface/path')``;

There are easy to remember classes for a number of standard models:
 - ANCE - an early single-representation dense retrieval model: ``Ance.firstp()``
 - GTR - a dense retrieval model based on the T5 pre-trained encoder: ``GTR.base()``
 - `E5 <https://huggingface.co/intfloat/e5-base-v2>`: ``E5.base()``
 - Query2Query - a query similarity model: Query2Query()

The standard pyterrier_dr pipelines can be used:

Indexing::
    model = pyterrier_dr.GTR.base()
    index = pyterrier_dr.FlexIndex('gtr.flex')
    pipe = (model >> index)
    pipe.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())

Retrieval::
    pipe.search("chemical reactions")
