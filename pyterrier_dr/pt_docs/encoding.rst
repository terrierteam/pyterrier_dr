Encoding
==================================================================

Sentence Transformers
------------------------------------------------------------------

With pyterrier_dr, its easy to support Sentence Transformer (formerly called SentenceBERT)
models, e.g. from HuggingFace, for dense retrieval.

The base class is ``SBertBiEncoder('huggingface/path')``.

Pretrained Encoders
------------------------------------------------------------------

These classes are convenience aliases to popular dense encoding models.

.. autoclass:: pyterrier_dr.Ance()
.. autoclass:: pyterrier_dr.BGEM3()
.. autoclass:: pyterrier_dr.CDE()
.. autoclass:: pyterrier_dr.E5()
.. autoclass:: pyterrier_dr.GTR()
.. autoclass:: pyterrier_dr.Query2Query()
.. autoclass:: pyterrier_dr.RetroMAE()
.. autoclass:: pyterrier_dr.TasB()
.. autoclass:: pyterrier_dr.TctColBert()

API Documentation
------------------------------------------------------------------

.. autoclass:: pyterrier_dr.BiEncoder
   :members:
