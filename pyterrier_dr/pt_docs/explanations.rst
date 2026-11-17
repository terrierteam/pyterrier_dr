Explanations (Sparse Autoencoders)
=====================================================

A :class:`~pyterrier_dr.FlexIndex` can be extended with *explanations* of its dense vectors, following
`Decoding Dense Embeddings: Sparse Autoencoders for Interpreting and Discretizing Dense Retrieval <https://arxiv.org/abs/2403.16058>`__.
A Batch-TopK sparse autoencoder is trained over the indexed document vectors, and its latents can be
described with an LLM and used to explain queries and documents.

The typical workflow is:

1. Train a sparse autoencoder with :meth:`~pyterrier_dr.FlexIndex.train_sae`. The model (``sae.pt``), its
   configuration (in ``pt_meta.json``) and the latent concepts of all documents (``latent_concepts.jsonl``)
   are saved to the index.
2. Optionally describe the latents with an LLM using
   :meth:`~pyterrier_dr.FlexIndex.generate_latent_descriptions` (requires the ``openai`` package).
   The descriptions are saved to ``latent_descriptions.jsonl`` in the index.
3. Annotate queries or documents with the descriptions of their most active latents using
   :meth:`~pyterrier_dr.FlexIndex.latent_describer`.

All of these methods load the SAE (via :meth:`~pyterrier_dr.FlexIndex.load_sae`) and the previously
serialised latent concepts and descriptions automatically when they are present in the index.

.. code-block:: python
   :caption: Example

   from pyterrier_dr import FlexIndex, SBertBiEncoder
   import pyterrier as pt

   model = SBertBiEncoder(
       "sentence-transformers/all-MiniLM-L6-v2",
       batch_size=32,
       text_field="text"
   )
   index = FlexIndex.from_hf('pyterrier-tutorial/bright.sustainable.mini.dense')

   # train a sparse autoencoder over the indexed vectors
   # (the model, its configuration and the latent concepts of all documents are saved to the index)
   sae, act_thresh, nmse = index.train_sae(hidden_mult=8, k=32, epochs=100)

   # describe the latents with an LLM (requires the "openai" package);
   # top_proportion=0.1 limits this to the 10% most frequently activated latents
   text_loader = pt.text.get_text(pt.get_dataset('irds:bright/sustainable-living'))
   descriptions = index.generate_latent_descriptions(text_loader, top_proportion=0.1, model='gpt-4.1-mini')

   # annotate queries with the descriptions of their most active latents
   explainer = index.latent_describer(descriptions, threshold=act_thresh, num_latents=5)
   (model.query_encoder() >> explainer).search('renewable energy storage and grid stability')


API Reference
=====================================================

.. autoclass:: pyterrier_dr.FlexIndex
   :noindex:

    Training
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. automethod:: train_sae
    .. automethod:: save_sae
    .. automethod:: load_sae
    .. automethod:: sae_config

    Latent Concepts
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. automethod:: extract_latent_concepts

    Descriptions
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. automethod:: generate_latent_descriptions
    .. automethod:: latent_describer
