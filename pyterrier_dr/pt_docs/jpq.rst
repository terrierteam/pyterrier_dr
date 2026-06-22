Joint Product Quantization
=====================================================

Joint Product Quantization (JPQ) first learns product quantization (PQ) 
centroids over document embeddings from an existing index, 
then jointly optimizes these centroids and the query encoder.

Training
-----------------------------------------------------

JPQ training requires:

- A :class:`~pyterrier_dr.BiEncoder`
- A :class:`~pyterrier_dr.FlexIndex` containing document embeddings produced by the :class:`~pyterrier_dr.BiEncoder`
- A list of training examples, where each item is a ``dict`` containing
    - ``query`` (str): the input query text
    - ``doc_id_a`` (str): document identifier for a relevant (positive) document
    - ``doc_id_b`` (str): document identifier for a non-relevant (negative) document


:class:`pyterrier_dr.JPQTrainer` implements the training code.
Call :meth:`~pyterrier_dr.JPQTrainer.fit` to train JPQ. After training, call :meth:`~pyterrier_dr.JPQTrainer.jpq_index` 
to save the :class:`~pyterrier_dr.JPQIndex`. See example below:

.. code-block:: python
    :caption: Example for training JPQ with E5

    from pyterrier_dr import FlexIndex, JPQTrainer, E5

    index = FlexIndex("path/to/e5_index")
    model = E5()
    trainer = JPQTrainer(model, index, M=96, nbits=8, pq_impl="faiss2opq")

    training_docpairs = [
        {
            "query": "chemical reactions in water",
            "doc_id_a": "doc_1",
            "doc_id_b": "doc_2",
        },
        # ...
    ]
    save_path = "e5_jpq"
    trainer.fit(training_docpairs=training_docpairs)
    jpq_index = trainer.jpq_index(save_path)
    trainer.query_encoder.model.save_pretrained(save_path)



Retrieval
-----------------------------------------------------

JPQ retrieval requires:

- The fine-tuned query encoder
- The corresponding :class:`~pyterrier_dr.JPQIndex`

With a :class:`~pyterrier_dr.JPQIndex`, you can create a PQ retriever by calling
:meth:`~pyterrier_dr.JPQIndex.retriever_pq`. An example is provided below:

.. code-block:: python
    :caption: Example for retrieval using JPQ (E5)

    from pyterrier_dr import JPQIndex, E5
    from sentence_transformers import SentenceTransformer

    path = "e5_jpq"

    model = E5()
    model.model = SentenceTransformer(path, device="cuda")
    jpq_index = JPQIndex(path)

    pipeline = model.query_encoder() >> jpq_index.retriever_pq()
    results = pipeline.search("chemical reactions in water")


API Reference
-----------------------------------------------------

.. autoclass:: pyterrier_dr.JPQTrainer
    :members: fit, jpq_index

.. autoclass:: pyterrier_dr.JPQIndex
    :members: docnos, codes, dvecs, opq, retriever_pq, retriever_flat, retriever_prune, build_zero_shot_index
