Dense Pseudo-Relevance Feedback
===============================

.. related:: pyterrier_dr.AveragePrf
.. related:: pyterrier_dr.VectorPrf

Dense Pseudo Relevance Feedback (PRF) is a technique to improve the performance of a retrieval system by expanding the
original query vector with the vectors from the top-ranked documents. The classical idea is that the top-ranked documents 
can help to create a more accurate query formulation. In dense retrieval terms, we take the document vectors of the 
top-ranked documents, and use these for a further round of retrieval. The revised query vectors can also take 
into account the original query vector, to prevent too much "query-drift".

PyTerrier-DR provides two dense PRF implementations: :class:`~pyterrier_dr.AveragePrf` and :class:`~pyterrier_dr.VectorPrf`.
