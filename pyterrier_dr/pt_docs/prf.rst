Dense Pseudo-Relevance Feedback
===============================

.. related:: pyterrier_dr.AveragePrf
.. related:: pyterrier_dr.VectorPrf

Dense Pseudo Relevance Feedback (PRF) is a technique to improve the performance of a retrieval system by expanding the
original query vector with the vectors from the top-ranked documents. The idea is that the top-ranked documents.

PyTerrier-DR provides two dense PRF implementations: :class:`~pyterrier_dr.AveragePrf` and :class:`~pyterrier_dr.VectorPrf`.
