"""RepCONC (Zhan et al., WSDM'22): dense retrieval with learnable discrete
representations trained via constrained (Sinkhorn) clustering.

See https://arxiv.org/abs/2110.05789 and the module docstrings for the
paper references and documented deviations from the official code.
"""

from .index import RepCONCIndex
from .train import RepCONCTrainer
from .retriever import RepCONCRetriever, RepCONCRetrieverPQ, RepCONCRetrieverFlat

__all__ = ["RepCONCTrainer", "RepCONCIndex", "RepCONCRetriever", "RepCONCRetrieverPQ", "RepCONCRetrieverFlat"]
