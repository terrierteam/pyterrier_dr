"""Model container for RepCONC: a :class:`~pyterrier_dr.biencoder.BiEncoder`
plus a :class:`~pyterrier_dr.repconc.pq.RepCONCPQ`.

The BiEncoder is an external object (a PyTerrier ``Transformer`` wrapping one
or more torch modules), so it is *referenced*, not embedded as a child
module. Its internal torch modules are enumerated for the optimizer and for
checkpointing (see :mod:`pyterrier_dr.repconc.checkpointing`).
"""

from typing import List, Tuple

import torch
from torch import nn

from pyterrier_dr.biencoder import BiEncoder

from .pq import RepCONCPQ

__all__ = ["RepCONCModel"]


class RepCONCModel(nn.Module):
    """Pairs a :class:`BiEncoder` with a learnable :class:`RepCONCPQ`.

    Encodings produced by this module are in the *rotated* space (queries and
    documents are rotated by the PQ's fixed rotation), in which dot-product
    similarity equals similarity in the original space.

    Args:
        biencoder: the query/document encoder (must implement
            ``encode_queries_torch`` and ``encode_docs_torch`` for training).
        pq: the learnable product quantizer.
    """

    def __init__(self, biencoder: BiEncoder, pq: RepCONCPQ):
        super().__init__()
        self.pq = pq
        self._biencoder = biencoder

    @property
    def biencoder(self) -> BiEncoder:
        return self._biencoder

    # -- biencoder module discovery -----------------------------------------

    def encoder_modules(self) -> List[Tuple[str, nn.Module]]:
        """The torch modules making up the biencoder.

        Discovers ``nn.Module`` attributes of the biencoder (e.g.
        ``model``, ``query_encoder_model``, ``doc_encoder_model``), deduplicated
        by object identity (models that share one module for queries and docs
        appear once). Modules that have not been instantiated yet (lazy
        loaders) must be materialised first, e.g. by an encode call.
        """
        seen = {}
        for name, value in vars(self._biencoder).items():
            if isinstance(value, nn.Module) and id(value) != id(self.pq):
                seen[id(value)] = (name, value)
        return list(seen.values())

    def encoder_parameters(self) -> List[torch.nn.Parameter]:
        """All (unique) parameters of the biencoder's torch modules."""
        params = []
        seen = set()
        for _, module in self.encoder_modules():
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    params.append(p)
        return params

    # -- encoders (rotated space) -------------------------------------------

    def encode_queries(self, texts, batch_size: int | None = None) -> torch.Tensor:
        """Encode query texts to rotated continuous embeddings (with grad)."""
        return self.pq.rotate(self._biencoder.encode_queries_torch(list(texts), batch_size=batch_size))

    def encode_docs(self, texts, batch_size: int | None = None) -> torch.Tensor:
        """Encode document texts to rotated continuous embeddings (with grad)."""
        return self.pq.rotate(self._biencoder.encode_docs_torch(list(texts), batch_size=batch_size))

    # -- train/eval propagation ---------------------------------------------

    def set_train(self, mode: bool = True) -> "RepCONCModel":
        """Set train/eval mode on this module *and* the biencoder's modules."""
        super().train(mode)
        for _, module in self.encoder_modules():
            module.train(mode)
        return self
