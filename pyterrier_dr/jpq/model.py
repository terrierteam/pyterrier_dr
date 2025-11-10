import numpy as np
import torch
import torch.nn as nn
import pyterrier_dr

class QueryEncoder(nn.Module):
    """Frozen wrapper that adapts a pyterrier_dr-style model into a query encoder.

    The wrapped `dr_model` must implement either:
      - `encode_queries(texts: list[str], batch_size: int) -> np.ndarray`, or
      - `query_encoder(batch_size=...).encode(texts, batch_size=...) -> np.ndarray`

    Attributes
    ----------
    dr : object
        The wrapped dense retriever model.
    normalise : bool
        Whether the output vectors are normalised.
    batch_size : int
        Default batch size for encoding.

    Methods
    -------
    encode_texts(texts: list[str], batch_size: int | None = None) -> torch.Tensor
        Encodes a list of text strings into a float32 tensor of shape (N, D).
        Automatically batches inputs and optionally normalises vectors.
    """
    def __init__(
        self,
        dr_model : pyterrier_dr.BiEncoder,
        batch_size: int = 64
    ) -> None:
        super().__init__()
        self.dr = dr_model
        self.batch_size = batch_size

        # Make it clear this module is frozen
#        for p in self.parameters():
#            p.requires_grad = False
#        self.eval()

#    def parameters(self):  # type: ignore[override]
#        # No trainable params; return an empty iterator
#        return iter(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dr(x) # type: ignore

    def encode_texts_torch(self, texts: list[str], batch_size: int | None = None) -> torch.Tensor:
        """Encode a list of texts into a Torch tensor (float32), optionally L2-normalised."""
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)
        return self.dr.encode_queries_torch(texts, batch_size=batch_size) # type: ignore

        # bs = batch_size or self.batch_size
        # outputs: list[torch.Tensor] = []

        # for i in range(0, len(texts), bs):
        #     chunk = texts[i:i + bs]

        #     arr = self.dr.encode_queries_torch(chunk, batch_size=bs)
        #     # # Try direct API first
        #     # if hasattr(self.dr, "encode_queries"):
        #     #     arr = self.dr.encode_queries(chunk, batch_size=bs)
        #     # else:
        #     #     # Fallback: nested query_encoder().encode(...)
        #     #     qe = getattr(self.dr, "query_encoder", lambda **kw: None)(batch_size=bs)
        #     #     if qe is None or not hasattr(qe, "encode"):
        #     #         raise RuntimeError("dr_model must expose `encode_queries` or `query_encoder().encode`")
        #     #     arr = qe.encode(chunk, batch_size=bs)

        #     t = torch.from_numpy(np.asarray(arr, dtype=np.float32, order="C"))
        #     outputs.append(t)

        # return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int | None = None) -> torch.Tensor:
        """Encode a list of texts into a Torch tensor (float32), optionally L2-normalised."""
        if not texts:
            return torch.empty((0, 0), dtype=torch.float32)

        bs = batch_size or self.batch_size
        outputs: list[torch.Tensor] = []

        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]

            # Try direct API first
            if hasattr(self.dr, "encode_queries"):
                arr = self.dr.encode_queries(chunk, batch_size=bs)
            else:
                # Fallback: nested query_encoder().encode(...)
                qe = getattr(self.dr, "query_encoder", lambda **kw: None)(batch_size=bs)
                if qe is None or not hasattr(qe, "encode"):
                    raise RuntimeError("dr_model must expose `encode_queries` or `query_encoder().encode`")
                arr = qe.encode(chunk, batch_size=bs)

            t = torch.from_numpy(np.asarray(arr, dtype=np.float32, order="C"))
            outputs.append(t)

        return torch.cat(outputs, dim=0)


class PassageEncoder(nn.Module):
    """
    A PyTorch module that reconstructs dense passage embeddings from PQ codes
    using a JPQ-style product quantiser.

    Each of the M sub-quantisers is represented as an `nn.Embedding` layer,
    initialised from pre-trained PQ centroids. During training, these embeddings
    can be fine-tuned to optimise retrieval quality.

    Attributes
    ----------
    M : int
        Number of sub-quantisers.
    k : int
        Number of centroids per sub-quantiser.
    dsub : int
        Dimensionality of each sub-embedding.
    sub_embeddings : nn.ModuleList[nn.Embedding]
        The list of embedding tables, one per sub-quantiser.

    Methods
    -------
    forward(doc_codes: torch.Tensor) -> torch.Tensor
        Converts integer PQ codes of shape (B, M) into reconstructed dense
        embeddings of shape (B, M * dsub) by concatenating sub-embeddings.
    """
    def __init__(self, M, ksub, dsub, cents):
        super().__init__()
        self.M, self.k, self.dsub = M, ksub, dsub
        self.sub_embeddings = nn.ModuleList(
            [nn.Embedding(self.k, self.dsub) for _ in range(self.M)]
        )
        for i in range(self.M):
            self.sub_embeddings[i].weight.data.copy_(torch.from_numpy(cents[i]).float()) # type: ignore

    def forward(self, doc_codes: torch.Tensor) -> torch.Tensor:
        assert doc_codes.dim() == 2
        parts = [self.sub_embeddings[i](doc_codes[:, i]) for i in range(self.M)]
        return torch.cat(parts, dim=1)


class JPQBiencoder(nn.Module):
    """
    A simple container class bundling a `QueryEncoder` and a `PassageEncoder`
    into a single bi-encoder model.

    This class does not implement training or loss computation itself; it
    simply groups both encoders and provides a unified `.to(device)` method
    for device management.

    Attributes
    ----------
    query : QueryEncoder
        Query encoder submodule.
    passage : PassageEncoder
        Passage encoder submodule.

    Methods
    -------
    to(device: str) -> JPQBiencoder
        Moves both the query and passage encoders to the specified device
        and returns `self`.
    """
    def __init__(self, query_encoder: QueryEncoder, passage_encoder: PassageEncoder):
        super().__init__()
        self.query = query_encoder
        self.passage = passage_encoder

    def forward(self):
        raise NotImplementedError(
            "JPQBiencoder is just a container; call "
            "self.query(...) or self.passage(...), not the model directly."
        )

#    def to(self, device: str):
#        self.query = self.query.to(device)
#        self.passage = self.passage.to(device)
#        return self



