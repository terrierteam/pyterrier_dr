import numpy as np
import torch
import torch.nn as nn

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
        dr_model, 
        normalise: bool = True, 
        batch_size: int = 64
    ) -> None:
        super().__init__()
        self.dr = dr_model
        self.normalise = normalise
        self.batch_size = batch_size

        # Make it clear this module is frozen
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def parameters(self):  # type: ignore[override]
        # No trainable params; return an empty iterator
        return iter(())

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
            if self.normalise:
                t /= (t.norm(dim=1, keepdim=True) + 1e-12)
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
        parts = [self.sub_embeddings[i](doc_codes[:, i]) for i in range(self.M)]
        return torch.cat(parts, dim=1)


class JPQLoss(nn.Module):
    """
    A pairwise contrastive loss module for JPQ-based bi-encoder training.

    Given a query encoder and a passage encoder, this module computes a
    two-class cross-entropy loss encouraging the query to be closer to its
    positive passage than to its negative passage.

    The loss operates on cosine similarities between the query vector and
    reconstructed passage embeddings.

    Attributes
    ----------
    query_encoder : QueryEncoder
        A frozen query encoder that provides normalised query embeddings.
    passage_encoder : PassageEncoder
        A learnable passage encoder that reconstructs passage embeddings
        from PQ codes.
    cos_sim : nn.CosineSimilarity
        Computes cosine similarity between query and passage embeddings.
    loss_f : nn.CrossEntropyLoss
        Computes binary classification loss on stacked positive/negative scores.

    Methods
    -------
    forward(batch: dict[str, torch.Tensor]) -> torch.Tensor
        Computes the JPQ loss given a batch containing:
            - "query_text" : list[str] or list of text queries,
            - "pos_codes"  : torch.LongTensor of PQ codes for positive docs,
            - "neg_codes"  : torch.LongTensor of PQ codes for negative docs.
        Returns a scalar loss value.
    """
    def __init__(self, query_encoder: QueryEncoder, passage_encoder: PassageEncoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.loss_f = nn.CrossEntropyLoss()

    def forward(self, batch):
        # Find the device from the passage encoder (the model we are training)
        device = next(self.passage_encoder.parameters()).device
        # Encode queries on CPU, then move to the same device
        with torch.no_grad():
            q = self.query_encoder.encode_texts(batch["query_text"])
        q = q.to(device)
        # Bring positive/negative PQ codes to same device
        pos = batch["pos_codes"].to(device)
        neg = batch["neg_codes"].to(device)
        # Reconstruct document embeddings on the passage encoder’s device
        pos = self.passage_encoder(pos)
        neg = self.passage_encoder(neg)
        # Compute cosine similarities
        s_pos = self.cos_sim(q, pos)
        s_neg = self.cos_sim(q, neg)
        # Stack [s_pos, s_neg] and create 0 labels for CrossEntropy
        scores = torch.stack([s_pos, s_neg], dim=1)
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=device)

        return self.loss_f(scores, labels)


class JPQBiencoder:
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
        self.query = query_encoder
        self.passage = passage_encoder

    def to(self, device: str):
        self.query = self.query.to(device)
        self.passage = self.passage.to(device)
        return self
    