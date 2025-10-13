import numpy as np
import torch
import torch.nn as nn

class QueryEncoder(nn.Module):
    """Frozen wrapper that adapts a pyterrier_dr-style model into a query encoder.

    The wrapped `dr_model` must implement either:
      - `encode_queries(texts: list[str], batch_size: int) -> np.ndarray`, or
      - `query_encoder(batch_size=...).encode(texts, batch_size=...) -> np.ndarray`

    Args:
        dr_model: Underlying dense retrieval model.
        normalise: If True, L2-normalise output vectors row-wise.
        batch_size: Default batch size used inside `encode_texts`.
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
        self.batch = batch_size

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

        bs = batch_size or self.batch
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
    """Expose a JPQ embedding model for passage encoding as a Pytorch module"""
    def __init__(self, M, ksub, dsub, cents):
        super().__init__()
        self.M, self.k, self.dsub = M, ksub, dsub
        self.sub_embeddings = nn.ModuleList(
            [nn.Embedding(self.k, self.dsub) for _ in range(self.M)]
        )
        for i in range(self.M):
            self.sub_embeddings[i].weight.data.copy_(torch.from_numpy(cents[i]).float())

    def forward(self, doc_codes: torch.Tensor) -> torch.Tensor:
        parts = [self.sub_embeddings[i](doc_codes[:, i]) for i in range(self.M)]
        return torch.cat(parts, dim=1)


class JPQLoss(nn.Module):
    """Expose the JPQ loss as a Pytorch module"""
    def __init__(self, query_encoder: QueryEncoder, passage_encoder: PassageEncoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch):
        # Find the device from the passage encoder (the model we are training)
        device = next(self.passage_encoder.parameters()).device
        # Encode queries on CPU, then move to the same device
        with torch.no_grad():
            q = self.query_encoder.encode_texts(batch["query_text"])
        q = q.to(device)
        # Bring positive/negative PQ codes to same device
        pos = batch["pos_codes"].to(device) #, non_blocking=True ONLY use if pin_memory=True is set for DataLoader
        neg = batch["neg_codes"].to(device) #, non_blocking=True

        # Reconstruct document embeddings on the passage encoder’s device
        pos = self.passage_encoder(pos)
        neg = self.passage_encoder(neg)
        # Compute cosine similarities
        s_pos = self.cos_sim(q, pos)
        s_neg = self.cos_sim(q, neg)
        # Stack [s_pos, s_neg] and create 0 labels for CrossEntropy
        scores = torch.stack([s_pos, s_neg], dim=1)
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=device)

        return self.loss_fct(scores, labels)


class JPQBiencoder:
    """A biencoder using JPQ to embed documents"""
    def __init__(self, query_encoder: QueryEncoder, passage_encoder: PassageEncoder):
        self.query = query_encoder
        self.passage = passage_encoder

    def to(self, device: str):
        self.query = self.query.to(device)
        self.passage = self.passage.to(device)
        return self
    