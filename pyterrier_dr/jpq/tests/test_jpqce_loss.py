import unittest
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pyterrier_dr.jpq.losses import JPQCELoss

# ----- Fakes / Spies -----

class FakeQueryEncoder:
    """
    Minimal stand-in for QueryEncoder:
    - forward(texts) -> L2-normalized embeddings [B, D] on CPU
    - records how many times it's called
    """
    def __init__(self, vectors_by_text: Dict[str, torch.Tensor], l2_norm: bool = True):
        self.vectors_by_text = vectors_by_text
        self.l2_norm = l2_norm
        self.calls = 0

    def forward(self, texts: List[str]) -> torch.Tensor:
        self.calls += 1
        embs = []
        for t in texts:
            v = self.vectors_by_text[t].detach().clone().to(torch.float32)
            if self.l2_norm:
                v = F.normalize(v, p=2, dim=-1)
            embs.append(v)
        if not embs:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack(embs, dim=0)  # [B, D]


class FakePassageEncoder(nn.Module):
    """
    Minimal stand-in for PassageEncoder:
    - forward(codes: LongTensor [B]) -> embeddings [B, D]
      (embeds each code index via nn.Embedding)
    """
    def __init__(self, codebook_size: int, dim: int, device: torch.device):
        super().__init__()
        self.emb = nn.Embedding(codebook_size, dim)
        # init for test stability
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.5)
        self.to(device)

    def forward(self, codes: torch.LongTensor) -> torch.Tensor:
        return self.emb(codes)  # [B, D]


# ----- Test Suite -----

class TestJPQCELoss(unittest.TestCase):

    def setUp(self):
        # Deterministic
        torch.manual_seed(1234)
        np.random.seed(1234)

        self.device = torch.device("cpu")  # keep CPU to avoid CUDA assumptions
        self.D = 8
        self.V = 50  # codebook size

        # Create a small passage encoder
        self.passage = FakePassageEncoder(self.V, self.D, device=self.device)

        # Build a small batch of queries/positives/negatives
        self.B = 4
        self.pos_codes = torch.tensor([1, 2, 3, 4], dtype=torch.long)   # [B]
        self.neg_codes = torch.tensor([10, 11, 12, 13], dtype=torch.long)

        # Create query vectors that are close to the positives
        # Take the current passage embeddings as query targets (then L2-normalize)
        with torch.no_grad():
            pos_embs = self.passage(self.pos_codes).detach().cpu()  # [B, D]
        vectors_by_text = {f"q{i}": pos_embs[i].clone() for i in range(self.B)}
        self.query = FakeQueryEncoder(vectors_by_text, l2_norm=True)

        # JPQ loss module
        self.criterion = JPQCELoss(query_encoder=self.query, passage_encoder=self.passage) # type: ignore

        # Build the batch expected by JPQCELoss
        self.batch = {
            "query_text": [f"q{i}" for i in range(self.B)],
            "pos_codes": self.pos_codes.clone(),
            "neg_codes": self.neg_codes.clone(),
        }

    def test_loss_matches_manual_cross_entropy(self):
        """The JPQCELoss forward equals manual CE on stacked scores [s_pos, s_neg]."""
        loss = self.criterion(self.batch)

        # Manual computation using the same encoders
        with torch.no_grad():
            q = self.query.forward(self.batch["query_text"]).to(self.device)  # [B, D]
            pos = self.passage(self.batch["pos_codes"].to(self.device))                  # [B, D]
            neg = self.passage(self.batch["neg_codes"].to(self.device))                  # [B, D]
            s_pos = (q * pos).sum(dim=-1)                                                # [B]
            s_neg = (q * neg).sum(dim=-1)                                                # [B]
            scores = torch.stack([s_pos, s_neg], dim=1)                                  # [B, 2]
            labels = torch.zeros(self.B, dtype=torch.long, device=self.device)           # class 0 = positive
            manual = F.cross_entropy(scores, labels)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)
        self.assertAlmostEqual(loss.item(), manual.item(), places=6)

    def test_grad_flows_to_passage_encoder(self):
        """Backward pass should populate grads in passage encoder parameters."""
        loss = self.criterion(self.batch)
        self.passage.zero_grad(set_to_none=True)
        loss.backward()

        grads = [p.grad for p in self.passage.parameters() if p.requires_grad]
        self.assertTrue(len(grads) > 0, "No trainable params in passage encoder")
        self.assertTrue(any(g is not None for g in grads), "No gradients found")
        self.assertTrue(any(torch.isfinite(g).all().item() for g in grads), "NaN/Inf gradients found") # type: ignore

    def test_smaller_loss_when_positives_move_closer(self):
        """If we improve positive embeddings towards queries, the loss should decrease."""
        # Baseline loss
        base_loss = self.criterion(self.batch).item()

        # Nudge the positive embeddings towards the query vectors and recompute
        opt = torch.optim.SGD(self.passage.parameters(), lr=0.5)
        for _ in range(5):
            opt.zero_grad()
            loss = self.criterion(self.batch)
            loss.backward()
            opt.step()

        improved_loss = self.criterion(self.batch).item()
        self.assertLess(improved_loss, base_loss, "Loss did not decrease after optimization")

    def test_output_dtype_and_shape(self):
        loss = self.criterion(self.batch)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.shape, torch.Size([]))  # scalar

    def test_empty_batch_raises(self):
        """Empty batches should raise a clear error (CrossEntropy with B=0 is invalid)."""
        empty_batch = {"query_text": [], "pos_codes": torch.empty(0, dtype=torch.long),
                       "neg_codes": torch.empty(0, dtype=torch.long)}
        with self.assertRaises(Exception):
            _ = self.criterion(empty_batch)

    def test_query_encoder_called(self):
        """Ensure the loss uses the QueryEncoder path (forward)."""
        before = self.query.calls
        _ = self.criterion(self.batch)
        after = self.query.calls
        self.assertEqual(after, before + 1, "forward was not called exactly once")


if __name__ == "__main__":
    unittest.main(verbosity=2)