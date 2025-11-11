from pyterrier_dr.jpq.model import PassageEncoder, QueryEncoder


import torch
import torch.nn as nn


class JPQCELoss(nn.Module):
    """
    A pairwise contrastive loss module for JPQ-based bi-encoder training.
    Does not support in-batch negatives.

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
        self.loss_f = nn.CrossEntropyLoss()
        print("Using CE loss")

    def forward(self, batch):
        # Find the device from the passage encoder (the model we are training)
        device = next(self.passage_encoder.parameters()).device
        # Encode queries on CPU, then move to the same device
        q = self.query_encoder.encode_texts_torch(batch["query_text"])
        q = q.to(device)
        # Bring positive/negative PQ codes to same device
        pos = batch["pos_codes"].to(device)
        neg = batch["neg_codes"].to(device)
        # Reconstruct document embeddings on the passage encoder’s device
        pos = self.passage_encoder(pos)
        neg = self.passage_encoder(neg)
        # Compute dot products
        s_pos = torch.sum(q * pos, dim=-1)  # dot product per sample
        s_neg = torch.sum(q * neg, dim=-1)
        # Stack [s_pos, s_neg] and create 0 labels for CrossEntropy
        scores = torch.stack([s_pos, s_neg], dim=1) # [batch, 2]
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=device) # positive score as position 0 in scores

        return self.loss_f(scores, labels)


class JPQCELossInBatchNegs(nn.Module):
    """
    CE loss with in-batch negatives and optional jpq negatives.
    """
    def __init__(self, query_encoder, passage_encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.loss_f = nn.CrossEntropyLoss()
        print("Using CE loss with IBNs")

    def forward(self, batch):
        device = next(self.passage_encoder.parameters()).device
        B = len(batch["query_text"])

        # 1. Encode queries
        q = self.query_encoder.encode_texts_torch(batch["query_text"]).to(device)  # [B, D]

        # 2. Encode positives
        pos = self.passage_encoder(batch["pos_codes"].to(device))  # [B, D]

        # 3. Encode negatives (if present)
        if "neg_codes" in batch:

            if "neg_jpq_codes" in batch: # jpq_negs from the last epoch
                all_neg_codes = torch.cat((batch["neg_codes"].unsqueeze(1), batch["neg_jpq_codes"]), dim=1) # [B, N_total, D_code]
                B, N, D_code = all_neg_codes.shape
                # Flatten negatives for similarity computation
                neg = self.passage_encoder(all_neg_codes.view(B*N, D_code).to(device))  # [B * N_total, D]
                neg = neg.view(B, N, -1)  # [B, N_total, D]
            else:
                neg = self.passage_encoder(batch["neg_codes"].to(device)).unsqueeze(1)  # [B, 1, D]
                # we only have ONE negative per batch, so hack in a unsqueeze here, to fit rest of code.

            # 4. Compute similarity scores
            # a) positives & in-batch negatives
            pos_scores = torch.matmul(q, pos.T)  # [B, B]

            # b) explicit negatives, applied for all queries
            neg_flat = neg.reshape(B * N, -1)     # [B*N, D]
            neg_scores = q @ neg_flat.T           # [B, B*N]

            # 5. Concatenate all scores
            scores = torch.cat([pos_scores, neg_scores], dim=1)  # [B, B + B*N]
        else:
            # only in-batch negatives
            scores = torch.matmul(q, pos.T)  # [B, B]

        # 6. Label for each query = its own positive index
        labels = torch.arange(B, dtype=torch.long, device=device)

        # 7. Compute loss
        return self.loss_f(scores, labels)


def lambdarank_fixed_ranks(scores, ranks, labels, sigma=1.0):
    """
    scores: [batch_size, num_docs] - model similarity scores
    ranks: [batch_size, num_docs] - integer global ranks (1 = top)
    labels: [batch_size, num_docs] - relevance labels (e.g., 1 or 0)
    """
    batch_size, num_docs = scores.shape
    loss = 0.0

    for b in range(batch_size):
        s = scores[b]
        r = ranks[b]
        y = labels[b]

        # Compute gain and discount for NDCG weighting
        gain = (2 ** y - 1.0)
        discount = 1.0 / torch.log2(2.0 + r.float())
        dcg = gain * discount

        # ΔNDCG approximated via pairwise differences in DCG
        delta_ndcg = torch.abs(dcg.unsqueeze(1) - dcg.unsqueeze(0))

        # pairwise differences in predicted scores
        diff_s = s.unsqueeze(1) - s.unsqueeze(0)
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)

        # only pairs with different relevance matter
        pos_pairs = diff_y > 0

        # logistic loss weighted by ΔNDCG
        pair_loss = torch.log1p(torch.exp(-sigma * diff_s)) * delta_ndcg * pos_pairs.float()
        loss += pair_loss.sum()

    return loss / batch_size


def lambdarank_fixed_ranks_vectorized(scores, ranks, labels, sigma=1.0):
    """
    Vectorized LambdaRank loss (no Python loops)
    scores: [B, num_docs]
    ranks: [B, num_docs]  (global ranks)
    labels: [B, num_docs] (binary or graded relevance)
    """
    # Compute DCG gain and discount
    gain = (2 ** labels - 1).float()                     # [B, num_docs]
    discount = 1.0 / torch.log2(2.0 + ranks.float())     # [B, num_docs]
    dcg = gain * discount                                 # [B, num_docs]

    # Pairwise differences
    diff_s = scores.unsqueeze(2) - scores.unsqueeze(1)   # [B, num_docs, num_docs]
    diff_dcg = torch.abs(dcg.unsqueeze(2) - dcg.unsqueeze(1))  # [B, num_docs, num_docs]
    diff_labels = labels.unsqueeze(2) - labels.unsqueeze(1)     # [B, num_docs, num_docs]

    print(diff_dcg.abs().mean().item())


    # Only consider pairs where i is more relevant than j
    pos_pairs = (diff_labels > 0).float()                # [B, num_docs, num_docs]

    # Logistic pairwise loss weighted by ΔDCG
    pair_loss = torch.log1p(torch.exp(-sigma * diff_s)) * diff_dcg * pos_pairs

    # Sum over pairs and average over batch
    loss = pair_loss.sum(dim=(1,2)).mean()

    print("ΔNDCG mean", diff_dcg.mean().item())
    print("Pair loss mean", pair_loss.mean().item())
    print("Num positive pairs", pos_pairs.sum().item())

    return loss


class JPQCELossJPQNegsLambaRank(nn.Module):
    """
    CE loss with JPQ negatives using LambdaRank.
    IBNs are NOT used here (we dont currently have their ranks)
    """
    def __init__(self, query_encoder, passage_encoder):
        super().__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        print("Using LambdaRank loss with JPQ negatives")

    def forward(self, batch):
        device = next(self.passage_encoder.parameters()).device
        B = len(batch["query_text"])

        # 1. Encode queries
        q = self.query_encoder.encode_texts_torch(batch["query_text"]).to(device)  # [B, D]

        # 2. Encode positives
        pos = self.passage_encoder(batch["pos_codes"].to(device))  # [B, D]
        rank_pos = batch["pos_ranks"].to(device).view(B, 1)

        # 3. Encode negatives (if present)
        if not "neg_codes" and "neg_ranks" in batch:
            raise ValueError("LambdaRank requires explicit negatives with known ranks.")

        if "neg_jpq_codes" in batch: # jpq_negs from the last epoch
            # need an unsqueeze here for the original neg_codes, as therse is only one per query
            all_neg_codes = torch.cat((batch["neg_codes"].unsqueeze(1), batch["neg_jpq_codes"]), dim=1) # [B, N_total, D_code]
            B, N, D_code = all_neg_codes.shape
            neg = self.passage_encoder(all_neg_codes.view(B*N, D_code).to(device))  # [B * N_total, D]
            neg = neg.view(B, N, -1)  # [B, N_total, D]
            rank_negs = torch.cat((batch["neg_ranks"].unsqueeze(1), batch["neg_jpq_ranks"]), dim=1).to(device)  # [B, N_neg_total]
        else:
            neg = self.passage_encoder(batch["neg_codes"].to(device))  # [B, D]
            rank_negs = batch["neg_ranks"].to(device)  # [B, N=1]
            if rank_negs.dim() == 1:
                rank_negs = rank_negs.view(B, -1)  # reshape to [B, N] if needed
            N = 1

        # sanity shapes
        assert q.dim() == 2 and pos.dim() == 2 and neg.dim() == 3
        Bq, Dq = q.shape
        Bp, Dp = pos.shape
        Bn, N, Dn = neg.shape
        assert Bq == Bp == Bn == B, f"Batch mismatch: {Bq},{Bp},{Bn}"
        assert Dq == Dp == Dn, f"Embedding dim mismatch: {Dq},{Dp},{Dn}"

        # 4. Compute similarities (per-query)
        # positive: [B, 1]
        pos_scores = (q * pos).sum(dim=1, keepdim=True)

        # negatives: [B, N]  (each query against its own N negatives)
        neg_scores = torch.einsum("bd,bnd->bn", q, neg)
        # alternative: neg_scores = torch.matmul(q.unsqueeze(1), neg.transpose(1,2)).squeeze(1)

        # 5. Concatenate scores, ranks, labels
        scores = torch.cat([pos_scores, neg_scores], dim=1)   # [B, 1 + N]
        ranks = torch.cat([rank_pos, rank_negs], dim=1)      # [B, 1 + N]
        labels = torch.cat([
            torch.ones((B, 1), device=device),
            torch.zeros((B, N), device=device)
        ], dim=1)

        # 6. Compute loss
        loss = lambdarank_fixed_ranks_vectorized(scores, ranks, labels, sigma=0.1)

        ce_loss = -torch.log_softmax(torch.cat([pos_scores, neg_scores], dim=1), dim=1)[:, 0].mean()
        print(f"LambdaRank loss: {loss.item():.4f}, CE loss: {ce_loss.item():.4f}")
        return loss