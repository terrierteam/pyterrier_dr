from pyterrier_dr.jpq.model import PassageEncoder, QueryEncoder


import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # ΔDCG approximated via pairwise differences in DCG
        delta_ndcg = (dcg.unsqueeze(1) - dcg.unsqueeze(0))

        # pairwise differences in predicted scores
        diff_s = s.unsqueeze(1) - s.unsqueeze(0)
        diff_y = y.unsqueeze(1) - y.unsqueeze(0)

        # only pairs with different relevance matter
        pos_pairs = diff_y > 0

        # logistic loss weighted by ΔNDCG
        pair_loss = torch.log1p(torch.exp(-sigma * diff_s)) * delta_ndcg.clamp(min=0) * pos_pairs.float()
        loss += pair_loss.sum()

    return loss / batch_size


def lambdarank_nic(s: torch.Tensor, not_ranks, y: torch.Tensor, sigma: float = 1.0, eps: float = 1e-12):
    
    B, N = s.shape
    # --------- Gains and ideal DCG (for NDCG normalization) ----------
    G = (2.0 ** y) - 1.0  # [B, N]

    # Ideal ordering by label (desc)
    ideal_y, _ = torch.sort(y, dim=1, descending=True)       # [B, N]
    ideal_G = (2.0 ** ideal_y) - 1.0                         # [B, N]

    # Discount for ranks 1..N (shared across batch)
    i = torch.arange(1, N + 1, dtype=torch.float32)  # [N]
    D = 1.0 / torch.log2(1.0 + i)                    # [N]
    iDCG = (ideal_G * D.unsqueeze(0)).sum(dim=1)     # [B]
    
    # --------- Current predicted positions & discounts ---------------
    # Order by predicted scores (desc)
    _, order = torch.sort(s, dim=1, descending=True)  # [B, N]
    # positions (0-based): inverse permutation
    i = torch.empty_like(order)
    i.scatter_(1, order, torch.arange(N).unsqueeze(0).expand(B, -1))
    # convert to 1-based ranks
    curr_rank = 1.0 + i                                      # [B, N]
    D = 1.0 / torch.log2(1.0 + curr_rank)            # [B, N]

    # --------- Pairwise tensors -------------------------------------
    # masks for ordered relevance pairs: y_i > y_j
    y_i = y.unsqueeze(2)  # [B, N, 1]
    y_j = y.unsqueeze(1)  # [B, 1, N]
    pair_mask = (y_i > y_j)    # [B, N, N]

    # score differences s_i - s_j
    s_i = s.unsqueeze(2)
    s_j = s.unsqueeze(1)
    s_diff = s_i - s_j

    # ΔNDCG swap weight = |(G_i - G_j)| * |Disc(p_i) - Disc(p_j)| / IDCG
    G_i = G.unsqueeze(2)
    G_j = G.unsqueeze(1)
    delta_G = (G_i - G_j).abs()

    D_i = D.unsqueeze(2)
    D_j = D.unsqueeze(1)
    delta_D = (D_i - D_j).abs()

    idcg_safe = iDCG.clamp_min(eps).unsqueeze(1).unsqueeze(2)  # [B,1,1]
    delta_NDCG = (delta_G * delta_D) / idcg_safe            # [B, N, N]

    # pairwise logistic loss with ΔNDCG weight
    pairwise = F.softplus(-sigma * s_diff) * delta_NDCG

    # apply mask and average per list by number of valid pairs
    pairwise = pairwise * pair_mask
    per_list_pairs = pair_mask.sum(dim=(1, 2)).clamp_min(1)
    per_list_loss = pairwise.sum(dim=(1, 2)) / per_list_pairs

    return per_list_loss.mean() 

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
    diff_dcg = (dcg.unsqueeze(2) - dcg.unsqueeze(1))  # [B, num_docs, num_docs]
    diff_labels = labels.unsqueeze(2) - labels.unsqueeze(1)     # [B, num_docs, num_docs]

    # Only consider pairs where i is more relevant than j
    pos_pairs = (diff_labels > 0).float()                # [B, num_docs, num_docs]

    num_pos_pairs = pos_pairs.sum().item()
    print("Num positive pairs per batch:", num_pos_pairs)
    print("ΔDCG mean for pospairs:", diff_dcg[(diff_labels > 0)].mean().item())

    # Logistic pairwise loss weighted by ΔDCG
    pair_loss = torch.log1p(torch.exp(-sigma * diff_s)) * diff_dcg.abs() * pos_pairs

    # Sum over pairs and average over batch
    loss = (pair_loss.sum(dim=(1,2)) / (pos_pairs.sum(dim=(1,2)) + 1e-8)).mean()

    print("ΔDCG mean all", diff_dcg.mean().item())
    print("Pair loss mean", pair_loss.mean().item())
    print("Num positive pairs", pos_pairs.sum().item())

    with torch.no_grad():
        corr = torch.corrcoef(torch.stack([scores.mean(dim=1), -ranks.float().mean(dim=1)], dim=0))[0, 1]
        print("Correlation(scores, -ranks):", corr.item())    

    return loss


class JPQCELossJPQNegsLambdaRank(nn.Module):
    """
    CE loss with JPQ negatives using LambdaRank.
    IBNs are NOT used here (we dont currently have their ranks)
    """
    def __init__(self, query_encoder, passage_encoder, use_inbatch_negatives=True, jpq_negs=True):
        super().__init__()
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.use_inbatch_negatives = use_inbatch_negatives
        self.jpq_negs = jpq_negs
        print("Using LambdaRank loss: JPQ negatives =", jpq_negs, ", IBNs =", use_inbatch_negatives)

    def forward(self, batch):
        device = next(self.passage_encoder.parameters()).device
        B = len(batch["query_text"])

        # 1. Encode queries
        q = self.query_encoder.encode_texts_torch(batch["query_text"]).to(device)  # [B, D]

        # 2. Encode positives
        pos = self.passage_encoder(batch["pos_codes"].to(device))  # [B, D]
        rank_pos = batch["pos_ranks"].to(device).view(B, 1)

        # 3. Encode negatives (if present)
        if not "neg_codes" in batch and "neg_ranks" in batch:
            raise ValueError("LambdaRank requires explicit negatives with known ranks.")

        if self.jpq_negs: # jpq_negs from the last epoch
            assert "neg_jpq_codes" in batch
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

        # Optionally add in-batch negatives
        if self.use_inbatch_negatives:
            # Essentially, the positive embeddings of other queries are treated as fixed negatives
            # — we don’t want their gradient flowing into those queries’ embeddings.
            with torch.no_grad():
                all_pos_as_negs = pos.unsqueeze(0).repeat(B, 1, 1)  # [B, B, D]
                mask_self = ~torch.eye(B, dtype=torch.bool, device=device)
                inbatch_negs = all_pos_as_negs[mask_self].view(B, B - 1, -1)
                # we assume in-batch negatives are ranked very low (e.g., 100)
                rank_inbatch = torch.full((B, B - 1), 100, device=device)

            # Concatenate all negatives
            all_negs = torch.cat([neg, inbatch_negs], dim=1)
            all_neg_ranks = torch.cat([rank_negs, rank_inbatch], dim=1)
        else:
            all_negs = neg
            all_neg_ranks = rank_negs
        
        print("rank_pos:", rank_pos.float().mean().item())
        print("rank_negs:", all_neg_ranks.float().mean().item())
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
        neg_scores = torch.einsum("bd,bnd->bn", q, all_negs)
        # alternative: neg_scores = torch.matmul(q.unsqueeze(1), neg.transpose(1,2)).squeeze(1)

        print("pos_scores:", pos_scores.mean().item())
        print("neg_scores:", neg_scores.mean().item())

        # 5. Concatenate scores, ranks, labels
        scores = torch.cat([pos_scores, neg_scores], dim=1)   # [B, 1 + N]
        ranks = torch.cat([rank_pos, rank_negs], dim=1)      # [B, 1 + N]
        labels = torch.cat([
            torch.ones((B, 1), device=device),
            torch.zeros((B, N), device=device)
        ], dim=1)

        # 6. Compute loss
        loss = lambdarank_fixed_ranks_vectorized(scores, ranks, labels, sigma=1)

        ce_loss = -torch.log_softmax(torch.cat([pos_scores, neg_scores], dim=1), dim=1)[:, 0].mean()
        print("Scores mean:", scores.mean().item(), "std:", scores.std().item())

        print(f"LambdaRank loss: {loss.item():.4f}, CE loss: {ce_loss.item():.4f}")
        return loss