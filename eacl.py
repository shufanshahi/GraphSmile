import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionAnchoredContrastiveLoss(nn.Module):
    """L_sup from EACL: contrastive loss over combined utterance+anchor pool.

    Positive pairs = same emotion label (utterance↔utterance and utterance↔anchor).
    Anchors always ensure every class is represented, solving the no-positives problem.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, utterance_repr, anchors, labels):
        """
        Args:
            utterance_repr: (N, D) feat_fusion for current batch
            anchors: (n_classes, D) emotion anchor vectors
            labels: (N,) long — emotion label per utterance (0..n_classes-1)
        Returns:
            scalar L_sup
        """
        n_classes = anchors.size(0)

        # Combined pool V: utterances + anchors → (N + n_classes, D)
        V = torch.cat([utterance_repr, anchors], dim=0)
        anchor_labels = torch.arange(n_classes, device=labels.device)
        V_labels = torch.cat([labels, anchor_labels], dim=0)   # (N + n_classes,)

        V_norm = F.normalize(V, dim=-1)
        sim = torch.matmul(V_norm, V_norm.T) / self.temperature   # (M, M)

        # Positive mask: same label, excluding self
        label_row = V_labels.unsqueeze(1)
        label_col = V_labels.unsqueeze(0)
        pos_mask = (label_row == label_col).float()
        diag_mask = torch.eye(V.size(0), device=V.device)
        pos_mask = pos_mask - diag_mask

        num_pos = pos_mask.sum(1)   # (M,)

        denom_mask = 1.0 - diag_mask
        log_denom = torch.log((torch.exp(sim) * denom_mask).sum(1) + 1e-8)

        per_sample = -(pos_mask * (sim - log_denom.unsqueeze(1))).sum(1) / (num_pos + 1e-8)

        valid = (num_pos > 0).float()
        return (per_sample * valid).sum() / (valid.sum() + 1e-8)


class AnchorAngleLoss(nn.Module):
    """L_Ag from EACL: maximize minimum pairwise angle between emotion anchors.

    Minimizing this loss pushes anchors apart uniformly on the hypersphere,
    preventing similar-emotion clusters from collapsing together.
    """

    def forward(self, anchors):
        """
        Args:
            anchors: (n_classes, D) emotion anchor representations
        Returns:
            scalar L_Ag (negative mean of minimum pairwise angles)
        """
        n = anchors.size(0)
        if n < 2:
            return torch.tensor(0.0, device=anchors.device)

        a_norm = F.normalize(anchors, dim=-1)
        cos_sim = torch.matmul(a_norm, a_norm.T)

        # Clamp for arccos numerical stability
        cos_sim_clamped = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        angles = torch.acos(cos_sim_clamped)

        diag_mask = torch.eye(n, device=anchors.device).bool()
        angles = angles.masked_fill(diag_mask, float('inf'))
        min_angles = angles.min(dim=1).values

        # Negative because we minimize loss → maximizes angles
        return -min_angles.mean()


class AnchorAdaptationLoss(nn.Module):
    """L_ada from EACL stage 2: cosine-similarity-based cross-entropy.

    Trains anchors to be the nearest-neighbor classifier in cosine space.
    After stage 2, inference uses argmax cosine_sim(feat_fusion, anchors).
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, utterance_repr, anchors, labels):
        """
        Args:
            utterance_repr: (N, D) feat_fusion (detached in stage 2)
            anchors: (n_classes, D) trainable anchors
            labels: (N,) long
        Returns:
            scalar L_ada
        """
        r_norm = F.normalize(utterance_repr, dim=-1)
        a_norm = F.normalize(anchors, dim=-1)
        logits = torch.matmul(r_norm, a_norm.T) / self.temperature   # (N, n_classes)
        return F.cross_entropy(logits, labels)
