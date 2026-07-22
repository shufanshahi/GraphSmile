"""Self-Paced Curriculum Learning utilities for GraphSmile."""

import numpy as np
import torch
import torch.nn.functional as F


def utterance_scores(logit_emo, label_emo, class_weight=None):
    """SPCL Eq. 3: unreduced per-utterance negative log likelihood."""
    return F.nll_loss(F.log_softmax(logit_emo, dim=-1), label_emo,
                      weight=class_weight, reduction="none")


def conversation_scores(uni_logits, label_emo, dia_lengths, normalize=True,
                        unbiased=False):
    """SPCL Eq. 4--5 on GraphSmile's flattened utterance layout."""
    if len(uni_logits) < 2:
        raise ValueError("SPCL conversation scoring needs at least two streams")
    device = label_emo.device
    lengths = torch.as_tensor(dia_lengths, device=device, dtype=torch.long)
    if int(lengths.sum()) != label_emo.numel():
        raise ValueError("sum(dia_lengths) must equal the number of labels")
    conv_id = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device), lengths)

    per_modality = []
    for modality in sorted(uni_logits):
        probabilities = F.softmax(uni_logits[modality], dim=-1)
        p_true = probabilities.gather(1, label_emo[:, None]).squeeze(1)
        score = torch.zeros(lengths.numel(), device=device,
                            dtype=p_true.dtype)
        score.index_add_(0, conv_id, p_true)
        if normalize:
            score = score / lengths.clamp(min=1).to(score.dtype)
        per_modality.append(score)

    modality_scores = torch.stack(per_modality, dim=0)
    conversation_score = modality_scores.std(dim=0, unbiased=unbiased)
    return conversation_score[conv_id], modality_scores


def difficulty(l_utt, s_conv, eps=1e-8):
    """SPCL Eq. 6: harmonic mean of the two difficulty signals."""
    return 2.0 * s_conv * l_utt / (s_conv + l_utt + eps)


def build_pairwise_mask(mask, dia_lengths, shift_win):
    """Mirror module.build_match_sen_shift_label's exact flattened ordering."""
    pair_masks = []
    start = 0
    for dia_len in dia_lengths:
        if shift_win == -1:
            chunk = mask[start:start + dia_len]
            pair_masks.append((chunk[:, None] * chunk[None, :]).reshape(-1))
        elif shift_win > 0:
            for win_start in range(0, dia_len, shift_win):
                win = min(shift_win, dia_len - win_start)
                chunk = mask[start + win_start:start + win_start + win]
                pair_masks.append((chunk[:, None] * chunk[None, :]).reshape(-1))
        else:
            raise ValueError("shift_win must be positive or -1")
        start += dia_len
    return torch.cat(pair_masks)


class SPCLScheduler:
    """SPCL Eq. 7--8: hard admission and exponential pacing."""

    def __init__(self, eps, alpha, min_frac=0.0):
        if eps < 0 or alpha <= 0 or not 0 <= min_frac <= 1:
            raise ValueError("eps >= 0, alpha > 0, and min_frac in [0, 1] required")
        self.thresh = float(eps)
        self.alpha = float(alpha)
        self.min_frac = float(min_frac)
        self.steps = 0
        self.floor_fires = 0

    def mask(self, rho):
        v = (rho <= self.thresh).to(rho.dtype)
        if self.min_frac > 0 and rho.numel():
            n_min = max(1, int(self.min_frac * rho.numel()))
            if int(v.sum()) < n_min:
                v = torch.zeros_like(rho)
                v[torch.topk(rho, n_min, largest=False).indices] = 1.0
                self.floor_fires += 1
        return v

    def step_epoch(self):
        self.thresh *= self.alpha
        self.steps += 1


class SPCLLogger:
    """Accumulate lightweight epoch diagnostics without retaining GPU graphs."""

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def accumulate(self, rho, v, modality_scores, labels, l_utt, s_conv):
        self.rho.append(rho.detach().cpu())
        self.mask.append(v.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.l_utt.append(l_utt.detach().cpu())
        self.s_conv.append(s_conv.detach().cpu())
        self.modality_sums += modality_scores.detach().sum(dim=1).cpu().double()
        self.conversations += modality_scores.shape[1]

    def summary(self):
        if not self.mask:
            return {}
        masks = torch.cat(self.mask)
        labels = torch.cat(self.labels)
        admitted = torch.bincount(labels[masks.bool()], minlength=self.n_classes)
        totals = torch.bincount(labels, minlength=self.n_classes).clamp(min=1)
        means = self.modality_sums / max(self.conversations, 1)
        ratios = means / means.min().clamp(min=torch.finfo(means.dtype).eps)
        x, y = torch.cat(self.l_utt).numpy(), torch.cat(self.s_conv).numpy()
        # Rank-transform locally to avoid adding scipy as a dependency.
        xr = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
        yr = np.argsort(np.argsort(y, kind="mergesort"), kind="mergesort")
        corr = float(np.corrcoef(xr, yr)[0, 1]) if len(x) > 1 else float("nan")
        return {
            "expanding_rate": float(masks.mean()),
            "admitted_per_class": (admitted.float() / totals).tolist(),
            "modality_ratio": dict(zip(('a', 't', 'v'), ratios.tolist())),
            "difficulty_correlation": corr,
        }

    def reset(self):
        self.rho, self.mask, self.labels = [], [], []
        self.l_utt, self.s_conv = [], []
        self.modality_sums = torch.zeros(3, dtype=torch.double)
        self.conversations = 0
