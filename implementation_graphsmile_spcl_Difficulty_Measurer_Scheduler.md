# Implementing SPCL (Dual-Level Difficulty Measurer + Learning Scheduler) on GraphSmile

**Target codebase:** `github.com/lijfrank/GraphSmile` (official TPAMI 2025 implementation; the paper text cites `lijfrank-open/GraphSmile`, the live repo is `lijfrank/GraphSmile` — same author, use the live one).
**Source method:** Nguyen et al., *Leveraging Self-Paced Curriculum Learning for Enhanced Modality Balance in MERC*, arXiv:2605.21565 (Eq. 1–12, Alg. 1).
**Repo files:** `run.py`, `model.py`, `module.py`, `trainer.py`, `dataloader.py`, `utils.py`, `requirements.txt`.

---

## 0. Why this pairing is empirically motivated (write this into your intro)

GraphSmile's own modality-ablation table (Table VI, MERC rows) is direct evidence of severe modality imbalance in the very model you are extending. WF1 deltas when a modality is removed:

| Dataset | Full WF1 | Δ w/o Textual | Δ w/o Visual | Δ w/o Acoustic |
|---|---|---|---|---|
| IEMOCAP-6 | 72.81 | **−18.58** | −1.58 | −4.55 |
| IEMOCAP-4 | 86.52 | **−24.30** | −0.32 | −3.24 |
| MELD | 66.71 | **−24.44** | −0.45 | −0.31 |
| CMU-MOSEI | 44.93 | **−16.65** | −1.36 | −0.36 |

On MELD, deleting audio *or* video costs under half a WF1 point — the model is effectively text-only despite consuming three modalities. The GraphSmile authors say as much themselves in their Limitation section: noise in the visual/acoustic channels leads the model to favour the textual modality in cross-modal modelling, reducing effective mining of the other two. That is precisely the failure mode SPCL was designed to correct, and SPCL has never been tested on an intermediate-fusion graph architecture. This table is your motivation figure.

---

## 1. The key structural result: SPCL fits GraphSmile *exactly*, with zero new parameters

This is the technical crux of the paper. Read this section carefully before writing any code.

SPCL requires (its Eq. 1–2) per-modality logits `z^m` whose **sum** is the joint logit:

```
z^m_ij = φ_m(x^m_ij; θ^m)          (SPCL Eq. 1)
z^joint_ij = Σ_m z^m_ij            (SPCL Eq. 2)
```

The SPCL authors state this restricts them to late-fusion backbones. **GraphSmile satisfies it anyway**, for two reasons visible in `model.py`:

**(a) Fusion is an unweighted sum of six modality-attributed streams through a *shared* transform.** From `GraphSmile.forward`:

```python
feat_fusion = (self.modal_fusion(featheter_tv[0]) + self.modal_fusion(featheter_ta[0])
             + self.modal_fusion(featheter_tv[1]) + self.modal_fusion(featheter_va[0])
             + self.modal_fusion(featheter_ta[1]) + self.modal_fusion(featheter_va[1])) / 6
```

with `self.modal_fusion = nn.Sequential(nn.Linear(hidden, hidden), nn.LeakyReLU())` — this is paper Eq. 8, `σ(XΘ_h)`, with `Θ_h` shared across all six terms (implementation adds a `/6` the paper omits; harmless).

Stream-to-modality mapping (from the constructor calls `hetergconv_tv((emo_t, emo_v))`, `hetergconv_ta((emo_t, emo_a))`, `hetergconv_va((emo_v, emo_a))`):

| Tensor | Paper notation | Anchor modality |
|---|---|---|
| `featheter_tv[0]` | X^{t←v} | textual |
| `featheter_ta[0]` | X^{t←a} | textual |
| `featheter_tv[1]` | X^{v←t} | visual |
| `featheter_va[0]` | X^{v←a} | visual |
| `featheter_ta[1]` | X^{a←t} | acoustic |
| `featheter_va[1]` | X^{a←v} | acoustic |

Because addition is associative, `feat_fusion = H^t + H^v + H^a` **exactly**, where each `H^m` is the sum of that modality's two streams (divided by 6).

**(b) The emotion classifier is linear.** `self.emo_output = nn.Linear(hidden_dim, n_classes_emo)`. Therefore:

```
logit_emo = feat_fusion @ W^T + b
          = (H^t + H^v + H^a) @ W^T + b
          = (H^t @ W^T + b/3) + (H^v @ W^T + b/3) + (H^a @ W^T + b/3)
          = z^t + z^v + z^a
```

**SPCL Eq. 2 holds identically.** You need no auxiliary heads, no extra parameters, and no change to the forward computation — only a re-grouping of terms that are already being computed. This is a substantially cleaner integration than the "attach auxiliary probing heads" approach that intermediate-fusion architectures normally require, and it is the single most defensible claim in your methods section.

### 1.1 The honest caveat you must state in the paper

`H^t` is **not** a unimodal representation. GSF's alternating propagation means odd layers carry inter-modal cues and even layers carry intra-modal cues, and Eq. 2's residual sum aggregates all layers plus the raw input `X^(0)`. So `H^t` is the *textual-anchored stream*, already contaminated with visual/acoustic information.

Consequence: SPCL's conversation-level score `s_i` measures **inter-stream contribution disparity**, not inter-*modality* predictive disparity as in the original paper. Do not paper over this. State it explicitly, and run **Variant B** (§7) as the control that quantifies how much it matters.

---

## 2. Pre-flight checklist

Do all of this *before* touching SPCL code.

1. **Clone and reproduce the baseline.** Get the datasets (Google Drive link in the repo README), set the `*_path` variables in `run.py`, and reproduce all four numbers with the README commands. You cannot claim a delta against a number you have not reproduced yourself.

   ```
   # IEMOCAP-6
   python -u run.py --gpu 0 --port 1530 --classify emotion \
     --dataset IEMOCAP --epochs 120 --textf_mode textf0 \
     --loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 \
     --win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7
   ```

   Reference targets: IEMOCAP-6 72.77 ACC / 72.81 WF1; IEMOCAP-4 86.53 / 86.52; MELD 67.70 / 66.71; CMU-MOSEI 46.82 / 44.93.

2. **Note the epoch budgets** — they differ per dataset and are *not* SPCL's 50: IEMOCAP-6 and IEMOCAP-4 use 120, MELD 50, CMU-MOSEI 60. Your pacing schedule must be calibrated per dataset accordingly (§6).

3. **Fix the seeding problem.** `trainer.py` hardcodes `seed = 2024` and calls `seed_everything()` at the top of *every* `train_or_eval_model` invocation. For multi-seed significance testing you must parametrize this. Add a `--seed` arg in `run.py` and thread it through; do **not** remove the call, since removing it changes dataloader shuffling behaviour and breaks baseline comparability.

4. **Record the flat-tensor layout.** `trainer.py` builds `dia_lengths` from `umask` and produces `label_emo = torch.cat(label_emotions)` — a 1-D tensor of length `sum(dia_lengths)`. `logit_emo` has shape `(sum(dia_lengths), C)`. There is **no padding** in the loss path. This makes conversation-level aggregation easy (§4.3) — use `dia_lengths`, never a padding mask.

---

## 3. Notation collisions — rename before you start

GraphSmile and SPCL both use λ and α for different things. Fix this now or you will introduce a silent bug.

| Symbol | GraphSmile meaning | SPCL meaning | Use in your code |
|---|---|---|---|
| λ | `lambd[0..2]` = loss weights (emo, sen, shift) | curriculum difficulty threshold | keep `lambd` for losses; use **`spcl_thresh`** for the curriculum |
| α | — | aging/pacing rate | **`spcl_alpha`** |
| ε | — | initial threshold | **`spcl_eps`** |
| B | `shift_win` = utterances per SDP segment | — | leave as `shift_win` |
| L | `heter_n_layers` = GSF depth | — | unchanged |

---

## 4. Implementation, step by step

Create a new file `spcl.py` at the repo root. Keep all SPCL logic in it so the diff against upstream stays reviewable — reviewers will ask.

### Step 4.1 — Expose per-modality fusion terms in `model.py`

Edit `GraphSmile.forward`. Replace the `feat_fusion` block with the grouped version and extend the return signature.

```python
# --- model.py, inside GraphSmile.forward, replacing the feat_fusion line ---

# textual-anchored streams: X^{t<-v}, X^{t<-a}
h_t = (self.modal_fusion(featheter_tv[0]) + self.modal_fusion(featheter_ta[0])) / 6
# visual-anchored streams:  X^{v<-t}, X^{v<-a}
h_v = (self.modal_fusion(featheter_tv[1]) + self.modal_fusion(featheter_va[0])) / 6
# acoustic-anchored streams: X^{a<-t}, X^{a<-v}
h_a = (self.modal_fusion(featheter_ta[1]) + self.modal_fusion(featheter_va[1])) / 6

feat_fusion = h_t + h_v + h_a

logit_emo = self.emo_output(feat_fusion)
logit_sen = self.sen_output(feat_fusion)
logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths)

# per-modality logits satisfying SPCL Eq. 2: z_t + z_v + z_a == logit_emo
W, b = self.emo_output.weight, self.emo_output.bias
uni_logits = {
    't': h_t @ W.t() + b / 3.0,
    'v': h_v @ W.t() + b / 3.0,
    'a': h_a @ W.t() + b / 3.0,
}

return logit_emo, logit_sen, logit_shift, feat_fusion, uni_logits
```

**Regression test (run this first, before anything else).** Add a temporary assertion:

```python
assert torch.allclose(z_t + z_v + z_a, logit_emo, atol=1e-4), "Eq.2 decomposition broken"
```

Use `allclose` with a tolerance, not `==`: regrouping the six-term sum changes floating-point accumulation order, so bitwise equality will not hold. If this assertion fails by more than ~1e-4 you have mismapped a stream — recheck the table in §1.

**Bias handling.** Splitting `b/3` makes the decomposition exact. Two alternatives, both defensible, both should appear as a one-line ablation footnote: (i) omit the bias entirely from `uni_logits` — harmless in practice because `s^m_i` (Eq. 4) only reads softmax probabilities, and a constant shift shared across modalities largely cancels in the standard deviation; (ii) give each stream the full `b` — breaks Eq. 2 exactness, do not use.

**Two-modality runs.** If you run ablations with `args.modals` set to two modalities, only one graph exists and `uni_logits` will have two entries. `torch.std` over two values is still well-defined (see §4.3 note on `unbiased`). Guard the dict construction against missing graphs rather than assuming three.

### Step 4.2 — Utterance-level difficulty score (SPCL Eq. 3)

This is just the per-sample emotion cross-entropy. In `trainer.py` the loss functions are called as `loss_function_emo(prob_emo, label_emo)` with `prob_emo = F.log_softmax(logit_emo, -1)`, i.e. NLL-style. You need the **unreduced** version.

In `spcl.py`:

```python
import torch
import torch.nn.functional as F


def utterance_scores(logit_emo, label_emo, class_weight=None):
    """SPCL Eq. 3. Returns (N,) per-utterance NLL. N = sum(dia_lengths)."""
    logp = F.log_softmax(logit_emo, dim=-1)
    return F.nll_loss(logp, label_emo, weight=class_weight, reduction='none')
```

**Important:** if the baseline's `loss_function_emo` uses class weights (check how it is constructed in `run.py`), pass the same weights here so the masked loss you build in §4.6 is numerically identical to the baseline loss when all samples are admitted. Otherwise your `v ≡ 1` regression test in §8 will silently fail.

### Step 4.3 — Conversation-level modality-discrepancy score (SPCL Eq. 4–5)

This is where GraphSmile's flat layout helps: build a `conv_id` index once per batch and use `index_add_`.

```python
def conversation_scores(uni_logits, label_emo, dia_lengths,
                        normalize=True, unbiased=False):
    """
    SPCL Eq. 4-5, adapted to GraphSmile's flat (N, C) layout.

    uni_logits : dict {modality: (N, C)}
    label_emo  : (N,) long
    dia_lengths: list[int], sum == N
    normalize  : if True, average over utterances instead of summing (see notes)
    Returns    : (N,) conversation-level score, broadcast back to utterances
    """
    device = label_emo.device
    lengths = torch.as_tensor(dia_lengths, device=device)
    n_conv = lengths.numel()
    conv_id = torch.repeat_interleave(
        torch.arange(n_conv, device=device), lengths)          # (N,)

    per_modality = []
    for m in sorted(uni_logits):                                # deterministic order
        p = F.softmax(uni_logits[m], dim=-1)                    # (N, C)
        p_true = p.gather(1, label_emo.unsqueeze(1)).squeeze(1)  # (N,)
        s_m = torch.zeros(n_conv, device=device, dtype=p_true.dtype)
        s_m.index_add_(0, conv_id, p_true)                      # Eq. 4
        if normalize:
            s_m = s_m / lengths.clamp(min=1).to(s_m.dtype)
        per_modality.append(s_m)

    S = torch.stack(per_modality, dim=0)                        # (M, n_conv)
    s_conv = S.std(dim=0, unbiased=unbiased)                    # Eq. 5
    return s_conv[conv_id], S                                   # (N,), (M, n_conv)
```

Three decisions embedded here, each of which you must report:

1. **`normalize=True` is a deliberate deviation from SPCL Eq. 4, and you should default to it.** SPCL *sums* `p_true` over the utterances of a conversation, so `s^m_i` scales with dialogue length. IEMOCAP averages ~49 utterances per dialogue; MELD averages ~10. With the raw sum, `s_i` on IEMOCAP lands in the units-to-tens range while on MELD it is several times smaller — which is exactly why the SPCL authors needed a per-dataset grid search over ε, and why they flag hyperparameter tuning as their headline limitation. Averaging makes `s^m_i ∈ [0,1]` and length-invariant, so a single calibration procedure transfers across all four GraphSmile datasets. Run `normalize=False` as an ablation and report both; if the normalized version transfers better, that is a small but genuine methodological contribution over the source paper.

2. **`unbiased=False`** (population std). SPCL says only "the standard deviation function." Biased vs. unbiased differ by a constant factor `sqrt(M/(M-1))`, which is absorbed into ε — but it *changes* between 3-modality and 2-modality ablation runs, so fixing `unbiased=False` keeps the difficulty scale comparable across your modality-ablation conditions. Note this choice in a footnote.

3. **Sorted modality keys** so the score is reproducible across runs and Python versions.

### Step 4.4 — Combine into difficulty (SPCL Eq. 6)

```python
def difficulty(l_utt, s_conv, eps=1e-8):
    """SPCL Eq. 6: harmonic mean of recognition difficulty and modality misalignment."""
    return 2.0 * s_conv * l_utt / (s_conv + l_utt + eps)
```

Behavioural note worth stating in the paper: the harmonic mean is dominated by whichever term is smaller (`2·min` in the limit of large disparity). So a sample is "easy" if it is *either* well-classified *or* modality-balanced — not necessarily both. This is the intended design in SPCL (they argue it prevents one factor from dominating), but it is worth verifying empirically on GraphSmile by logging the correlation between `l_utt` and `s_conv`; if they are near-independent, the harmonic mean behaves very differently than if they are correlated.

### Step 4.5 — Learning Scheduler: hard regularizer + exponential pacing (SPCL Eq. 7–8)

```python
class SPCLScheduler:
    """SPCL Eq. 7-8. Hard binary regularizer with exponential threshold growth."""

    def __init__(self, eps, alpha, min_frac=0.0):
        self.thresh = float(eps)
        self.alpha = float(alpha)
        self.min_frac = float(min_frac)   # safety floor, see note below

    def mask(self, rho):
        """Eq. 7. rho: (N,) detached. Returns (N,) float mask in {0,1}."""
        v = (rho <= self.thresh).float()
        if self.min_frac > 0.0:
            n_min = max(1, int(self.min_frac * rho.numel()))
            if v.sum().item() < n_min:
                idx = torch.topk(rho, n_min, largest=False).indices
                v = torch.zeros_like(rho)
                v[idx] = 1.0
        return v

    def step_epoch(self):
        """Eq. 8. Call ONCE per epoch, after the epoch completes."""
        self.thresh *= self.alpha
```

**The `min_frac` floor is an addition, not in SPCL.** Justification: SPCL's Alg. 1 computes the mask per mini-batch, and with GraphSmile's batch size of 16 *conversations* a single batch can contain thousands of utterances but still have a pathological epoch-0 mask where almost nothing passes. An all-zero mask produces a zero-gradient step or a divide-by-zero in the normalizer. Set `min_frac ≈ 0.05–0.10` and log how often the floor actually fires; if it fires after epoch 2 your ε is too small. Report the floor as part of the method and ablate `min_frac = 0`.

**Threshold updates once per epoch, not per batch.** SPCL Alg. 1 puts `Update threshold λ` outside the mini-batch loop. In GraphSmile the epoch loop lives in `run.py`, while `train_or_eval_model` handles the batches — so call `scheduler.step_epoch()` in `run.py` after the training call, and **only for training epochs**, never for validation/test calls.

### Step 4.6 — Wire the mask into the loss (SPCL Eq. 11)

Edit `trainer.py`. Signature change plus the loss block:

```python
def train_or_eval_model(model, loss_function_emo, loss_function_sen,
                        loss_function_shift, dataloader, epoch, cuda, modals,
                        optimizer=None, train=False, dataset='IEMOCAP',
                        loss_type='', lambd=[1.0, 1.0, 1.0], epochs=100,
                        classify='', shift_win=5,
                        spcl_scheduler=None, spcl_cfg=None):   # <-- new
```

Inside the batch loop, after the model call:

```python
logit_emo, logit_sen, logit_sft, extracted_feature, uni_logits = model(
    textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask, dia_lengths)

prob_emo = F.log_softmax(logit_emo, -1)
prob_sen = F.log_softmax(logit_sen, -1)
prob_sft = F.log_softmax(logit_sft, -1)
label_sft = build_match_sen_shift_label(shift_win, dia_lengths, label_sen)

use_spcl = train and (spcl_scheduler is not None)

if not use_spcl:
    loss_emo = loss_function_emo(prob_emo, label_emo)
    loss_sen = loss_function_sen(prob_sen, label_sen)
    loss_sft = loss_function_shift(prob_sft, label_sft)
else:
    l_utt = utterance_scores(logit_emo, label_emo,
                             class_weight=spcl_cfg.get('class_weight'))

    with torch.no_grad():                       # difficulty is a gating decision
        s_conv, S = conversation_scores(
            uni_logits, label_emo, dia_lengths,
            normalize=spcl_cfg['normalize'],
            unbiased=False)
        rho = difficulty(l_utt.detach(), s_conv)
        v = spcl_scheduler.mask(rho)            # (N,) in {0,1}

    denom = v.sum().clamp(min=1.0)
    loss_emo = (v * l_utt).sum() / denom        # SPCL Eq. 11

    # Stage-dependent masking of the auxiliary tasks -- see Step 4.7
    if spcl_cfg['mask_sen']:
        l_sen = F.nll_loss(prob_sen, label_sen, reduction='none')
        loss_sen = (v * l_sen).sum() / denom
    else:
        loss_sen = loss_function_sen(prob_sen, label_sen)

    if spcl_cfg['mask_sft']:
        v_pair = build_pairwise_mask(v, dia_lengths, shift_win)   # see below
        l_sft = F.nll_loss(prob_sft, label_sft, reduction='none')
        loss_sft = (v_pair * l_sft).sum() / v_pair.sum().clamp(min=1.0)
    else:
        loss_sft = loss_function_shift(prob_sft, label_sft)

    spcl_cfg['logger'].accumulate(rho=rho, v=v, S=S, labels=label_emo)
```

The downstream `loss_type == 'emo_sen_sft'` branch (`lambd[0]*loss_emo + lambd[1]*loss_sen + lambd[2]*loss_sft`) is left completely untouched.

**Three correctness traps:**

- **Detach the difficulty.** `rho` and `v` must be computed under `no_grad` and from `l_utt.detach()`. The mask is a hard `{0,1}` decision with zero gradient anyway, but leaving `s_conv` attached builds a large unused autograd graph through all three GSF towers and will inflate memory noticeably on IEMOCAP-6 with `hidden_dim=512` and `heter_n_layers 7 7 7`.
- **Normalize by `Σv`, not by `N`.** SPCL Eq. 11 divides by the count of admitted samples. Dividing by `N` instead makes the effective learning rate shrink with the mask, which confounds the curriculum effect with a learning-rate schedule — a reviewer will catch this.
- **Never apply the mask at eval time.** The `train` flag guard above handles it; verify by asserting `spcl_scheduler is None or train` inside the SPCL branch.

**Pairwise mask for the SDP task.** `build_match_sen_shift_label` constructs shift labels over utterance pairs within `shift_win`-sized segments. Because you did not write that indexing, do **not** guess it — read `module.py::build_match_sen_shift_label`, mirror its exact index construction, and set `v_pair(i,j) = v_i · v_j`. Add a shape assertion `assert v_pair.shape == label_sft.shape`. If mirroring it proves fiddly, that is a strong argument for keeping Stage 1 (§4.7) as your headline configuration and treating SDP masking as an ablation only.

### Step 4.7 — Staged rollout (do not mask everything at once)

Run these in order. Each stage is one row of your ablation table.

| Stage | `mask_sen` | `mask_sft` | Rationale |
|---|---|---|---|
| **S1 (headline)** | False | False | Masks only `L_e`. Cleanest attribution: any change is due to the emotion-task curriculum alone. SPCL was only ever defined over a single-task emotion loss, so this is the faithful port. |
| S2 | True | False | Sentiment head shares `feat_fusion` and is per-utterance, so masking is trivial and well-defined. |
| S3 | True | True | Full masking. Highest risk: SDP is GraphSmile's contrastive-like regularizer, and starving it early may destabilize the representation the curriculum is trying to balance. |

Predict S1 ≥ S3 and say so in advance.

### Step 4.8 — Epoch loop in `run.py`

```python
from spcl import SPCLScheduler

spcl_scheduler = SPCLScheduler(eps=args.spcl_eps,
                               alpha=args.spcl_alpha,
                               min_frac=args.spcl_min_frac) if args.use_spcl else None

for e in range(n_epochs):
    train_out = train_or_eval_model(..., epoch=e, train=True,
                                    spcl_scheduler=spcl_scheduler, spcl_cfg=spcl_cfg)
    valid_out = train_or_eval_model(..., train=False, spcl_scheduler=None)
    test_out  = train_or_eval_model(..., train=False, spcl_scheduler=None)

    if spcl_scheduler is not None:
        spcl_scheduler.step_epoch()          # SPCL Eq. 8, once per epoch
        log_spcl_diagnostics(e, spcl_scheduler, spcl_cfg['logger'])
        spcl_cfg['logger'].reset()
```

New CLI args to add: `--use_spcl`, `--spcl_eps`, `--spcl_alpha`, `--spcl_min_frac`, `--spcl_normalize`, `--spcl_mask_sen`, `--spcl_mask_sft`, `--seed`.

---

## 5. Hyperparameter calibration (replace grid search with this)

SPCL's own limitations section calls brute-force tuning of (ε, α) impractical. Since you are changing backbones *and* recommending normalized conversation scores, SPCL's published ranges (ε ∈ [0.6, 1.2], α ∈ [1.05, 1.4]) **will not transfer** — they were calibrated for un-normalized, length-summed scores on ~49-utterance IEMOCAP dialogues. Use a data-driven procedure instead, and present it as a methodological improvement:

**Calibration run (once per dataset, ~2 epochs, no masking):**

1. Train with `use_spcl=False` for 1–2 epochs. At the end, do one full pass over the training set computing `rho` for every utterance (no gradient).
2. Set `ε = Quantile(rho, q₀)` with `q₀ ∈ [0.3, 0.5]` → initial curriculum expanding rate of 30–50%. SPCL's Fig. 3 shows their best-performing backbone (DialogueGCN) starting near 0.3 and rising steadily.
3. Set `α = (Quantile(rho, 0.99) / ε)^(1/T_full)` where `T_full = 0.4 × total_epochs`, so the curriculum saturates at rate ≈ 1.0 around 40% through training. Per dataset: `T_full ≈ 48` for IEMOCAP-6/4 (120 epochs), `≈ 20` for MELD (50), `≈ 24` for CMU-MOSEI (60).

Then sweep only `q₀ ∈ {0.3, 0.4, 0.5}` and `T_full ∈ {0.3, 0.4, 0.5} × epochs` — 9 runs per dataset instead of a blind 2-D grid. SPCL's Sec. 5.2.2 finding is that each backbone has one optimal expanding rate and both faster and slower pacing degrade performance; parametrizing by *expanding rate* directly rather than by (ε, α) is the point.

Everything else stays at GraphSmile's published values (`lr`, `hidden_dim`, `win`, `heter_n_layers`, `drop`, `shift_win`, `lambd`, AdamW, L2 = 1e-3). **Do not re-tune GraphSmile's hyperparameters** — if you do, you can no longer attribute gains to SPCL.

---

## 6. Diagnostics to log every epoch

These are not optional; two of them are your paper's analysis figures.

1. **Curriculum expanding rate** = `Σv / N` per epoch. Reproduces SPCL Fig. 3. Verify it increases monotonically and saturates before training ends.
2. **Threshold trajectory** `spcl_thresh` per epoch.
3. **Modality ratio** — SPCL Sec. 5.2.5, their headline balance evidence. Using the per-conversation `S` matrix from §4.3, compute per epoch `r_m = mean_i(S[m,i]) / min_m' mean_i(S[m',i])`. Plot `r_t, r_v, r_a` with and without SPCL. **Expected result: `r_t` falls and `r_v`, `r_a` rise.** If `r_t` does not fall, SPCL is not doing what it claims on this backbone, and that is your paper's finding.
4. **Per-class composition of the admitted set** — fraction of admitted samples belonging to each emotion class, per epoch. See §7 for why this matters.
5. **Per-class F1** on test, every epoch, for both baseline and SPCL runs.
6. **Correlation between `l_utt` and `s_conv`** (Spearman, over the training set, per epoch). Tells you whether the harmonic mean is doing meaningful work or is just a monotone function of the loss.
7. **Wall-clock per epoch.** SPCL claims negligible overhead; GraphSmile's Table VII reports its own time/memory. Report yours in the same format.

---

## 7. Pre-registered risks and predictions

Write these down before you run anything, then report against them. This is what turns a "we plugged X into Y" paper into a credible empirical study.

**R1 — Curriculum-induced minority-class starvation (highest-value risk).** The hard regularizer excludes high-loss samples, and in ERC the highest-loss samples are precisely the minority classes. GraphSmile already scores 14.00 F1 on MELD *Fear* and 26.47 on *Disgust*, and 0.00 on CMU-MOSEI *Highly Negative*. If SPCL withholds these classes for the first 30–40% of training, per-class F1 on the tail may degrade even while weighted F1 improves. Diagnostic #4 above is the instrument. If confirmed, this is a genuinely novel negative finding about curriculum learning on class-imbalanced ERC — publishable in its own right, and it motivates an obvious fix (class-stratified admission quotas) that could become your second contribution.

**R2 — MELD gains will be small.** The SPCL authors themselves report that MELD's short, fragmented dialogues limit their curriculum's effectiveness. GraphSmile's MELD config compounds this: `win 3 3`, `shift_win 3`, only 50 epochs. Predict smallest gains here. Do not quietly drop MELD if it comes out flat.

**R3 — Smaller headroom than the source paper.** SPCL's +6.6 to +10.4 WF1 figures came from weak or deliberately weakened baselines (DialogueGCN was originally text-only and they extended it with naive late fusion). GraphSmile is a tuned TPAMI SOTA. Realistic expectation: **+0.5 to +2.0 WF1**, not +6. Frame the contribution as "does training-time balancing still help a strong intermediate-fusion model," not as a leaderboard claim.

**R4 — The stream-contamination confound (§1.1).** `H^t` already contains cross-modal information, so `s_conv` may under-estimate true modality disparity. **Variant B control:** attach two frozen-architecture auxiliary linear heads directly to the pre-graph features `featdim_t`, `featdim_v`, `featdim_a` (available in `forward` before `batch_to_all_tva`), compute a genuinely unimodal `s_conv` from those, and compare against the Variant A (decomposition) difficulty. Report the rank correlation between the two difficulty orderings. High correlation → the confound is benign and Variant A's zero-parameter elegance stands. Low correlation → Variant B is the honest method and Variant A is an approximation. Either outcome is reportable.

**R5 — Multi-task interaction.** GraphSmile's `L_total` has three terms; SPCL was designed for one. Masking `L_e` while leaving `L_s` and `L_o` unmasked effectively *reweights* the multi-task balance epoch by epoch (the emotion term is averaged over a shrinking subset). The staged rollout in §4.7 is what isolates this; you may also want an S1-control where `lambd[0]` is rescaled by `Σv/N` to hold the effective weighting constant.

---

## 8. Validation checklist before reporting any number

- [ ] `torch.allclose(z_t + z_v + z_a, logit_emo, atol=1e-4)` passes.
- [ ] `torch.allclose(h_t + h_v + h_a, feat_fusion_original, atol=1e-5)` passes against the unmodified code path.
- [ ] **Null-curriculum regression test:** with `spcl_eps = +inf` (so `v ≡ 1`) and the same seed, the SPCL run reproduces the baseline metrics to within floating-point noise across all 120 epochs. *If this fails, everything downstream is invalid.* The most common cause is a class-weight mismatch between your unreduced `utterance_scores` and the repo's `loss_function_emo`.
- [ ] Mask is never applied on validation or test passes.
- [ ] `spcl_scheduler.step_epoch()` is called exactly once per training epoch (assert a call counter equals the epoch index).
- [ ] Expanding rate reaches 1.0 before the final epoch on every dataset.
- [ ] Baseline reproduced to within ±0.5 WF1 of the published number on all four datasets, **using your own runs, not the paper's table**, before any delta is computed.

---

## 9. Experiment matrix

Five seeds per cell, paired t-test at p < 0.05 (matching both papers' convention). Metrics: ACC and WF1, plus per-class F1.

| # | Config | Datasets | Purpose |
|---|---|---|---|
| 1 | GraphSmile baseline (reproduced) | all 4 | anchor |
| 2 | + SPCL, Stage S1, normalized | all 4 | **headline result** |
| 3 | + SPCL, S1, un-normalized (faithful Eq. 4) | all 4 | tests the normalization contribution |
| 4 | + SPCL, S2 / S3 | IEMOCAP-6, MELD | multi-task masking ablation |
| 5 | w/o utterance-score (`ρ = s_conv`) | IEMOCAP-6, MELD | mirrors SPCL Table 5 |
| 6 | w/o conversation-score (`ρ = l_utt`) | IEMOCAP-6, MELD | mirrors SPCL Table 5 |
| 7 | Soft Linear / Soft Logistic regularizer | IEMOCAP-6, MELD | mirrors SPCL Table 7; tests whether "hard wins" is backbone-specific |
| 8 | Cosine / MA / competence-based pacing | IEMOCAP-6 | mirrors SPCL Table 9 |
| 9 | `min_frac = 0` | IEMOCAP-6 | ablates your safety floor |
| 10 | Variant B (auxiliary unimodal heads) | IEMOCAP-6 | R4 confound control |
| 11 | Modality ablations (No T / No V / No A) + SPCL | IEMOCAP-6, MELD | GraphSmile's analogue of SPCL's TAV/TA/TV/AV table |
| 12 | MSAC task (`--classify sentiment`) + SPCL | all 4 | SPCL's own stated future direction: sentiment analysis in conversation |

Cell 12 is nearly free — GraphSmile already supports it via `--classify`, and the SPCL paper explicitly names conversational sentiment analysis as a feasible extension. Including it materially strengthens the paper's scope claim for very little extra compute.

---

## 10. What to report

Whatever the outcome, the paper has three reportable results:

1. **The decomposition itself** (§1) — a general observation that any MERC architecture with additive modality-attributed fusion and a linear head admits SPCL-style per-modality logits for free. This extends SPCL's applicability beyond the late-fusion family its authors thought it was restricted to, and it generalizes to other models in the same lineage (GraphCFC, M3Net, MMGCN).
2. **Whether training-time curriculum balancing still helps a strong intermediate-fusion backbone**, with modality-ratio evidence (diagnostic #3) showing whether the mechanism works as advertised. A null or mixed result here is informative, not a failure — it would suggest GSF's alternating propagation already performs implicit balancing, which is a finding about GSF.
3. **The class-starvation analysis** (R1), which no existing curriculum-learning MERC paper reports.

Do not build the paper's value on (2) alone. Results (1) and (3) hold regardless of whether the WF1 number moves.

---

## 11. Reference points

- SPCL: Nguyen, P.-A., Le, T.-S., Le, D.-T., Nguyen, C.-V. T. arXiv:2605.21565. Eq. 1–12, Alg. 1; ablations Tables 5–9; limitations §5.2.6.
- GraphSmile: Li, J., Wang, X., Zeng, Z. *IEEE TPAMI* 47(10):8786–8803, 2025. doi:10.1109/TPAMI.2025.3581236. Eq. 8 (fusion), Eq. 9–10 (emotion), Eq. 11–13 (SDP), Eq. 16 (total objective); Table VI (modality ablation = your motivation); Table VII (time/memory format to match).
- Code: `github.com/lijfrank/GraphSmile`.
- Related methods to cite as comparators: Ada2I (ACM MM 2024), DynCIM (arXiv:2503.06456), OGM-GE (CVPR 2022), FAGM (ACM MM 2023), OPM (TPAMI 2024).
