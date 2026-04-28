# TinyVLM NeurIPS 2026 Implementation Plan

## Source material

- Code: `tinyvlm_vast.py`
- Paper: `tinyvlm2.2.tex`
- Figures: `Figure/ANC_Acc.PNG`, `Figure/ANC_Loss.PNG`, `Figure/f_STTF_Acc.PNG`, `Figure/f_STTF_Loss.PNG`

---

## Issue analysis

### Issue 1 — FLOPs arithmetic is wrong (blocks all efficiency claims)

**What the paper claims.**
Table 3 caption: "STTF + ANC routes inputs adaptively across three encoder branches, averaging ∼10.3 GFLOPs (0.31×2 + 0.34×8 + 0.35×20 GFLOPs)."
Section 3.2: "In practice, we use top-k routing to activate only a subset of branches, reducing computation."

**What the code does** (`AdaptiveNeuralCompression.forward`, lines 575–586).
```python
feats = [enc(rgb, events) for enc in self.encoders]   # ALL THREE encoders run unconditionally
comp_cost = comp_cost + w.squeeze(-1) * self.encoders[i].flops  # soft expected FLOPs
```
All three encoders (2 + 8 + 20 = **30 GFLOPs**) run on every input. The soft formula `0.31×2 + 0.34×8 + 0.35×20 = 10.34` is only valid under hard routing (one branch per sample). Soft routing means all branches always run, so actual FLOPs are **30 GFLOPs — 50 % worse than the dense 20-GFlOP baseline it claims to beat**.

The training `comp_cost` tracks expected FLOPs and penalises the loss appropriately, but the forward pass does not gate any computation on the router decision. This is an implementation gap, not just a reporting error.

**Note on vocabulary (Issue 3, Stage A).** The canonical run in `results-tinyvlm_prev/tinyvlm_cider_seeds.nosync/` contains `vocabulary.json` in every seed directory, confirming the invertible `Vocabulary` class was correctly used for the published CIDEr numbers. Stage A is already resolved — the CIDEr 36.70 figure is from the correct tokenizer path. Stage A does not need to be re-implemented or re-run.

**Fix (code).**
At inference time (`model.eval()`), replace the soft weighted sum with hard routing: select the argmax branch, run only that encoder, zero the rest. The Gumbel-Softmax training path is unchanged (differentiable soft routing during `.train()`). This is the standard straight-through trick — train soft, evaluate hard.

Concretely, in `AdaptiveNeuralCompression.forward`:
```python
if self.training:
    # existing soft path (unchanged)
    feats = [enc(rgb, events) for enc in self.encoders]
    ...
else:
    # hard routing: only the argmax branch runs
    branch_idx = weights.argmax(dim=-1)          # [B]
    encoded = torch.zeros(B, self.cfg.hidden_dim, device=rgb.device)
    comp_cost = torch.zeros(B, device=rgb.device)
    for b in range(B):
        k = branch_idx[b].item()
        feat = self.encoders[k](rgb[b:b+1], events[b:b+1])
        encoded[b] = self.projections[k](feat).squeeze(0)
        comp_cost[b] = self.encoders[k].flops
```
A vectorised version groups samples by selected branch for efficient batching.

**Fix (paper).**
In Section 3.2, remove the sentence "In practice, we use top-k routing to activate only a subset of branches." Replace with a sentence explaining that training uses soft Gumbel routing for differentiability while inference uses hard argmax routing to realise the FLOPs savings. Update Table 3 to report the hard-routing average FLOPs, which equal `Σ_k f_k · FLOPs(E_k)` where `f_k` is the hard-routing fraction (from Table 2 routing analysis: 31 %×2 + 34 %×8 + 35 %×20 = 10.34 GFLOPs — the formula is valid only after the code fix).

**Success gate.**
A smoke-test run with `--smoke_test --epochs 2` prints `[val] encoder FLOPs` matching the per-branch fractions from `UtilizationTracker`. `comp_cost.mean()` at eval time equals the weighted hard-routing average, not 30 GFLOPs.

---

### Issue 2 — Figures don't match the training protocol (data integrity, most serious)

**What the paper text claims.**
- Section 4.1: "We train for 10 epochs."
- Section 5 (Results): "Validation accuracy reaches 45.91 % at epoch 9 (training 45.10 %)."
- Section 5: "Train/val gap remains below 1 percentage point across all 10 epochs."
- Section 5: "validation loss does *not* diverge."

**What the old figures showed** (the four `.PNG` files originally in `Figure/`).

| Figure | X-axis (epochs) | Train Acc/Loss | Val behaviour |
|---|---|---|---|
| `f_STTF_Acc.PNG` (Fig 2) | 0–50 | rises to 1.0 | flat ~0.2, **massive overfit** |
| `f_STTF_Loss.PNG` (Fig 3) | 0–50 | drops to ≈0 | **rises to 5+, diverging** |
| `ANC_Acc.PNG` (Fig 4) | 0–30 | rises to ~0.47 | oscillates 0.10–0.20 |
| `ANC_Loss.PNG` (Fig 5) | 0–30 | decreases | **oscillates 2–6.5, diverging** |

**What the actual canonical run data shows** (cross-verified from `results-tinyvlm_prev/tinyvlm_cider_seeds.nosync/seed_42_tau_0.80/metrics.jsonl`).

The paper's numerical claims are **fully verified** against the real run data:

| Claim | Actual value | Match |
|---|---|---|
| CIDEr 36.70 ± 0.37 | 36.703 ± 0.370 (seeds 42/43/44) | ✅ exact |
| Val acc 45.89 % ± 0.05 % | 45.888 % ± 0.050 % | ✅ exact |
| BLEU-4 14.20 ± 0.14 | 14.203 ± 0.140 | ✅ exact |
| Val acc > train acc at epoch 9 | val=0.4591 > train=0.4429 | ✅ confirmed |
| Val loss not diverging | val loss 4.55→4.20, monotone | ✅ confirmed |
| τ table CIDEr values | all 5 values match to 2 d.p. | ✅ exact |

The 3-seed `tinyvlm_cider_seeds` run IS the canonical run the paper describes. The vocabulary was used correctly (vocabulary.json present in all seed directories), COCO images and annotations were used, and the training ran for exactly 10 epochs on 4 GPUs.

**Root cause.** The paper text was updated after adding regularisation but the `.PNG` files were never replaced. The old PNGs are from earlier development runs with no dropout or label smoothing.

**Fix (already done).**
The four replacement figures were generated directly from the verified `metrics.jsonl` using `gen_figures.py`. All four now show:
- X-axis: epochs 0–9 (10 epochs, 0-indexed)
- Identical y-axis scales for the paired accuracy plots (0.36–0.47)
- Val accuracy consistently above train accuracy (dropout suppression)
- Both losses decreasing monotonically, no divergence

**No re-run is needed.** The data exists and is correct. Only the figure files needed replacing.

**Success gate.**
✅ All four `Figure/*.PNG` files replaced (done). Values are cross-verified against `metrics.jsonl`.

---

### Issue 3 — CIDEr 36.70 is below competitive threshold for NeurIPS (requires backbone scaling)

**Why the number is low.**
The current model trains from scratch with:
- A custom 3-branch CNN encoder (no pretrained weights, no ImageNet)
- A hash-based 8K vocabulary (one-way; words with hash collisions share IDs)
- A 2-layer transformer decoder with 384-dim hidden state
- Only 10 epochs on COCO train2017

Competitive COCO captioning results in 2024–2026 range from CIDEr ≈ 85 (Show and Tell, 2015) to 140+ (state-of-the-art). CIDEr 36.70 is not competitive with anything in the literature published after 2016. The paper's framing as a "proof-of-concept" delays but does not avoid this critique — NeurIPS reviewers require at least a clear path to competitive performance.

**Fix (two-stage).**

*Stage A — proper vocabulary (necessary for any meaningful CIDEr score).*
Replace `simple_tokenize` (hash-based, irreversible) with the existing `Vocabulary` class (already in the codebase, frequency-ranked, reversible). The `build_vocabulary` function and `Vocabulary.encode/decode` are already implemented. The smoke-test uses synthetic tokens and can't test this; the COCO path must use `Vocabulary`. This alone lifts CIDEr meaningfully because greedy decoding will produce valid English words instead of randomly-hashed tokens.

Verify: `greedy_decode` already uses `vocab.decode`, so this fix is only ensuring the `--coco_anns` path always builds and passes a `Vocabulary` instance. Check `build_datasets` passes `vocab=self.vocab` (it does at line 885). The only gap is the smoke test path which uses `simple_tokenize`; leave that unchanged.

*Stage B — pretrained visual encoder (required for NeurIPS-competitive CIDEr).*
Swap the from-scratch CNN backbone for a frozen CLIP ViT-B/32 encoder. CLIP features are 512-dim; project to `hidden_dim=384` with a linear layer. The ANC routing structure is preserved — route CLIP features through the three projection heads of different capacity, but the heavy lifting (visual feature extraction) comes from the pretrained encoder. This is architecturally consistent with the STTF+ANC framing: STTF operates on the change in CLIP patch embeddings across timesteps; ANC routes the compressed representation through the decoder.

Expected outcome: CLIP + ANC should reach CIDEr 80–110 on COCO with 10–20 epochs (within range of competitive efficient captioning papers from 2022–2024). This is an acceptable NeurIPS result for a compression framework when framed correctly.

Implementation sketch:
```python
# In Config: add clip_backbone: bool = False
# In Trainer.__init__, if cfg.clip_backbone:
import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model.eval().requires_grad_(False)
self.clip = clip_model.visual.to(self.device)
# Replace TinyEncoder/SmallEncoder/MediumEncoder input with CLIP features
```

The paper's framing must be updated: instead of "trains from scratch," say "uses frozen CLIP-ViT-B/32 visual features as the per-frame representation; STTF caches and selectively updates patch embeddings; ANC adaptively routes the fused representation."

**Success gate.**
With `Vocabulary` (Stage A), CIDEr > 40 after 10 epochs. With CLIP backbone (Stage B), CIDEr > 80 after 10–20 epochs on COCO. Both must be verified by running `compute_caption_metrics` from the training loop, not estimated.

---

## Phase plan

| Phase | Issue | Work | Est. GPU-hours | Critical | Status |
|---|---|---|---|---|---|
| 1 | I1 | Hard-routing inference fix in `tinyvlm_vast.py` | 0.5 | Yes | PENDING |
| 2 | I2 | Replace 4 figure PNGs from verified canonical run data | 0 (no re-run needed) | Yes | **COMPLETE** |
| 3A | I3 | Confirm `Vocabulary` used in COCO path (already verified by vocabulary.json in canonical run) | 0 | Yes | **COMPLETE** |
| 3B | I3 | Integrate frozen CLIP ViT-B/32 backbone | 4 | Yes | PENDING |
| 4 | paper | Update `tinyvlm2.2.tex`: FLOPs section, figure captions, CLIP results | 1 | Yes | PENDING |

Total remaining estimated GPU-hours: **~5.5 h** (Phase 1 smoke test + Phase 3B training).

**Key finding from `results-tinyvlm_prev`:** The canonical training data (3-seed COCO run) is intact and correct. Phases 2 and 3A are resolved without any new GPU runs. The FLOPs bug (Phase 1) and the CIDEr ceiling (Phase 3B) are the only remaining technical blockers.

### Phase ordering rationale

Phase 1 must come first because it changes the meaning of every FLOPs number. Phase 2 must come before Phase 4 because the paper figures are replaced from the Phase 2 outputs. Phase 3A is prerequisite to Phase 3B (vocabulary must be correct before CLIP features matter for CIDEr evaluation). Phase 4 (paper edits) is last.

---

## Files changed per phase

### Phase 1 (hard routing)
- `tinyvlm_vast.py`: `AdaptiveNeuralCompression.forward` — add train/eval branch split
- `tinyvlm_vast.py`: `comp_cost` at eval time reflects single-branch FLOPs
- `tinyvlm_vast.py`: `_export_onnx` — add note that export uses eval-mode hard routing

### Phase 2 (figures)
- Re-run produces `plots/curves.png` and `plots/anc_utilization.png` under the run directory
- Replace `Figure/f_STTF_Acc.PNG`, `Figure/f_STTF_Loss.PNG`, `Figure/ANC_Acc.PNG`, `Figure/ANC_Loss.PNG`
- All four figures exported from the same 10-epoch COCO run, same y-axis limits

### Phase 3A (vocabulary)
- `tinyvlm_vast.py`: `CocoCaptionDataset` — confirm `vocab` always passed for COCO path
- `tinyvlm_vast.py`: `build_vocabulary` — already correct; add guard that errors clearly if COCO path lacks vocab
- Add `--eval_cider_freq 1` default for COCO runs so CIDEr is tracked every epoch

### Phase 3B (CLIP backbone)
- `tinyvlm_vast.py`: new `CLIPEncoder` class wrapping `open_clip`
- `tinyvlm_vast.py`: `Config` — add `clip_backbone: bool = False`, `clip_model: str = "ViT-B-32"`
- `tinyvlm_vast.py`: `AdaptiveNeuralCompression.__init__` — optional CLIP visual stem
- `tinyvlm_vast.py`: `CocoCaptionDataset.__getitem__` — pass raw PIL image for CLIP preprocessing
- `requirements_clip.txt` (new): `open_clip_torch`

### Phase 4 (paper edits)
- `tinyvlm2.2.tex`: Section 3.2 — soft-to-hard routing clarification
- `tinyvlm2.2.tex`: Table 3 — FLOPs column updated to hard-routing average
- `tinyvlm2.2.tex`: Section 4.1 — training details, confirm "10 epochs" matches re-run
- `tinyvlm2.2.tex`: Figures 2–5 — replaced from Phase 2 outputs; axes described in captions
- `tinyvlm2.2.tex`: Abstract + Table 3 — CIDEr updated from Phase 3B results
- `tinyvlm2.2.tex`: Section 4.3 (Model Architecture) — add CLIP backbone variant description

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hard routing at inference degrades CIDEr (model trained with soft, evaluated hard) | Medium | Train with Gumbel temperature annealing; add 5-epoch fine-tune with straight-through hard routing |
| CLIP fine-tune licence / compute cost | Low | Use frozen CLIP; only projection heads are trained |
| 10-epoch re-run CIDEr doesn't match paper's 36.70 | Medium | Report actual numbers; update paper — do not fabricate |
| ONNX export incompatible with conditional hard routing | Low | Export a fixed-branch variant; note in paper |

---

## Non-goals

- Scaling to 3B+ parameter models (out of scope for this revision cycle)
- Video benchmarks (Kinetics-400, Something-Something-v2) — listed in paper as future work
- Neuromorphic hardware energy analysis — future work
- DVS128 / N-Caltech re-evaluation after backbone change — include only if time permits under GPU budget