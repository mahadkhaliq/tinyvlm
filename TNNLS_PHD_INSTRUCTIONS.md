# TNNLS PhD Researcher Instructions

**Role:** Second author. Heavy-compute experiments on the university GPU cluster.
**Reads first:** `TNNLS_REVISION_PLAN.md` for context (master plan). Then this file.
**Working tree:** `/path/to/your/clone/of/github.com/mahadkhaliq/tinyvlm`. Branch off `tnnls/integration` for each E-id you start: `git checkout -b tnnls/phd-E1` etc.
**Status board:** `TNNLS_STATE.json` at repo root. Update after every cluster job completes.
**Result format:** every experiment writes into `tnnls_results/E<id>/`. Required contents:
* `seed_<N>.json` — per-seed metrics
* `aggregate.json` — mean, std, 95% CI (use t-distribution with the right df), Cohen's d vs the relevant baseline
* `NOTES.md` — at most one page; what you ran, what came out, what surprised you, what the Lead should know to interpret the table
* `protocol.json` — dataset split, preprocessing, decoding settings, metric implementation, checkpoint commit/hash, hardware, and deviations from the plan
The Lead reads `aggregate.json` directly into the paper — do not re-aggregate on their side.

---

## 0. Setup (Day 1, before any cluster job)

1. **Pull latest:** `git fetch && git checkout tnnls/integration && git pull`.
2. **Read** `tinyvlm_vast.py` end-to-end. Pay attention to the `Config` dataclass — it is the single source of truth for hyperparameters.
3. **Read** `tinyvlm_balanced.nosync/` and `tinyvlm_cider_seeds.nosync/` — these are the baseline training runs from the prior submission. Reuse hyperparameters from there.
4. **Confirm cluster access** + storage quota. Datasets you will need on the cluster:
   * MS COCO 2017 (~25 GB, captioning)
   * MSR-VTT (~6 GB, video captioning — you will download in E2)
   * DVS128 Gesture (~3 GB, event)
   * N-Caltech101 (~2.5 GB, event)
5. **Reserve compute budget**: rough estimate for everything below is ~400–600 A100-hours total. Spread across 6 weeks that is ~10 GPU/day equivalent. Confirm headroom.
6. **Read governance files from Lead before first full run:** `TNNLS_STATISTICAL_PLAN.md` and `TNNLS_DATA_PROTOCOL.md`. If they are missing, run only smoke tests and data-pipeline checks until Lead creates them.
7. **No review-derived numbers.** If a target or baseline value appears only in a review document, treat it as a hypothesis. Your result files must point to logs, released checkpoints, citations, or measured hardware jobs.

---

## E1 — Fix DVS128 accuracy (Weeks 1–3, hard deadline end of Week 3)

**Why:** Closes F1, the single most damaging fatal issue. Currently 28.3% on an 11-class task with random=9.1%. SOTA is 96–99%. The model is barely above 3× chance.

**Diagnosis order (do not skip steps):**
1. **Input pipeline sanity check.** Visualize a batch of event volumes you actually feed into the model. Common bugs: polarity channels swapped or zeroed, time-bin scaling wrong, voxel grid not normalized. If the input looks like noise to your eye, the encoder cannot learn from it.
2. **Loss landscape sanity check.** Train a 2-layer CNN classifier from scratch on the same DVS128 splits, ignore the VLM, just classify. If even a simple CNN can't get >80%, the data pipeline is broken. Fix the pipeline before touching the VLM.
3. **Encoder capacity.** The event-stream encoder in `tinyvlm_vast.py` likely treats events as zero tensors on COCO and is undersized for real event data. Replace the event branch with a small 3D CNN (Conv3D blocks over (T, H, W) voxel grids) or a small SNN (`snntorch`) with ~5M parameters.
4. **Voxel-grid binning.** Use the canonical Maxim/Gehrig 5- or 10-bin polarity-separated voxel grid representation. Cite Gehrig et al. 2019 ("End-to-End Learning of Representations for Async Event-Based Data").
5. **Augmentation.** Random temporal cropping, polarity flipping, spatial jitter — DVS128 is small (1342 samples), augmentation matters more than architecture choice.

**Acceptance targets:**
* **Soft target:** ≥85% top-1 (paper-defensible — within striking distance of weakest SOTA).
* **Hard target:** ≥90% top-1 (parity with the older SOTA tier, closes the issue cleanly).
* **Pivot trigger:** if by end of Week 3 you cannot exceed 70%, *email the Lead immediately*. Lead will execute the "remove DVS128 entirely" pivot — the paper rescopes to N-Caltech101 (event) + MSR-VTT (video) only, and DVS128 disappears from main text and appendix.

**Acceptance criterion (success path):** 3-seed aggregate, mean ≥85% top-1 with reported CI. Confusion matrix as supplementary figure. NOTES.md explains what changed vs the prior submission's encoder.

**Deliverable:** `tnnls_results/E1/{seed_42.json, seed_43.json, seed_44.json, aggregate.json, protocol.json, NOTES.md, confusion_matrix.png}`.

---

## E2 — MSR-VTT video benchmark (Weeks 1–6, ongoing)

**Why:** Closes F4 (STTF on static COCO is conceptually invalid) and M1 (missing video benchmark). Without a video benchmark the entire "temporal token reuse" claim is theatrical.

**Why MSR-VTT and not ActivityNet:** MSR-VTT is the canonical compact-VLM video captioning benchmark, 10K clips, 20 captions each, 1–2 GPU-days per training run. ActivityNet Captions is larger and harder. We want one video benchmark done well, not two done poorly.

**Action plan:**

* **Week 1 — data prep and protocol lock:**
  * Download MSR-VTT from `https://www.mediafire.com/folder/h14iarbs62e7p/shared` (or the more stable HuggingFace mirror `iejMac/CLIP-MSR-VTT`).
  * Extract 30 fps frames at 224×224, save as `frames/<video_id>/<idx:06d>.jpg`.
  * Sample to 8 frames per clip (uniform), or use the standard MSR-VTT 16-frame protocol.
  * Verify the official train (6513) / val (497) / test (2990) split.
  * Write `tnnls_results/E2/protocol.json` before training. Include split source, frame rate, sampled frame count, caption selection rule, tokenizer, decoding, and metric implementation.

* **Week 2–3 — training:**
  * Extend `CocoCaptionDataset` in `tinyvlm_vast.py` into a new `MSRVTTDataset` that returns `(frame_tensor[T,C,H,W], caption_token_ids, real_change_mask[T,N])` where the real change mask comes from frame-difference thresholding at pixel level (`|frame[t] − frame[t−1]| > τ_pixel`).
  * Train STTF+ANC CNN with seeds {42, 43, 44, 45, 46} for 5 seeds — this is the video version of E11 so the seed-5 count is the same effort.
  * Hyperparameters: same as COCO defaults except batch_size halved (memory) and total epochs ~15.

* **Week 4 — eval:**
  * Run `pycocoevalcap` BLEU-4, METEOR, ROUGE-L, CIDEr on test split.
  * Run STTF token-reuse rate measurement: for each test video, log the fraction of tokens cached vs recomputed per frame, sweep τ ∈ {0.70, 0.75, 0.80, 0.85, 0.90}. This is *the headline result* — STTF's first measurement on genuinely temporal data.

* **Week 5 — analysis:**
  * Baseline: train a per-frame Dense baseline (no STTF) on MSR-VTT with the same protocol. STTF+ANC vs Dense should show clear speedup at competitive CIDEr.
  * Optionally: TokenLearner baseline with the same per-frame setup.

**Acceptance:**
* MSR-VTT test CIDEr reported with 5-seed mean ± CI. Does not need to beat video-VLM SOTA (we are compact). Should be within ~30% of Vid2Seq-Tiny or similar compact baseline.
* Token-reuse rate curve across τ — plot α(τ), the active-token fraction, on the y-axis.
* The phrase "STTF is now exercised on genuinely temporal video sequences" appears in the paper §4.
* If you cannot match a recognized MSR-VTT protocol, label the result in `NOTES.md` as a controlled temporal stress test rather than a benchmark. This is not a failure, but the Lead must not overstate it.

**Deliverable:** `tnnls_results/E2/{seed_*.json, aggregate.json, protocol.json, token_reuse_curve.json, NOTES.md}`.

---

## E4 — CLIP-variant ANC redesign with FLOPs gap (Weeks 1–4)

**Why:** Closes F3. Current CLIP ANC has near-identical FLOPs across branches because the CLIP backbone dominates. The "ANC saves 90% FLOPs" claim only applies to the CNN backbone in the current paper.

**Design (locked in master plan §4):**
* **Branch 0** = CLIP-ViT-B/32 with 50% random-token-drop on the patch tokens + 128-d head. Target ≈8 GFLOPs.
* **Branch 1** = full CLIP + 256-d head. Target ≈17.6 GFLOPs.
* **Branch 2** = full CLIP + 384-d head + 2-layer cross-modal MLP. Target ≈19 GFLOPs.

**Action:**
1. In `tinyvlm_vast.py`, find the `CLIPVisualEncoder` / `_CLIPBranchWrapper` classes. Implement a per-branch CLIP variant:
   * For Branch 0, take the CLIP patch embeddings (49 + 1 CLS = 50 tokens for ViT-B/32 at 224×224) and randomly drop 50% of the patch tokens before the transformer encoder. Use a Bernoulli mask sampled per-image, fixed at eval (same seed).
   * Branches 1 and 2 use the standard CLIP forward.
2. First run a forward-pass parity test on 16 images to verify token dropping preserves tensor shapes, positional indexing, and deterministic eval behavior. If CLIP internals make pre-transformer token dropping brittle, switch to an explicit token-pruned CLIP wrapper rather than patching hidden library code.
3. Verify per-branch GFLOPs with `fvcore.nn.FlopCountAnalysis`. Print to log. If actual gap < 1 GFLOP between b0 and b1, the design failed and you need to drop more tokens or apply token-pruning at a deeper layer.
4. Retrain with seeds {42, 43, 44, 45, 46} on COCO. This is the CLIP version of E11 — same seeds.
5. Aggregate routing-weighted GFLOPs and verify it differs from "all branches at b2 FLOPs" by ≥2 GFLOPs.

**Acceptance:**
* Per-branch GFLOPs differ by ≥1 GFLOP at the b0-vs-b1 boundary (acceptance) or ≥3 GFLOPs (target).
* Routing-weighted CIDEr lift over no-routing baseline is statistically significant at α=0.05, n=5 (Welch t-test).
* If both pass: ANC's FLOPs-saving claim extends to CLIP and the paper can lead with this in the CLIP variant.
* If only the FLOPs gap passes but CIDEr lift is not significant: report as "ANC trades capacity for FLOPs at the operating point, with directional but non-significant accuracy effect" — still valuable, but Lead reframes accordingly.
* If neither passes: signal Lead immediately; Lead executes the "restrict FLOPs claim to CNN backbone" fallback.

**Deliverable:** `tnnls_results/E4/{seed_*.json, aggregate.json, protocol.json, flops_breakdown.json, NOTES.md}`.

---

## E5 — Modern compact-VLM baselines on COCO + SD888 (Weeks 2–4)

**Why:** Closes F2 and M2. Current paper compares only to 2021-era token-pruning methods (ToMe, EViT, DynamicViT, TokenLearner). Reviewers will demand a current comparison set.

**Comparison set (lock these four):**
* **MobileVLM** (Chu et al. 2023) — 1.7B params, designed for mobile.
* **TinyLLaVA** (Zhou et al. 2024) — 1.4–3B params, knowledge distillation.
* **MoE-LLaVA** (Lin et al. 2024) — sparse MoE, ~1.6B effective params.
* **MiniCPM-V** (Hu et al. 2024) — 2.5B params.

**Action:**
1. **CIDEr on COCO val:** for each baseline, download the released checkpoint, evaluate on COCO val using the model's own zero-shot caption-generation pipeline. Use the same eval harness as TinyVLM (`pycocoevalcap`).
2. **SD888 latency:** attempt ONNX export for each, submit to AI Hub only if the model is realistically exportable and the Lead approves token use. **Realistic outcome:** most of these will not cleanly ONNX-export at 1B+ params for SD888 NPU — they will fail with op-coverage errors. *That is OK.* In that case:
   * Cite the latency the original paper reports for the same baseline on a comparable mobile/edge device.
   * Footnote: "Latency for [model] cited from original paper as SD888 ONNX export failed at AI Hub validation; see [paper ref]."
   * Mark the table row as `cited`, not `measured`, in `pareto.json`.
3. **Parameter scale table:** add a 4-column table — Model · Params (M) · CIDEr · SD888 latency. TinyVLM appears as a small dot in the bottom-left (low params, low latency, modest CIDEr); baselines as larger dots in the upper-right. This is the Pareto-frontier story.
4. **Pareto plot (Figure 6 candidate):** x-axis = parameters (log scale), y-axis = CIDEr. TinyVLM is the lowest-params point. Caption: "TinyVLM occupies the small-scale corner of the parameters × accuracy Pareto frontier; the gap to billion-parameter compact VLMs reflects scale, not algorithmic deficiency."

**Acceptance:**
* 4 baselines reported on COCO with CIDEr at minimum.
* SD888 latency reported (measured or cited) for ≥2 of them.
* The Pareto figure exists and TinyVLM is visibly the small-scale corner.
* `NOTES.md` explicitly separates direct experimental comparisons from contextual literature comparisons. Do not claim a like-for-like win against a baseline unless it used the same dataset split, decoding, and metric code.

**Deliverable:** `tnnls_results/E5/{<model>_eval.json each, pareto.json, protocol.json, NOTES.md}` plus the figure source.

---

## E11 — 5-seed sweep for headline rows (Weeks 4–5, coordinated with E2 and E4)

**Why:** Closes C4 (insufficient power, n=3 with df=2 has t-critical 4.303, comically wide CIs) and M4 (Cohen's d absent).

**Coordination:** E4 (CLIP redesign retrain) already runs 5 seeds; E2 (MSR-VTT) already runs 5 seeds. E11 only adds 5-seed runs for the *remaining* headline configurations:
* Dense MediumEncoder (3 seeds → 5 seeds — add seeds 45, 46)
* TokenLearner baseline (3 seeds → 5 seeds — add seeds 45, 46)
* STTF+ANC CNN (3 seeds → 5 seeds — add seeds 45, 46)

CLIP STTF+ANC is covered by E4. So this is 6 additional training runs (3 configs × 2 new seeds).

**Action:**
1. Use exact same hyperparameters as the existing 3-seed runs. Read from `tinyvlm_full.nosync/seed_42_tau_0.80/config.json` etc.
2. Push to cluster as a batch job. Each run is ~1 GPU-day.
3. Re-aggregate with n=5: mean, std, 95% CI using t-distribution df=4 (t-critical=2.776, much tighter), and Cohen's d for every pairwise comparison (Dense vs STTF+ANC, TokenLearner vs STTF+ANC).
4. Update `tnnls_results/E11/aggregate.json` to overwrite the prior 3-seed numbers.

**Acceptance:** Every headline row in Tables 1 and 6 has 5-seed mean ± 95% CI + Cohen's d for the pairwise comparison versus its strongest baseline.

**Deliverable:** `tnnls_results/E11/{seed_*.json, aggregate.json, protocol.json, NOTES.md}` covering Dense, TokenLearner, and STTF+ANC CNN at n=5.

---

## E12 — STTF attention fusion module specification + ablation (Week 4)

**Why:** Closes C5. Equation 6 in the current paper introduces an attention module but never specifies its architecture, parameter count, or FLOPs contribution. And it is never ablated — we do not know if the attention is doing anything or could be replaced with concat.

**Action:**
1. **Specify it:** in `tinyvlm_vast.py` find the STTF fusion module. Read off — head count, head dim, total params, FLOPs per call. Write this in `tnnls_results/E12/spec.md`. Lead will copy into §3.2 of the paper.
2. **Three-way ablation, single seed (=42), each 5 epochs on COCO** (this is a small comparative ablation; if signals are clear at 5 epochs the result holds):
   * (a) STTF + learned attention fusion (current, baseline of the ablation)
   * (b) STTF + simple concat fusion (concat active and cached tokens along feature dim → linear projection back to original dim)
   * (c) STTF + zero fusion (cached tokens used directly, no fusion — naive cache)
3. Report final CIDEr + (a) vs (b) Δ and (a) vs (c) Δ.

**Acceptance criterion:** If (a) clearly beats (b) and (c): the attention fusion module is justified and the spec + ablation table go into the paper. If (a) ≈ (b): the paper simplifies to "concat fusion" and Eq. 6 is rewritten. If (a) ≈ (c): STTF works *without* fusion and the entire fusion module gets deleted (good news — simpler model).

**Deliverable:** `tnnls_results/E12/{spec.md, ablation.json, protocol.json, NOTES.md}`.

---

## E16 — N-Caltech101 learning curve (Week 5–6)

**Why:** Closes T3. The Expert Review wants a generalization argument. The cheapest credible version is a learning curve at training-set fractions {25%, 50%, 75%, 100%}.

**Action:**
1. Sample N-Caltech101 train split at fractions 0.25, 0.5, 0.75, 1.0 (the last is the existing baseline). Use stratified sampling — preserve per-class proportions.
2. Train STTF+ANC CNN at each fraction with seed=42 only (single seed; learning curves are about trend, not significance).
3. Plot val accuracy vs training-set fraction with the routing utilization (f_0, f_1, f_2) annotated at each point — does routing change behavior under data scarcity? That is the interesting question.

**Acceptance:** Figure with 4 points; accompanying paragraph in §5 says either "data efficiency holds — routing structure is stable across data scales" or whatever the data actually shows.

**Deliverable:** `tnnls_results/E16/{fraction_25.json, fraction_50.json, fraction_75.json, fraction_100.json, learning_curve.json, protocol.json, NOTES.md}`.

---

## E17 — TokenLearner distillation experiment (Weeks 5–6)

**Why:** Closes C7. The −8.4 CIDEr gap to TokenLearner is the most-likely-to-be-pressed weakness. The Expert Review specifically calls out: "Distillation from TokenLearner or an analysis of representational capacity loss from hard argmax routing would strengthen this substantially."

**Action:**
1. Train a TokenLearner CNN teacher (or use the existing 45.09 CIDEr checkpoint as teacher).
2. Train STTF+ANC CNN with the following added to the loss:
   ```
   L_total = L_CE + λ_KD · KL(student_logits || teacher_logits)
   ```
   with λ_KD ∈ {0.1, 0.3, 1.0}, seed=42 only for the sweep, then the best λ_KD with seeds {42, 43, 44}.
3. Report whether distillation closes ≥30% of the −8.4 CIDEr gap.
4. **Either way the result is a paper contribution.** If distillation closes the gap → "knowledge distillation recovers most of the static-encoder advantage at one-fifth the FLOPs." If distillation does *not* close the gap → "the gap reflects an unavoidable representational cost of hard-argmax routing; distillation alone is insufficient to close it" — interesting because it characterizes the routing trade-off precisely.

**Qualitative companion:** pick 20 COCO val images. For TokenLearner, visualize which 8 learned token positions get selected (heatmap). For STTF+ANC, visualize which tokens are kept active vs cached. Side-by-side panel.

**Acceptance:** Distillation Δ-CIDEr quantified. Qualitative side-by-side figure produced. NOTES.md interprets the result for Lead's discussion section.

**Deliverable:** `tnnls_results/E17/{distillation_sweep.json, distill_3seed.json, qualitative_tokens.png, protocol.json, NOTES.md}`.

---

## What is NOT your job

* Theory derivations (lemmas, bounds) — Lead handles E13, E14.
* Paper text writing — Lead handles every `.tex` edit.
* Architecture audit / parameter table — Lead's E3 (you do not need to verify counts; trust Lead's appendix table).
* AI Hub jobs — Lead has the token quota and runs E10. *Exception:* if E4 / E5 require new AI Hub jobs (CLIP redesign latency, modern-VLM baselines), email Lead with the ONNX file and label; Lead submits and forwards results.
* Cover letter, submission, code release — Lead.

---

## Reporting protocol

1. **Daily-ish:** push to your branch as work happens. Branch naming: `tnnls/phd-E<id>` per experiment.
2. **End of each experiment:** open a PR into `tnnls/integration`. PR body = NOTES.md content. Lead reviews and merges.
3. **Weekly Friday sync:** join the 30-min call. Bring: current cluster job queue status, anything blocked on Lead.
4. **Surprise findings:** if a result contradicts the master plan's assumption (e.g., E1 says DVS128 cannot exceed 60% by Week 2 trajectory), email Lead immediately — do not wait for Friday.
5. **Protocol deviations:** if you change split, preprocessing, decoding, architecture, or metric implementation after the first committed run, record it in `protocol.json` and flag it in the PR body. Silent protocol drift invalidates the result for paper integration.

---

## File ownership summary (your side)

| Path | You write |
|------|-----------|
| `tnnls_results/E1/*` (DVS128) | yes |
| `tnnls_results/E2/*` (MSR-VTT) | yes |
| `tnnls_results/E4/*` (CLIP redesign) | yes |
| `tnnls_results/E5/*` (compact-VLM baselines) | yes |
| `tnnls_results/E11/*` (5-seed sweep, non-CLIP) | yes |
| `tnnls_results/E12/*` (attention ablation) | yes |
| `tnnls_results/E16/*` (learning curve) | yes |
| `tnnls_results/E17/*` (distillation) | yes |
| `tnnls_results/E7/*, E8, E9, E10, E15, E18, E19` | Lead writes — do not touch |
| `tinyvlm_vast.py` for the MSRVTTDataset class and CLIP-redesign branches | yes, push for Lead review |
| `tinyvlm_tnnls_v1.tex` | NO — Lead only |
| `TNNLS_STATE.json` | both write |

---

## Final note on autonomy

You are the second author. Your judgment matters for experiment-design decisions inside your owned experiments. Tell the Lead when an instruction here is wrong for the data you are actually seeing. The master plan's experiment list is the contract; how each experiment is best run is your call.
