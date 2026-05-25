# TNNLS Lead Instructions — Mahbub

**Role:** First author. Paper writing, theory, vast.ai single-GPU experiments, Qualcomm AI Hub, analysis on existing checkpoints.
**Reads first:** `TNNLS_REVISION_PLAN.md` for context. Then this file.
**Working tree:** `/Users/mahbub/Documents/projects/tinyvlm/` (NOT the nested `tinyvlm/tinyvlm/` workspace — confirm before editing anything in there).
**Status board:** `TNNLS_STATE.json` at repo root. Update after every completed item.

---

## 0. Setup (Day 1, before any experiment)

1. **Confirm active repository.** The current top-level workspace may not be a Git checkout; the nested `tinyvlm/` directory is a separate tree. Run `git status` in the tree you intend to edit. If both root and nested copies exist, confirm with the collaborator before touching duplicated files.
2. **Branch:** `git checkout -b tnnls/paper` on the active repo. All paper-side edits live here.
3. **Initialize state file:**
   ```bash
   echo '{"experiments":{}}' > TNNLS_STATE.json
   ```
   then add one entry per E-id you own as you start each item.
4. **Clone manuscript:** `cp tinyvlm_4.3.tex tinyvlm_tnnls_v1.tex`. From this point forward, every paper edit happens in `tinyvlm_tnnls_v1.tex`. Do not touch `tinyvlm_4.3.tex` (frozen reference).
5. **Create governance files before results are inspected:**
   * `TNNLS_STATISTICAL_PLAN.md`
   * `TNNLS_DATA_PROTOCOL.md`
   * `tnnls_claim_ledger.csv`
6. **Reading checklist:** finish reading the Expert Review and the Journal Recommendations end-to-end, then re-read §3 (Issue Inventory) of the master plan. Treat review numbers as leads, not evidence.

---

## E23 — Claim-source ledger (starts Day 1, maintained until submission)

**Why:** Closes M7. The reviews contain useful criticism but also stale or unverified numbers. A TNNLS paper cannot inherit any value unless it traces to a log, job ID, derivation, or citation.

**Action:**
1. Create `tnnls_claim_ledger.csv` with columns:
   `section,claim_type,claim_text,value,source_path_or_citation,job_id_or_commit,owner,status`.
2. As you edit each section, add every numeric claim and every comparative claim ("faster than", "lower FLOPs than", "statistically significant").
3. Valid source types: result JSON, training log, AI Hub job ID, Jetson result file, derivation in appendix, or cited external paper. Invalid source types: expert review prose, memory note, TODO, verbal recollection.
4. Before integration, filter for `status != verified`; remove or recompute every row that remains.

**Acceptance:** No manuscript number lacks a verified ledger row.

**Deliverable:** `tnnls_claim_ledger.csv` committed and updated with every paper-facing change.

---

## E24 — Statistical analysis plan (Day 1, before new results)

**Why:** Closes M8. With many ablations and metrics, reviewers can read the paper as cherry-picked unless the analysis rules are explicit.

**Action:**
1. Write `TNNLS_STATISTICAL_PLAN.md` before looking at new E1/E2/E4/E11 results.
2. Declare primary metrics:
   * COCO/MSR-VTT: CIDEr primary; BLEU-4, METEOR, SPICE secondary.
   * DVS128/N-Caltech101: top-1 accuracy primary; macro-F1 secondary if class imbalance is visible.
   * Hardware: measured latency primary; peak memory and energy/power secondary only if measured.
3. Declare seed policy: n=5 for headline rows, n=3 for supporting rows, single-seed exploratory ablations clearly labeled.
4. Declare tests: paired tests when same seeds/checkpoints are compared, Welch tests otherwise; report 95% CI and Cohen's d. State how multiple comparisons are handled, at minimum by labeling non-primary ablations exploratory.

**Acceptance:** Every table caption is compatible with the plan.

**Deliverable:** `TNNLS_STATISTICAL_PLAN.md`.

---

## E25 — Dataset/protocol audit (Week 1, updated by integration)

**Why:** Closes M9. The revision depends on multiple datasets and baseline families; split leakage or decoding drift would be a major-review trigger.

**Action:**
1. Create `TNNLS_DATA_PROTOCOL.md`.
2. For each dataset, record split source, preprocessing, frame sampling or voxelization, tokenizer/decoding settings, checkpoint source, metric implementation, and leakage checks.
3. Require every PhD `NOTES.md` to include the same fields before results are integrated.
4. For compact-VLM baselines, distinguish measured numbers from author-reported numbers in both the protocol file and the manuscript table footnotes.

**Acceptance:** A reviewer can reproduce the evaluation protocol without reading private chat history.

**Deliverable:** `TNNLS_DATA_PROTOCOL.md`.

---

## E3 — Architecture audit (Day 1, ~6 hours)

**Why:** Closes F5. The Expert Review will not even consider the rest of the paper if a ViT-B/16-vs-21M-CNN contradiction is allowed to stand.

**Action:**
1. `grep -n -iE "vit-?b/?(16|32)|21M|encoder|param" tinyvlm_tnnls_v1.tex checklist.tex` — capture every architectural claim with line numbers.
2. Reconcile with the code (`tinyvlm_vast.py` is the source of truth — read `Config` dataclass + `TinyEncoder`/`SmallEncoder`/`MediumEncoder`/`CLIPVisualEncoder` modules). Compute parameter counts via `sum(p.numel() for p in module.parameters())`.
3. Add a new appendix table `tab:arch` with columns: Module · Layer count · Hidden dim · #Params (M). One row per branch (Tiny / Small / Medium) and per backbone variant (CNN / CLIP).
4. Edit every prose mention of the architecture to match exactly one of: "21M-parameter scratch CNN encoder" or "frozen CLIP ViT-B/32 (≈151M) + lightweight MLP heads." Delete every ViT-B/16 reference.
5. Update `checklist.tex` if it still says "ViT-B/16."

**Acceptance:** `grep -n "vit-b/16\|ViT-B/16" tinyvlm_tnnls_v1.tex checklist.tex` returns empty. The Architecture table exists in the appendix. Reviewers can pick any architectural claim in the paper and trace it to `tab:arch`.

**Deliverable:** Commit `architecture: unified per-component param table; remove ViT-B/16 contradiction (closes F5)`.

---

## E6 — Fix latency Table 1 (Day 1–2, 4 hours)

**Why:** Closes F6. Table 1 currently says STTF+ANC latency = 18.9 ms with no device; Table 5 says SD888 STTF+ANC = 24.31 ms (measured). One number must go.

**Recommended action:** Remove the latency column from Table 1 entirely. Per-device latency belongs in Table 5 (the dedicated on-device table). Replace the column with a "Notes" column referencing the measured row in Table 5.

**Alternative (if you must keep latency in Table 1):** Replace 18.9 with the SD888 routing-weighted 24.31 ms, and add to the caption: "Latency measured on Snapdragon 888 (Galaxy S21) via AI Hub; see Table 5 for full device breakdown." Recompute any "speedup" cell that derived from 18.9.

**Acceptance:** No two latency cells in the paper disagree about the same (device, variant) pair.

**Deliverable:** Commit `tables: reconcile Table 1 latency with measured Table 5 (closes F6)`.

---

## E10 — N-Caltech101 direct AI Hub profiling (Day 2–3, ~2 days)

**Why:** Closes C3. Table 4 currently *estimates* N-Caltech101 latency by scaling SD 888 captioning latency by FLOPs ratio. That is not a measurement.

**Action:**
1. Extend `qai_hub_bench.py` `_build_model` to accept variant `ncaltech_anc` (load N-Caltech-trained checkpoint, build per-branch wrapper exactly like the existing `sttf_anc` path — the architecture is identical, only the head differs for classification).
2. Export 3 branches + 1 dense baseline to ONNX (static shapes, no dynamic axes — same gotchas as `project-aihub-cycle` memory).
3. Submit to AI Hub: SD 888 (4 jobs) + X Elite (4 jobs). ~9 free-tier jobs available on token 2 from `~/.zshrc` (`$QAI_HUB_API_TOKEN`).
4. Aggregate routing-weighted using the same (0.31, 0.34, 0.35) weights from training.
5. Replace the estimated row in Table 4 with the measured one. Update caption to say "measured via Qualcomm AI Hub" and remove the FLOPs-ratio scaling language. Add the new job IDs to the footnote in §4.5.

**Acceptance:** Table 4 latency column footer reads "measured" not "estimated." `qai_hub_results/jobs.json` contains new entries for `ncaltech_*`. Job IDs are cited in the paper.

**Deliverable:** Commit `ncaltech: measured on-device latency replaces estimate (closes C3)`.

---

## E7 — FLOPs-matched dense baseline (Day 4–7, ~4 days)

**Why:** Closes C6. Reviewers will say STTF+ANC at 10.3 GFLOPs vs Dense MediumEncoder at 20 GFLOPs is unfair. Add a Dense SmallEncoder-only at ~8–10 GFLOPs.

**Action:**
1. On vast.ai (existing GPU instance), train Dense SmallEncoder on COCO with seeds {42, 43, 44}. Use exact same hyperparameters as the existing Dense MediumEncoder runs in `tinyvlm_full.nosync/` — same epoch count, batch size, learning rate. Only the encoder branch changes.
2. Command template (adjust paths to vast instance):
   ```bash
   python tinyvlm_vast.py \
       --coco_imgs /workspace/coco/train2017 \
       --coco_anns /workspace/coco/annotations/captions_train2017.json \
       --encoder_only small --baseline dense --seeds 42,43,44 \
       --epochs 10 --output_dir /workspace/runs/dense_small_flops_matched
   ```
   If `--encoder_only` flag does not exist, add it as a CLI flag bound to a `Config` field that selects which single branch to instantiate. Plumb through `tinyvlm_vast.py`. About 30 lines of code change.
3. Aggregate CIDEr/BLEU-4/METEOR/SPICE mean ± 95% CI + Cohen's d vs STTF+ANC.
4. Insert new row into Table 1 of `tinyvlm_tnnls_v1.tex` between TokenLearner and STTF+ANC.

**Acceptance:** Table 1 contains a row "Dense SmallEnc" at ~8–10 GFLOPs that allows a like-for-like FLOPs comparison with STTF+ANC. The row has 3-seed CI. Vast instance disk is cleaned (delete intermediate checkpoints, keep only `summary.json`).

**Deliverable:** Commit `baseline: FLOPs-matched Dense SmallEnc row (closes C6)`. Push checkpoint summary JSON to `tnnls_results/E7/`.

---

## E8 — Routing-adaptivity validation (Week 2, ~1 week)

**Why:** Closes C1. The Expert Review's deepest skepticism is that uniform routing ≠ adaptive routing. We need to either show adaptivity exists or stop claiming it.

**Action:**
1. Load best CNN STTF+ANC checkpoint (`tinyvlm_full.nosync/.../best.pt`). On COCO val set, hook the complexity estimator g_ψ output and the assigned branch for every sample.
2. Save (g_ψ scalar, assigned_branch) pairs to a numpy file. ~5k samples.
3. Compute Spearman correlation between g_ψ value and branch index. Plot a 2-D scatter with kernel-density overlay.
4. Compute per-branch validation CIDEr and BLEU-4 on the subset of samples actually routed to each branch (not the uniform val set). This shows whether the branches have learned different competencies.
5. Repeat for CLIP STTF+ANC.

**Decision rule:**
* If |Spearman| ≥ 0.2 and the per-branch CIDEr difference > the 95% CI of the headline result: keep the "adaptive" claim, add the figure as Figure 3.
* If correlation is essentially zero: replace "adaptive routing" prose with "learned load-balanced routing" everywhere, and report this as a clean negative result in §5 ("routing distributes load but does not specialize on input complexity in our experiments"). Negative results are valued at TNNLS.

**Acceptance:** New figure `fig:routing_adaptivity` exists in `tinyvlm_tnnls_v1.tex` with a clear caption stating what the figure shows and what conclusion follows.

**Deliverable:** Commit `routing: g_psi vs branch scatter + per-branch eval (closes C1)`.

---

## E9 — Soft vs hard routing gap (Day 8, ~2 days)

**Why:** Closes C2. Training uses soft weighted sum; inference uses hard argmax. The gap was never measured.

**Action:**
1. Load best STTF+ANC checkpoint. In `tinyvlm_vast.py`, add an `--eval_routing_mode {soft,hard}` flag that toggles the forward path.
2. Run eval twice on COCO val (CNN backbone) and once each on CLIP val. Report delta CIDEr / BLEU-4.
3. Add a new paragraph in §3.4 titled "Train–inference routing gap" with the measured numbers.

**Acceptance:** Paper reports a single number "Δ CIDEr (soft − hard) = X.XX" for each backbone, with one sentence of interpretation.

**Deliverable:** Commit `routing: measure train-soft vs inference-hard gap (closes C2)`.

---

## E13 — Routing-stability lemma (Week 3, ~1 week)

**Why:** Closes T1. This is the single biggest theory deliverable for TNNLS acceptance.

**Goal:** State and prove (or sketch + appendix-prove) a lemma of the form:

> **Lemma 1 (Routing utilization lower bound).** Let f_k be the empirical utilization of branch k on a training batch. Under the load-balancing loss L_balance = K · Σ_k f_k · P_k (Shazeer et al. notation), the total objective L_total = L_CE + λ₃ L_balance has the property that any local minimum satisfies min_k f_k ≥ ε(λ₃, ∥∇L_CE∥), where ε is monotonically increasing in λ₃.

**Proof sketch (your job to make this rigorous):**
1. Take ∂L_total/∂f_k = 0 at a local minimum. Solve for f_k.
2. Show the cross-entropy gradient pulls toward a single dominant branch; the L_balance gradient pulls toward 1/K.
3. The equilibrium f_k* is the convex combination weighted by λ₃.
4. Establish that λ₃ ≥ ∥∇L_CE∥ / (K−1) is sufficient to ensure f_k > 0 for all k.

**Action:**
1. Draft the lemma + sketch in §3.3 of the paper (1 paragraph + 1 equation).
2. Full proof goes in Appendix B.
3. Empirically validate: read training logs of the routing-collapse ablation (`tinyvlm_balanced.nosync/`) and plot ε(λ₃) measured at convergence vs ε(λ₃) predicted by the lemma. Add as Figure 4.

**Acceptance:** Lemma 1 stated formally, sketch in main text, full proof in appendix, empirical validation figure. A reviewer with basic optimization background can follow the proof.

**Deliverable:** Commit `theory: routing-stability lemma + empirical validation (closes T1)`.

**Fallback if proof is non-trivial:** Demote to an empirical observation with a heuristic argument citing Shazeer et al. 2017 and Fedus et al. 2022 for the Switch Transformer prior treatment. *Do not promise a future formal proof* — no "in preparation" hedge.

---

## E14 — STTF information-loss bound (Week 4, ~1 week)

**Why:** Closes T2. Pairs with E13 — the paper benefits from one theorem per core component.

**Goal:** Bound the L2 distance between the exact token representation V_t and the STTF-approximated V̂_t as a function of τ and frame-to-frame change magnitude.

**Approach (Lipschitz argument):**
1. Define Δ_t = ∥V_t − V_{t−1}∥_F (Frobenius norm of frame-to-frame feature change).
2. Per-token: a token is reused if its detected change is below τ. Bound the reused-token error by τ · ∥V_t∥_∞.
3. Active tokens are recomputed exactly (zero error contribution).
4. Aggregate: ∥V̂_t − V_t∥_F ≤ τ · √(N_reused) · ∥V_t∥_∞ where N_reused = (1 − α_t) · N and α_t is the active-token fraction at time t.
5. Combine with attention-fusion module's contraction constant (E12 will give us this — coordinate with PhD).

**Action:**
1. Theorem statement + proof in Appendix C.
2. One paragraph in §3.2 quoting the bound and discussing the τ→0 (exact) and τ→∞ (full reuse) regimes.
3. Empirically validate: on MSR-VTT (when PhD E2 lands), measure actual ∥V̂_t − V_t∥_F vs the bound across τ ∈ {0.7, 0.8, 0.9}. Plot tightness.

**Acceptance:** Theorem stated, proved, and empirically validated within a factor of ≤2 of measured error. If the bound is loose by more than ×5, weaken claims and present as a "first-order analysis" rather than a tight bound.

**Deliverable:** Commit `theory: STTF information-loss bound (closes T2)`.

---

## E15 — FLOPs trajectory + λ₂ ablation (Week 3, ~5 days)

**Why:** Closes T4. The paper says target=5G but achieves ~10G. Reviewers will ask why.

**Action:**
1. Scrape per-epoch E[FLOPs] from existing training logs in `tinyvlm_full.nosync/`, `tinyvlm_balanced.nosync/`. Plot E[FLOPs] vs epoch with a horizontal line at 5G target.
2. On vast.ai, run a small λ₂ ablation: λ₂ ∈ {0.01, 0.05, 0.1, 0.5, 1.0}, single seed, 10 epochs each, CNN backbone. Measure (final CIDEr, final E[FLOPs]) for each setting.
3. Plot the resulting Pareto frontier — accuracy on y, FLOPs on x, one point per λ₂.
4. Add Figure 5 + a §3.5 paragraph "FLOPs-accuracy trade-off via λ₂" explaining that the default λ₂=0.1 was selected for accuracy retention but the entire frontier is exposed by the loss.

**Acceptance:** Reviewers can answer "why didn't you reach 5G?" by pointing to Figure 5 themselves — at higher λ₂ we *do* reach lower FLOPs, at the cost of CIDEr. The 5G target was an aspirational lower bound, not a binding constraint.

**Deliverable:** Commit `theory: lambda_2 Pareto frontier + 5G discussion (closes T4)`.

---

## E18 — Wall-clock + cache memory (Day 1, ~1 day, fits in slack)

**Why:** Closes M3.

**Action:**
1. From training logs scrape: GPU hours per training run for each headline configuration. Report as a table column in §4.1.
2. From inference profile: peak token-cache memory at runtime. Easy if you instrument `STTFCache` with `torch.cuda.max_memory_allocated()` around the forward pass on a 100-frame synthetic sequence.
3. ONNX-export model sizes: just `ls -lh models/onnx/`.
4. Add a small paragraph in §4.1: "Training each STTF+ANC CNN run takes ≈X GPU-hours on a single A100. Peak token-cache memory during inference is ≈Y MB at τ=0.8 on a 100-frame window. ONNX-exported models are Z MB."

**Acceptance:** Three numbers in §4.1, sourced and labeled.

**Deliverable:** Commit `paper: training time + cache memory + ONNX sizes (closes M3)`.

---

## E19 — BLEU-4 / METEOR / SPICE everywhere (Day 5, ~2 days)

**Why:** Closes M5. CIDEr alone reads as cherry-picked; the field reports the four together.

**Action:**
1. `pycocoevalcap` is already in `requirements.txt`. Write `tnnls_eval.py`:
   ```python
   # Loads a checkpoint, runs greedy beam=5 caption generation on COCO val,
   # then runs pycocoevalcap to compute BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE.
   ```
2. Run it once per headline checkpoint (Dense Medium, Dense Small, TokenLearner, STTF+ANC CNN, STTF+ANC CLIP) using existing checkpoints. No new training.
3. Add BLEU-4 / METEOR / SPICE columns to Table 1 and Table 6 (CLIP).

**Acceptance:** Every COCO captioning table reports {BLEU-4, METEOR, CIDEr, SPICE} at minimum.

**Deliverable:** Commit `eval: BLEU-4 + METEOR + SPICE on all captioning tables (closes M5)`. Push `tnnls_eval.py`.

---

## E20 — Code release prep (Week 4, ~3 days)

**Why:** Closes M6. TNNLS reviewers may check the repo if a link is included; absence of code at submission is a negative signal.

**Action:**
1. Fork or clone `github.com/mahadkhaliq/tinyvlm` to a fresh public-ready repo `tinyvlm-tnnls-release`. Initially private; flip to public on acceptance.
2. Top-level `README.md` walks through reproducing each table:
   * Table 1: `bash reproduce/table1.sh` runs Dense+TokenLearner+STTF+ANC sweeps with 5 seeds.
   * Table 4: `bash reproduce/table4_ncaltech.sh`.
   * Table 5: `bash reproduce/table5_ondevice.sh` (calls AI Hub or instructs Jetson run).
   * Each script ends with the exact `python` invocation that produces the result JSON.
3. `RUNS.md` is a flat index: every result in the paper → script that generates it → log file produced.
4. `requirements.txt` pinned exactly.
5. Add MIT license. Strip any token from history (`git filter-repo --invert-paths --path .qai_hub/`).
6. Do not promise future code release in the manuscript body. In the cover letter or artifact note, say that a reviewer-accessible repository is available during review and will remain archived with the accepted article.

**Acceptance:** A fresh-cloned reviewer can run `bash reproduce/table1.sh` end-to-end with no manual intervention beyond editing two path variables. The repository history contains no API tokens, large checkpoints, or private cluster paths.

**Deliverable:** Repo URL recorded in `TNNLS_STATE.json`. Commit `release: reproducibility repo skeleton (closes M6)`.

---

## E21 — TNNLS framing rewrite (Weeks 5–7, ~2.5 weeks)

**Why:** This is the most important deliverable on your plate. A NeurIPS-style framing does not pass TNNLS.

**Differences to enforce:**
* TNNLS opens with the *learning-systems* angle (adaptive computation, MoE routing, load-balancing learning dynamics), not deployment.
* TNNLS likes formal definitions and lemmas, even short ones. Add a `Definition` environment for STTF and for the ANC router; refer to them by name throughout.
* TNNLS expects related-work coverage of MoE literature (Switch Transformer, GLaM, Mixtral, Soft MoE) explicitly — half of §2 should be MoE-routing-for-vision, not token-pruning.
* Deployment evidence becomes §6 ("Edge Deployment Case Study") instead of being a primary contribution. The headline contributions are (1) the learning algorithm, (2) the theoretical guarantees, (3) the empirical validation on five datasets including video.

**Action:**
1. Rewrite the Introduction in 3 paragraphs:
   * P1 motivates adaptive computation in vision-language learning (cite Switch Transformer, AdaTape, MoE-LLaVA).
   * P2 introduces STTF and ANC as a *learning framework*, not a deployment trick.
   * P3 enumerates 4 contributions: (a) STTF token-reuse algorithm with information-loss bound, (b) ANC adaptive routing with stability lemma, (c) empirical validation across image, event, and video benchmarks, (d) measured edge deployment as a case study.
2. Restructure section order to: 1 Intro → 2 Related Work (with MoE depth) → 3 Method (STTF + ANC + Theory) → 4 Experiments (COCO, MSR-VTT, N-Caltech, optional DVS128) → 5 Analysis (ablations + adaptivity + routing gap) → 6 Edge Deployment Case Study → 7 Discussion.
3. Rewrite the abstract last, once everything else is final. Target: explicit mention of adaptive computation, formal bounds, video benchmark, edge measurement. Avoid "up to X%" superlatives.
4. Cover letter (1 page) maps each Expert Review fatal issue to the section that addresses it. Include a one-line list of new experiments since the prior draft.

**Acceptance:** A reviewer reading only the Introduction can state what's new at a TNNLS level (a learning algorithm with theoretical guarantees applied to a multimodal task, validated across multiple datasets including video, with edge-deployment evidence). The deployment angle is supporting, not leading.

**Deliverable:** Commits per section. Final commit `paper: TNNLS framing rewrite complete (E21)`.

---

## E22 — Cross-team integration + claim-vs-evidence audit (Week 8, ~5 days)

**Why:** Catches anything that fell through the cracks.

**Action:**
1. Pull PhD's `tnnls_results/E1, E2, E4, E5, E11, E12, E16, E17` into the paper only after each folder contains `aggregate.json`, `NOTES.md`, split/preprocessing details, and commit/checkpoint provenance.
2. Walk every numeric claim in the paper and update `tnnls_claim_ledger.csv`: claim → source (log file / derivation / external citation). Anything you cannot source gets removed or recomputed.
3. Compile-check: `pdflatex tinyvlm_tnnls_v1 && bibtex tinyvlm_tnnls_v1 && pdflatex tinyvlm_tnnls_v1 && pdflatex tinyvlm_tnnls_v1`. Zero undefined references, zero overfull hboxes in the main text.
4. Final §8 Definition-of-Done checklist sweep (see master plan §8), including statistical-plan and data-protocol compliance.
5. Do not submit if any Tier-1 fatal issue remains unresolved by data or by an explicit manuscript pivot.
6. Submit via TNNLS Manuscript Central. Confirm cover letter, supplementary appendix, artifact note, and any required IEEE metadata are attached.
7. After submission: revoke both AI Hub API tokens at aihub.qualcomm.com (per `security-tokens` memory).

**Acceptance:** Manuscript submitted. Confirmation email archived.

**Deliverable:** Tag the release commit `tnnls-submission-v1`.

---

## Coordination handles you own

* You initiate every Friday sync. Bring the PhD's `tnnls_results/` aggregates and your paper-side checklist.
* You write final paper text. PhD does not edit the manuscript directly.
* If PhD's E1 (DVS128) hits the 3-week deadline without ≥85%, *you* call the pivot to "remove DVS128 entirely." That decision sits with you.
* If E4 (CLIP redesign) cannot produce a FLOPs gap, *you* call the fallback to "ANC selects capacity, not compute" framing and restrict the FLOPs claim to CNN backbone.
* If E2 (MSR-VTT) cannot be completed under a standard protocol, *you* decide whether to delay submission or present it as a temporal stress test. Do not call it a benchmark unless E25 protocol requirements are met.
* You own the go/no-go decision. A week-8 submission is allowed only if every fatal issue is closed and the ledger has no missing sources.

---

## Things that are NOT your job

* Heavy retraining (DVS128, MSR-VTT, CLIP-redesign, modern-VLM baselines, 5-seed sweep, attention-fusion ablation, distillation, learning curve) — all PhD.
* Running compact-VLM baselines (MobileVLM, TinyLLaVA, etc.) — PhD on cluster.
* Generating per-experiment NOTES.md interpretations — PhD writes those; you integrate them into the paper.

---

## File ownership summary

| File / dir | Owner |
|-----|-------|
| `tinyvlm_tnnls_v1.tex` (manuscript) | Lead |
| `checklist.tex` | Lead |
| Appendix proofs | Lead |
| Cover letter | Lead |
| `tnnls_claim_ledger.csv` | Lead |
| `TNNLS_STATISTICAL_PLAN.md` | Lead |
| `TNNLS_DATA_PROTOCOL.md` | Lead owns, PhD supplies experiment sections |
| `qai_hub_bench.py` + `qai_hub_results/` | Lead |
| `TNNLS_REVISION_PLAN.md` (master) | Both — read-only after Day 1 |
| `TNNLS_STATE.json` | Both write |
| `tnnls_results/E1, E2, E4, E5, E11, E12, E16, E17` | PhD writes, Lead reads |
| `tnnls_results/E7, E8, E9, E10, E15, E18, E19` | Lead writes |
| `reproduce/` scripts | Lead |
| Code release repo | Lead |
| `tinyvlm_vast.py` for `--encoder_only` and `--eval_routing_mode` flags | Lead writes, PhD pulls latest before any new run |

If the PhD needs a `tinyvlm_vast.py` change for E11 or E4 retrains, Lead lands it within 24h of the request — PhD's cluster time is the scarce resource.
