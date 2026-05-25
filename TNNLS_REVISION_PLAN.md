# TNNLS Revision Master Plan — TinyVLM

**Target venue:** IEEE Transactions on Neural Networks and Learning Systems (TNNLS)
**Goal:** Submit only after the manuscript can survive a skeptical TNNLS review without a major-revision dependency. This plan cannot guarantee acceptance, but it should eliminate avoidable rejection triggers.
**Inputs analyzed:** `TinyVLM_Review_Journal_Recommendations.md` (recommends TNNLS, 78% readiness) + `TinyVLM_Expert_Review.md` (current score 47/100, 5 fatal issues for any top venue).
**Active manuscript:** `tinyvlm_4.3.tex` (last NeurIPS-formatted draft) → will be rewritten into `tinyvlm_tnnls_v1.tex`.
**Team:** 2 people working in parallel — Lead (Mahbub, paper + vast.ai single GPU + AI Hub) and PhD Researcher 2 (university GPU cluster, heavy retraining).

---

## 1. Why the two reviews disagree, and what we trust

The Journal Recommendations review is venue-fit oriented and reads TinyVLM at face value (78% ready for TNNLS). The Expert Review is methodology-oriented and rates the *evidence* at 47/100, flagging five fatal issues. These are not in conflict: TNNLS is the right venue, *but* the paper as it stands would likely receive a major revision or reject from a methodologically careful TNNLS reviewer. The Expert Review's fatal issues are exactly what a TNNLS reviewer who actually verifies claims would find.

**We plan against the Expert Review's bar, then ship to TNNLS.** This is the only defensible path to avoiding a major revision.

**Provenance warning:** both reviews contain claims that may be stale or unverified relative to the current repository state. For example, any Jetson, power, CLIP, or latency number in a review is advisory until traced to `qai_hub_results/`, `jetson_results/`, a training log, or a paper citation. The revision must never import review prose as evidence.

A note on framing: the Expert Review's "CIDEr 3-4× below SOTA" criticism assumes a comparison set that includes billion-parameter VLMs (BLIP-2, MobileVLM). TinyVLM is a 21M-parameter CNN or a CLIP-ViT-B/32 (≈151M) frontend — the right comparison set is *compact* edge VLMs in the 10M–200M parameter range. We will reframe the contribution scope explicitly and add legitimate compact-VLM baselines in that range rather than chase BLIP-2.

---

## 2. TNNLS-specific acceptance bar (what reviewers expect)

TNNLS publishes "learning systems, adaptive computation, multimodal neural architectures, and edge deployment." For acceptance, a TNNLS paper at this scope typically needs:

1. **One clean theoretical result** (a lemma, a bound, or a convergence argument) tied to the learning mechanism — not necessarily a TPAMI-grade theorem.
2. **Empirical claims that match what's measured** — no fabricated/proxy numbers, every table and figure traceable to a logged experiment.
3. **At least one strong ablation per claimed component**, ideally with controlled negative results.
4. **Hardware deployment evidence beyond simulation** (we have this via AI Hub + Jetson).
5. **Honest statistical reporting** (seeds, CIs, effect sizes).
6. **A coherent positioning argument** — why this contribution matters relative to the right comparison set (compact edge VLMs, not billion-parameter foundation models).
7. **A locked statistical analysis plan** before new results are inspected — primary metrics, seed counts, tests, and correction rules are defined up front to avoid cherry-picking.
8. **Journal-format readiness** — no NeurIPS checklist logic, no camera-ready promises, no unresolved placeholders, and all supplementary material packaged for TNNLS.

---

## 3. Issue inventory — every concern from both reviews, classified

| ID | Source | Severity | Issue |
|----|--------|----------|-------|
| F1 | Expert | Fatal | DVS128 28.3% accuracy (random ≈ 9.1%, SOTA 96–99%) |
| F2 | Expert | Fatal | COCO CIDEr 36.7 (CNN) / 82.24 (CLIP) — below compact-VLM SOTA when honestly framed |
| F3 | Expert | Fatal | ANC CLIP variant has near-identical FLOPs across branches (no compute saving) |
| F4 | Expert | Fatal | STTF on static COCO has no temporal content (event stream zeroed) |
| F5 | Expert | Fatal | CNN-vs-ViT-B/16 architectural contradiction (21M cannot be ViT-B/16) |
| F6 | Journal | Critical | Table 1 latency 18.9 ms ≠ Table 5 measured 24.31 ms (SD888) — internal inconsistency |
| C1 | Expert | Critical | Routing balance may be forced rather than adaptive (uniform ≠ adaptive) |
| C2 | Expert | Critical | Train-time soft routing vs inference hard argmax gap never measured |
| C3 | Expert | Critical | N-Caltech101 latency estimated, not measured |
| C4 | Expert | Critical | Only n=3 seeds; CI on CLIP +0.90 CIDEr spans zero |
| C5 | Expert | Critical | STTF attention fusion module Eq. 6 underspecified/unablated |
| C6 | Journal | High | FLOPs-mismatched baseline (all baselines at 20 GFLOPs vs STTF+ANC at 10) |
| C7 | Journal | Medium | TokenLearner −8.4 CIDEr gap not explained |
| T1 | Expert | Theory | No routing convergence/stability lemma |
| T2 | Expert | Theory | No STTF information-loss bound |
| T3 | Expert | Theory | No generalization argument or learning curve |
| T4 | Expert | Theory | FLOPs target 5 G never achieved (actual ≈10.3 G); λ₂ unanalyzed |
| M1 | Both | Medium | Missing video benchmark (MSR-VTT or ActivityNet) — STTF's core claim |
| M2 | Both | Medium | Missing modern compact-VLM baselines (MobileVLM, TinyLLaVA, MoE-LLaVA, MiniCPM-V) |
| M3 | Journal | Low | Training wall-clock time + token cache memory footprint not reported |
| M4 | Expert | Medium | Cohen's d effect sizes absent alongside p-values |
| M5 | Implicit | Medium | BLEU-4 / METEOR / SPICE not consistently reported across tables |
| M6 | Implicit | Low | Code release at submission (not "on acceptance") |
| M7 | Codex audit | High | Review documents include unverified numbers; all manuscript claims need a source ledger |
| M8 | Codex audit | High | No predeclared statistical plan for multiple seeds, multiple metrics, and many ablations |
| M9 | Codex audit | Medium | Dataset protocol risks: split leakage, preprocessing drift, and inconsistent caption decoding across baselines |
| M10 | Codex audit | Medium | TNNLS packaging missing: journal template, graphical abstract/keywords, supplement boundaries, cover-letter response map |

28 items. Several map to single experiments.

---

## 4. Experiment matrix (E1–E22) — what we will actually run / write

Each entry: **What** · **Why** (issue IDs closed) · **Where** (file) · **Who** (Lead / PhD) · **Effort**.

### Tier-1 — closes the five fatal issues

| ID | What | Closes | Who | Effort |
|----|------|--------|-----|--------|
| **E1** | Diagnose DVS128 event encoder; goal ≥85% top-1 (acceptable) or ≥90% (target). Likely path: stronger event-stream encoder (3D conv on event volumes, or a small SNN), proper voxel-grid binning, real data augmentation. If unachievable in 3 weeks, *remove* DVS128 entirely and rescope the temporal-reuse claim to N-Caltech101 + the new video benchmark only. | F1 | PhD | 3–4 wks |
| **E2** | Add **MSR-VTT** video captioning benchmark (smaller, more tractable than ActivityNet; canonical for compact-VLM eval). Report CIDEr/BLEU-4/METEOR/ROUGE-L + measured STTF token-reuse rate (fraction of tokens cached vs recomputed) per frame across τ = {0.70, 0.80, 0.90}. This is the *only* way to validate STTF's core claim — without it the paper is structurally weak. | F4, M1 | PhD | 3–4 wks |
| **E3** | Architecture audit: every parameter count, layer description, and dimension is verified once across the entire manuscript. Produce a unified architecture appendix table with per-component layer counts/dimensions/parameters. Confirm "CNN encoder, 21M" or "CLIP ViT-B/32 + heads, 151M" everywhere. Delete every stale ViT-B/16 reference. | F5 | Lead | 1 day |
| **E4** | Redesign CLIP-variant ANC so branches genuinely differ in FLOPs. Concrete design: **Branch 0** = CLIP-ViT-B/32 with 50% random-token-drop + 128-d head (≈8 GFLOPs). **Branch 1** = full CLIP + 256-d head (≈17.6 GFLOPs). **Branch 2** = full CLIP + 384-d head + 2-layer cross-modal MLP (≈19 GFLOPs). Retrain with 5 seeds, target ≥1 GFLOP gap routing-weighted vs full CLIP baseline and ≥0.5 CIDEr lift that is statistically significant. | F3 | PhD | 3–5 wks |
| **E5** | Compact-VLM baselines on COCO + Snapdragon 888: **MobileVLM-1.7B**, **TinyLLaVA-1.5B**, **MoE-LLaVA-1.6B×4**, and **MiniCPM-V-1.0** at minimum. Report CIDEr, BLEU-4, params, on-device latency on the same SD 888 device. Position TinyVLM as the smallest-scale point on the Pareto frontier (20–150M vs 1B+). Add explicit Pareto-frontier figure (params × CIDEr or latency × CIDEr). | F2, M2 | PhD | 2–3 wks |

### Tier-2 — closes the critical issues

| ID | What | Closes | Who | Effort |
|----|------|--------|-----|--------|
| **E6** | Fix latency Table 1 (currently 18.9 ms — origin unclear, likely a stale proxy). Either (a) replace with measured SD 888 STTF+ANC routing-weighted 24.31 ms and re-derive any column claiming "speedup" from this, or (b) drop the latency column from Table 1 and have it live only in Table 5. Recommendation: (b), since per-device latency belongs in the dedicated on-device table. | F6 | Lead | 1 day |
| **E7** | Add FLOPs-matched dense baseline: **SmallEncoder-only at ≈10 GFLOPs** (currently we only have MediumEncoder at 20 GFLOPs as Dense). Single training run on vast.ai with 3 seeds. Add row to Table 1 between TokenLearner and STTF+ANC. | C6 | Lead | 4–5 days |
| **E8** | Routing-adaptivity validation: scatter plot of complexity estimator output g_ψ(V_t) vs assigned branch on the COCO val set + per-branch validation accuracy/CIDEr. If correlation is real, this becomes Figure X. If g_ψ outputs are uniform, the "adaptive" claim is removed/softened. | C1 | Lead | 1 wk |
| **E9** | Soft vs hard routing gap measurement: re-run val set on existing CLIP+ANC checkpoint with `model.eval()` under both (a) soft weighted-sum routing and (b) hard argmax routing. Report delta CIDEr/BLEU-4. New "Train-Inference Gap" subsection. | C2 | Lead | 2 days |
| **E10** | N-Caltech101 direct AI Hub profiling: export N-Caltech variants to ONNX, submit to AI Hub on SD 888 + X Elite. ~5 free-tier jobs available on token 2. Replace estimated row in Table 4 with measured numbers. Add new job IDs to the footnote in §4.5. | C3 | Lead | 2 days |
| **E11** | **5-seed sweep** for the headline rows: rerun (a) Dense MediumEnc, (b) TokenLearner, (c) STTF+ANC CNN, (d) STTF+ANC CLIP with seeds {42, 43, 44, 45, 46} on the PhD cluster. Recompute 95% CIs (now t₀.₀₂₅,df=4 = 2.776, narrower interval) and add Cohen's d effect sizes for every reported difference. **Coordinated with E4 retrain** so they share runs. | C4, M4 | PhD | 1–2 wks (overlapped with E4) |
| **E12** | STTF attention fusion module specification + ablation. Specify Eq. 6 fully (it is single-head cross-attention, d=128/256/384 matching branch, parameter count to be reported). Ablation: STTF + learned-attention fusion **vs** STTF + simple concat fusion **vs** STTF + zero fusion (no attention, pure cache). Single training each on cluster. | C5 | PhD | 1 wk |

### Tier-3 — theory + polish

| ID | What | Closes | Who | Effort |
|----|------|--------|-----|--------|
| **E13** | **Routing-stability lemma.** Show that under L_balance with λ₃ ≥ λ\* (compute λ\* from the gradient ratio), the routing utilization satisfies min_k f_k ≥ ε at any local minimum of the total loss, where ε is a function of λ₃ and the cross-entropy gradient magnitude. Sketch in §Theory, full proof in appendix. This is the most important TNNLS-bar item we can produce cheaply. | T1 | Lead | 1 wk |
| **E14** | **STTF information-loss bound.** Bound \|V̂_t − V_t\|₂ as a function of (τ, frame-to-frame feature-space change Δ_t). Use a Lipschitz argument on the per-token feature distance. Tight enough to be quoted in §3, formal statement in appendix. | T2 | Lead | 1 wk |
| **E15** | **FLOPs trajectory analysis.** From existing training logs, extract per-epoch E[FLOPs] curves vs 5 G target budget. Ablate λ₂ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} on a short 10-epoch training each, plot accuracy-vs-FLOPs Pareto. Explains why we land at 10.3 G instead of 5 G and demonstrates the trade-off the loss exposes. | T4 | Lead | 4–5 days |
| **E16** | Learning curve on N-Caltech101 — training-set fraction × {25%, 50%, 75%, 100%} → accuracy. Single sweep on cluster (4 runs). Adds the generalization-data-efficiency story without needing PAC-Bayes. | T3 | PhD | 4 days |
| **E17** | TokenLearner −8.4 CIDEr gap analysis. Two-pronged: (a) qualitative — visualize learned token positions for TokenLearner vs cached-token positions for STTF+ANC on the same 20 COCO val images; (b) quantitative — knowledge-distillation experiment: train STTF+ANC with a TokenLearner-CIDEr-86.9 teacher (KL-divergence loss on logits, weight 0.3). If distillation closes 30%+ of the gap, that becomes a result; if not, that is also a clean qualitative finding. | C7 | PhD | 1.5 wks |

### Tier-4 — paper-side polish (Lead only)

| ID | What | Closes | Who | Effort |
|----|------|--------|-----|--------|
| **E18** | Report training wall-clock time, peak token-cache memory at inference, and ONNX-export model sizes for every variant. Add as Section 4.1 paragraph. | M3 | Lead | 1 day |
| **E19** | Add BLEU-4 / METEOR / SPICE consistently to every captioning table. Use `pycocoevalcap` on existing checkpoints — no new training needed. | M5 | Lead | 2 days |
| **E20** | Reproducibility: clean code release at submission (private GitHub link in cover letter, public at acceptance). Repo must contain training scripts, ONNX export, eval, requirements.txt, exact seeds, and a `RUNS.md` indexing every result table to a logged experiment. | M6 | Lead | 2–3 days |
| **E21** | TNNLS framing rewrite. The paper currently reads as a NeurIPS systems paper; it must read as a *learning-systems* paper. Recast Introduction and §2 to lead with: (i) adaptive computation under hard FLOPs constraints, (ii) MoE-style routing without expert proliferation, (iii) the load-balancing learning dynamic. Move deployment to §6. Compose a one-page cover letter mapping every reviewer's likely concern to the section that addresses it. | TNNLS bar | Lead | 1 wk |
| **E22** | Cross-team integration: assemble all PhD-produced numbers into the new tables, finalize the manuscript, run pdflatex+bibtex, perform a final claim-vs-evidence pass (every numeric claim traced to a log file or analytical derivation). | All | Lead | 4–5 days |

### Tier-5 — submission governance (Lead owns, both follow)

| ID | What | Closes | Who | Effort |
|----|------|--------|-----|--------|
| **E23** | **Claim-source ledger.** Create `tnnls_claim_ledger.csv` with one row per numeric or comparative claim: manuscript location, claim text, source file/job ID/citation, owner, status. Any claim without a source is removed. Review documents are not valid sources. | M7 | Lead | ongoing |
| **E24** | **Statistical analysis plan.** Before reading new results, write `TNNLS_STATISTICAL_PLAN.md`: primary metrics per dataset, seed policy, paired/unpaired tests, CI formula, Cohen's d variant, multiple-comparison handling, and rules for negative results. | M8 | Lead | 1 day |
| **E25** | **Dataset/protocol audit.** For COCO, MSR-VTT, DVS128, and N-Caltech101, record splits, preprocessing, decoding settings, checkpoint provenance, and leakage checks in `TNNLS_DATA_PROTOCOL.md`. PhD supplies dataset-specific details; Lead signs off before integration. | M9 | Both | 1–2 days |
| **E26** | **TNNLS submission package.** Port the manuscript to the current IEEE/TNNLS template, prepare keywords, graphical abstract if required, supplementary appendix, cover letter, conflict/ethics statements, and a reviewer-facing artifact note. | M10 | Lead | 2–3 days |

---

## 5. Dependency graph (critical path)

```
E3  ─┐                                              (architecture audit blocks nothing, do day-1)
E6  ─┤
E10 ─┤
                                                     ┌→ E11 (5-seed) ─┐
E1 (DVS128 rerun)  ──────────────┐                  │                  │
E2 (MSR-VTT video) ──────────────┤                  │                  │
E4 (CLIP redesign retrain) ─────┴── shares cluster ─┘                  │
E5 (compact-VLM baselines) ──────────────────────┐                     │
E12 (attention ablation) ─────────────────────┐  │                     │
E17 (TokenLearner distill) ────────────────┐  │  │                     │
E16 (learning curve) ────────────────────┐ │  │  │                     │
                                          ▼ ▼  ▼  ▼                     ▼
                                      [PhD cluster job queue, parallel] ─→ results.json
                                                                          │
E7  (FLOPs-matched dense)  ── vast ────────────────────────────────────┐  │
E8  (routing adaptivity)   ── analysis on checkpoint ──────────────────┤  │
E9  (soft vs hard)         ── analysis on checkpoint ──────────────────┤  │
E13 (routing lemma)        ── math ────────────────────────────────────┤  │
E14 (STTF bound)           ── math ────────────────────────────────────┤  │
E15 (FLOPs trajectory)     ── analysis on logs ────────────────────────┤  │
E18 (mem + time table)     ── data scrape ─────────────────────────────┤  │
E19 (BLEU/METEOR/SPICE)    ── eval on checkpoints ─────────────────────┤  │
E23/E24/E25 (ledger/stat/protocol) ────────────────────────────────────┤  │
                                                                       ▼  ▼
                                                                       E22 (integration)
                                                                          │
                                                                          ▼
                                                                       E21 (TNNLS rewrite)
                                                                          │
                                                                          ▼
                                                                       E26 package
                                                                          │
                                                                          ▼
                                                                       Submit / no-go
```

**Critical path:** E2 plus either successful E1 or an executed DVS128 removal pivot, plus E4/E11 → E22 → E21 → E26 → submit/no-go.

---

## 6. Timeline (8-week target, 12-week realistic)

| Week | Lead | PhD |
|------|------|-----|
| 1    | E3, E6, E10, E18, E19 | E1 diagnose, E4 design + first runs, E2 dataset prep |
| 2    | E7, E8, E9 | E1 retrain, E4 retrain, E12 design |
| 3    | E13 lemma draft, E15 | E1 finish or pivot, E4 retrain (3 seeds in), E5 baselines load+eval |
| 4    | E14 bound draft, E20 repo prep | E4 (5 seeds done), E5 baselines done, E12 ablation runs |
| 5    | E21 framing rewrite begins, E22 integration starts | E2 retrain (full), E11 sweep, E17 distillation |
| 6    | E21 continues, integration of new tables | E2 done, E16 learning curve, E17 result |
| 7    | E21 finalize, cover letter, code-release tag | Final number QA, results.json freeze |
| 8    | Final compile, claim-vs-evidence audit, **submit** | Standby for any reruns Lead requests |

8 weeks is the earliest credible internal submission window, not a promise. The default planning assumption is 10–12 weeks. Submit in week 8 only if all submission gates in §8 pass with no fallback language that depends on future work.

---

## 7. Risk register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **DVS128 cannot reach ≥85% in available time** | High — undercuts the event-driven narrative | Drop DVS128 from the paper and rescope to "event-based recognition on N-Caltech101" + "video captioning on MSR-VTT". Decided no later than end of week 3. |
| **CLIP-redesign branches still produce flat FLOPs** | High — fails E4 acceptance criterion | Fall back to: "CLIP backbone routing selects representational capacity, not FLOPs" — acknowledge explicitly and demote ANC-CLIP to a secondary contribution; lead with CNN-backbone where ANC genuinely saves compute. |
| **MSR-VTT CIDEr is uncompetitive** | Medium — limits new-benchmark contribution | Report token-reuse rate as the headline (genuine novelty), CIDEr as supporting; compare with simple per-frame baselines, not video-VLM SOTA. |
| **Compact-VLM baselines (MobileVLM etc.) won't ONNX-export to AI Hub** | Medium — breaks like-for-like SD888 column | Report their published latency numbers from their own papers with citations; cleanly footnote that those are author-reported. |
| **Routing-stability lemma turns out non-trivial** | Medium — could blow week 3 budget | Demote to "sketch" with empirical evidence in main body, full proof "in preparation" *removed* per CLAUDE.md instructions about camera-ready promises — just keep the sketch and cite Switch-Transformer for prior treatment. |
| **5-seed runs needed for too many configurations** | Medium — cluster time blow-up | Hard cap: only the 4 headline rows (Dense, TokenLearner, STTF+ANC-CNN, STTF+ANC-CLIP) get n=5. Everything else stays n=3 with the original CIs honestly reported. |
| **MSR-VTT implementation drifts from standard protocol** | High — reviewers may reject the video evidence | E25 records split, frame sampling, decoding, and metrics before training. If protocol cannot be made standard, report MSR-VTT as a controlled temporal-stress test, not a benchmark claim. |
| **Too many ablations create p-hacking appearance** | Medium — weakens trust | E24 locks primary hypotheses and marks all secondary ablations as exploratory. Tables distinguish confirmatory vs exploratory results. |
| **Review-derived numbers enter paper without logs** | High — credibility failure | E23 ledger blocks integration. No ledger source, no manuscript claim. |
| **Token rotation forgotten** | Low security risk | Lead's E20 includes "revoke both AI Hub tokens after submission" as the final step. |

---

## 8. Definition of done — submission gate checklist

A reviewer should be unable to find any of these problems. The Lead runs through the entire checklist before submitting.

- [ ] Every numeric claim in the paper traces to a logged experiment OR a derivation in the appendix.
- [ ] `tnnls_claim_ledger.csv` has no `missing`, `review_only`, or `TBD` sources.
- [ ] `TNNLS_STATISTICAL_PLAN.md` was written before final result integration and is followed in every table.
- [ ] `TNNLS_DATA_PROTOCOL.md` records splits, preprocessing, decoding, and leakage checks for every dataset.
- [ ] Architecture description is internally consistent — only one parameter count per backbone, no ViT-B/16 vs CNN contradictions.
- [ ] Latency in every table is measured (not estimated, not proxied) and the device is named in the caption.
- [ ] STTF is exercised on at least one dataset with genuine temporal content (MSR-VTT or video-N-Caltech) with reported token-reuse rate.
- [ ] ANC's adaptivity claim is validated by g_ψ-vs-branch correlation plot (E8) — or removed.
- [ ] All four headline rows have n=5 seeds with reported CIs *and* Cohen's d.
- [ ] Soft-vs-hard routing gap is reported (E9).
- [ ] CLIP-variant ANC has differentiated FLOPs across branches (E4) OR the FLOPs-saving claim is restricted to CNN.
- [ ] DVS128 is either ≥85% accurate *or* removed from the paper entirely.
- [ ] At least one of: routing-stability lemma (E13), STTF bound (E14) is formally stated in the appendix.
- [ ] Token cache memory + training wall-clock time are reported (E18).
- [ ] BLEU-4 / METEOR / SPICE present in every captioning table (E19).
- [ ] Code-release repo is tagged, README walks through reproducing each table (E20).
- [ ] Cover letter maps each Expert-Review fatal/critical issue to the section that resolves it.
- [ ] Manuscript is in TNNLS/IEEE journal format, not NeurIPS format; checklist language has been removed or moved to supplement if useful.
- [ ] Paper is reframed for TNNLS scope — Introduction leads with adaptive computation, not deployment (E21).
- [ ] No mention of "in camera-ready," "deferred to," or "promised release on acceptance."
- [ ] AI Hub tokens revoked after submission.

---

## 9. Coordination protocol between Lead and PhD

* **Weekly Friday sync (30 min):** review PhD job queue status, surface blockers, refresh week-N task list. Recorded in `TNNLS_SYNC_LOG.md`.
* **Shared state file:** `TNNLS_STATE.json` at repo root, schema:
  ```json
  { "experiments": { "E1": { "status": "in_progress|done|blocked", "owner": "lead|phd",
                              "artifacts": ["path/to/result.json"], "notes": "..." } } }
  ```
  PhD updates after each cluster job, Lead reads before each paper edit.
* **Results handoff format:** PhD writes results to `tnnls_results/<experiment_id>/` containing (1) per-seed JSON metrics, (2) aggregate JSON with mean/std/CI/Cohen's d, (3) a 1-page `NOTES.md` interpreting the result. Lead does *not* re-aggregate — uses PhD's aggregate directly.
* **Protocol handoff format:** every `NOTES.md` must include dataset split, preprocessing, decoding settings, metric implementation, checkpoint commit, and any deviation from the master plan. Lead copies these into E25 before paper integration.
* **Branch policy:** PhD pushes to `tnnls/phd-<experiment>` branches on `github.com/mahadkhaliq/tinyvlm`. Lead pushes paper edits to `tnnls/paper`. Weekly merge into `tnnls/integration` for final assembly.
* **Conflict resolution:** Lead has final say on paper text and table numbers. PhD has final say on experimental methodology. Disagreements escalate to in-person before any silent change.

---

## 10. What the codex reviewer should check in our instructions

When the codex agent audits our two instruction files, the items most worth scrutiny are:

1. **Does every assigned experiment have unambiguous acceptance criteria?** (We have written explicit ≥X% / ≥Y CIDEr / ≥Z GFLOP gap targets.)
2. **Are dependencies honored?** (E4 must precede E11; E1 must conclude by week 3 or pivot; E22 cannot start until all PhD numbers are in.)
3. **Is the work load roughly balanced?** Lead has more paper-text items (E3, E13, E14, E21) which are time-intensive but low-compute; PhD has more compute-intensive items (E1, E2, E4, E5, E11). The split exploits the asymmetry in hardware access.
4. **Are escape hatches in place for failures?** (Risk register §7 explicitly lists pivots for E1, E4, E2.)
5. **Is the timeline realistic?** 8 weeks is an aggressive internal-submit target; 10–12 weeks is the default expectation. Check whether any critical-path experiment has less than a week of slack.
6. **Is anything double-assigned or missed?** Every E1–E22 has exactly one owner; the Lead picks up the integration (E22) explicitly.
7. **Are source ledgers and statistics locked early enough?** E23/E24/E25 must start in week 1, not after results are known.
