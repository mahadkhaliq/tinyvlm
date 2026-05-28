# TNNLS_STATISTICAL_PLAN.md

**Status:** Pre-registered analysis plan for the TNNLS revision. Closes issue **M8**.
**Owner:** Lead. **Created:** 2026-05-28. **Schema version:** 1.

This document declares — *before* the confirmatory event-data results (E1/E2/E4/E11) are
inspected — which metrics are primary, how many seeds back each claim, which significance
tests apply, and how negative results are reported. Every table caption in
`tinyvlm_tnnls_v1.tex` must be compatible with the rules below.

---

## 0. Pre-registration honesty statement (read first)

This plan is **partially retrospective and partially prospective**, and the manuscript must
label each accordingly:

- **Already executed before this plan existed (EXPLORATORY / observational):** the COCO
  Lead-side cycle of 2026-05-25 — E18, E19, E8, E9, E7, E15. These were run and inspected
  *before* this plan was written. They are reported as exploratory observations, not
  confirmatory hypothesis tests. Their p-values (e.g. E7 Welch p=0.084) are descriptive
  effect-size context, **not** pre-registered confirmatory tests.
- **Not yet inspected when this plan was written (CONFIRMATORY / pre-registered):** the
  event-data and video experiments — E1 (DVS128), E2 (MSR-VTT), E4 (CLIP-ANC redesign),
  E11 (5-seed headline sweep). These are genuinely pre-registered by this document. Their
  primary-metric tests, seed counts, and decision thresholds are fixed here and may not be
  changed after results are seen.

Rationale: the COCO findings (routing collapse, dense-beats-ANC) have already redefined the
paper's contribution path. Honesty about *when* the plan was fixed is what protects the
confirmatory event-data results from a cherry-picking critique. Do not retro-fit this plan
to the COCO numbers.

---

## 1. Primary vs secondary metrics (per dataset family)

| Dataset family | Primary metric | Secondary metrics | Notes |
|---|---|---|---|
| COCO captioning | **CIDEr-D** | BLEU-4, ROUGE-L, METEOR, SPICE | METEOR/SPICE require JVM; report only when the scorer runs to completion (see §6). |
| MSR-VTT video captioning (E2) | **CIDEr-D** | BLEU-4, METEOR, ROUGE-L | Also report STTF token-reuse rate (cached/recomputed fraction) per τ ∈ {0.70,0.80,0.90} as a *descriptive* system metric, not a quality metric. |
| DVS128 Gesture (E1) | **top-1 accuracy** | macro-F1 | Report macro-F1 only if class imbalance is visible in the confusion matrix. |
| N-Caltech101 (E4-adjacent, E16) | **top-1 accuracy** | macro-F1 | Same imbalance rule. |
| On-device hardware | **measured latency (ms)** | peak memory (MB), energy/power | Secondary metrics reported **only if physically measured**; never scaled/estimated and presented as measured (closes the C3-style critique). |

A claim is "headline" if it appears in the abstract, an intro contribution bullet, or a
main-table bold cell. Everything else is "supporting" or "exploratory."

---

## 2. Seed policy

| Row class | Seeds required | Seed set |
|---|---|---|
| Headline rows (abstract / main-table bold) | **n = 5** | {42, 43, 44, 45, 46} |
| Supporting rows (main table, non-bold) | **n = 3** | {42, 43, 44} |
| Exploratory ablations | n = 1 allowed | seed 42; **must be labeled "single-seed, exploratory"** in caption |

- E11 upgrades the four headline rows (Dense MediumEnc, TokenLearner, STTF+ANC CNN,
  STTF+ANC CLIP) to n=5 on the PhD cluster. Until E11 lands, the current 3-seed COCO numbers
  are reported as supporting with explicit n=3 in the caption.
- The existing single-seed E15 λ-sweep and E9 soft-vs-hard (seed 42 only) are **exploratory**
  and must be labeled as such.

---

## 3. Confidence intervals

Report **95% CI** as a mean ± half-width using the Student-t interval:

```
half_width = t_{0.975, df=n-1} · s / sqrt(n)
```

| n | df | t₀.₉₇₅ |
|---|---|---|
| 3 | 2 | 4.303 |
| 5 | 4 | 2.776 |

**Small-n caveat (mandatory in the paper):** at n=3 the t-multiplier is 4.303, so CIs are
wide and a single outlier dominates (e.g. E7 Dense-Small CIDEr 42.10 ± 8.24, driven by
seed_43 = 45.93 vs ~40 for the others). Any difference asserted on n=3 must be reported with
its CI shown, and headline differences must be re-confirmed at n=5 (E11) before bolding.

---

## 4. Significance tests

| Comparison setup | Test | Effect size |
|---|---|---|
| Same seeds / same checkpoints across conditions (e.g. soft vs hard on identical ckpts; λ sweep on one seed) | **Paired** two-sided t-test on per-seed deltas | Cohen's **d_z** (mean delta / SD of deltas) |
| Independent runs / different training (e.g. Dense-Small vs STTF+ANC, each trained separately) | **Welch's** two-sided t-test (unequal variance) | Cohen's **d** with pooled SD, **Hedges g** small-sample correction reported alongside when min(n) ≤ 5 |

- Report the test name, t statistic, df, two-sided p, and the effect size for **every**
  asserted difference. No bare "X > Y" without these.
- `scipy.stats.ttest_rel` / `ttest_ind(equal_var=False)`; fall back to the numpy
  implementation in `paired_stats.py` if scipy is unavailable.
- α = 0.05, two-sided, for all confirmatory tests.

---

## 5. Multiple comparisons

- **Confirmatory family** = the primary-metric tests on the pre-registered event/video
  experiments (E1, E2, E4, E11 headline rows). Apply **Holm–Bonferroni** across this family
  and report both raw and adjusted p-values.
- **Exploratory comparisons** (all COCO Lead-cycle observations, λ sweep, per-branch
  breakdowns) are **not** corrected, but every such comparison must carry the word
  "exploratory" in its caption/text so a reviewer cannot read them as confirmatory.
- Do not silently run many ablations and report only the significant ones. The full set of
  comparisons attempted is recorded in `tnnls_claim_ledger.csv`.

---

## 6. Metric-implementation rules

- CIDEr-D, BLEU-1..4, ROUGE-L via **pycocoevalcap** (single pinned version across all rows).
- METEOR and SPICE are JVM-dependent and the pycocoevalcap Meteor wrapper can **deadlock on a
  dead JVM subprocess** rather than raising (observed in E19). Rule: METEOR/SPICE are reported
  only from a run that completed cleanly; a deadlocked/skipped scorer is recorded as "n/a
  (scorer unavailable)" — never imputed, never carried over from another row.
- Greedy decode, max_length = 64, one hypothesis per image_id (dedupe), identical decoding
  settings across every compared row. Decoding settings are logged in `TNNLS_DATA_PROTOCOL.md`.

---

## 7. Negative / null-result rules (pre-committed)

To be fixed *now*, so that none of these can be quietly dropped after the fact:

1. **Routing collapse (E8, E15)** is reported as a primary negative finding in the
   "Routing analysis" subsection — not buried. If E1/E2 event data show non-collapsed
   routing, both results are reported side by side (COCO collapses, events do not).
2. **Dense-Small ≥ STTF+ANC on COCO (E7, +6.08 CIDEr)** is reported in the main comparison
   table with its CI and effect size, regardless of how it reflects on ANC.
3. A pre-registered experiment that **fails its acceptance threshold** (e.g. E1 < 85% top-1,
   or E4 < 0.5 CIDEr significant lift) is reported as a negative result with the threshold
   stated; it is not rerun-until-significant and it is not deleted unless the *whole dataset*
   is rescoped per its plan entry (E1 explicitly permits dropping DVS128 if ≥85% is
   unreachable in the time box — that rescope must be stated, not silent).
4. "Marginal" results (0.05 < p < 0.10, e.g. E7 p=0.084) are described as marginal/not
   significant at α=0.05, never upgraded to "significant."

---

## 8. Acceptance

This plan is satisfied when: every table caption in `tinyvlm_tnnls_v1.tex` states n, the CI
basis, and the test used; every headline difference cites test + effect size; exploratory
rows are labeled; and `tnnls_claim_ledger.csv` traces each numeric claim to a log artifact.

See also: [[project_neurips2026_state]], `TNNLS_DATA_PROTOCOL.md`, `TNNLS_REVISION_PLAN.md`
(E24/E25), `FINDINGS_TNNLS.md`.
