# TNNLS_DATA_PROTOCOL.md

**Status:** Dataset & evaluation-protocol audit for the TNNLS revision. Closes issue **M9**.
**Owner:** Lead owns the file and signs off; **PhD fills the dataset-specific sections**
(DVS128, MSR-VTT, N-Caltech101) before any of those results are integrated.
**Created:** 2026-05-28. **Schema version:** 1.

Goal: a reviewer can reproduce every evaluation in the paper **without reading private chat
history**. Each dataset section records: split source, preprocessing, frame
sampling/voxelization, tokenizer & decoding settings, checkpoint provenance, metric
implementation, and leakage checks. Every PhD `NOTES.md` must repeat these fields before
results are accepted into a table.

---

## Required fields (template — copy per dataset)

```
### <DATASET>
- Split source:            <canonical citation / file / script that produced train/val/test>
- Split sizes:             <n_train / n_val / n_test, unique items>
- Preprocessing:           <resize, normalization, channel order>
- Frame sampling / voxel:  <video frame rate & count; event voxel-grid bins, time window, polarity handling>
- Tokenizer / vocab:       <vocab source, size, build corpus, OOV handling>      [captioning only]
- Decoding settings:       <greedy/beam, max_length, dedupe rule>                [captioning only]
- Checkpoint provenance:   <which run, seeds, commit, S3 URI, md5>
- Metric implementation:   <library + pinned version; which metrics computed>
- Leakage checks:          <how train/val/test disjointness was verified>
- Measured vs reported:    <for baselines: which numbers we measured vs cited from authors>
```

---

## COCO captioning  — **Lead-filled**

- **Split source — VERIFIED 2026-05-28 (code-confirmed, was an open audit flag):** the
  manuscript (`tinyvlm_tnnls_v1.tex` L236, L241) claims the **Karpathy split (113k/5k/5k)**.
  The code does **not** implement this. `tinyvlm_vast.py::build_datasets` (L1184–1208) builds
  the dataset from `coco_anns` (default `captions_train2017.json`) and applies a **seeded
  `random_split(full, [n_train, n_val])` with `val_fraction = 0.05`** (L129, L1205–1208).
  Consequences:
  - **No Karpathy partition.** It is a random 95/5 split of COCO **train2017**.
  - **No test set at all** — only train + a 5% "val". The claimed "5k test" does not exist
    in the training/eval path.
  - **Image-level leakage risk:** `CocoCaptionDataset` is indexed **per annotation**
    (≈5 captions/image; `len(val2017 ds) ≈ 25K` annotations for 5K images). `random_split`
    partitions *annotations*, so the **same image can appear in both train and val** with
    different captions. The 5% "val" is therefore not image-disjoint from train.
  - The separate TNNLS E19 caption-metric run (`tnnls_eval.py`) evaluated on `val2017`
    (standard COCO val, 5K images) — a *different* set from the training-loop's random 5% val
    that produced the `tab:main` Val-Acc/CIDEr numbers. Do not conflate the two.
  - **The "Karpathy split" claim in the paper is false as written and must change.**
    Decision required (Lead, before E11 reruns): **(a)** adopt a proper **image-disjoint**
    split — ideally the published Karpathy test-5k — for all headline rows (E11 reruns it),
    and re-state; or **(b)** keep the random split but make it **image-disjoint** (split on
    `image_id`, not annotation) and describe it honestly as a seeded held-out val, dropping
    any "Karpathy"/"test" language. Option (a) is strongly preferred for reviewer defensibility.
- **Split sizes (as run):** full `val2017` annotation set ≈25K annotations → **5,000 unique
  images** after dedupe (one hypothesis per `image_id`).
- **Preprocessing:** CNN backbone → ImageNet mean/std normalization, 224×224. CLIP backbone →
  `clip_stem.preprocess` (open_clip ViT-B/32 transform) instead of ImageNet norm. Mixing the
  two collapses CIDEr — confirm per-row which transform was used.
- **Frame sampling / events:** single still image; **event stream simulated as a zero tensor**
  in `CocoCaptionDataset`. This zero-event input is the documented root cause of the inference
  routing collapse (E8). State this explicitly in the paper — COCO carries no real event signal.
- **Tokenizer / vocab:** `Vocabulary` (frequency-ranked, invertible), `vocab_size = 8192`,
  built from the training caption corpus, saved as `vocabulary.json` alongside each checkpoint.
  The hash-based `simple_tokenize` is **smoke-test only** and must never appear in a reported
  COCO number (it collapses CIDEr to ~0).
- **Decoding settings:** greedy decode, `max_length = 64`, dedupe to one hypothesis per
  `image_id`, identical across all compared rows.
- **Checkpoint provenance:** STTF+ANC CNN — `tinyvlm_cider_seeds.nosync` seeds {42,43,44}
  (vocab+config present); Dense-Small — trained fresh in E7, S3
  `…/checkpoints/phase_E7_dense_small/seed_{42,43,44}/best.pt`; TokenLearner —
  `tokenlearner_baseline.nosync` seed 42 (md5 not yet verified — **verify before final table**).
- **Metric implementation:** pycocoevalcap BLEU-1..4 + ROUGE-L + CIDEr-D (single pinned
  version). METEOR/SPICE per `TNNLS_STATISTICAL_PLAN.md` §6 (JVM-gated; "n/a" if scorer did not
  complete — E19 Meteor deadlocked).
- **Leakage checks — FAILS as currently coded.** `random_split` is seeded and deterministic
  but operates on annotations, so `set(train_image_ids) & set(val_image_ids) ≠ ∅` (image-level
  leakage; see Split source above). Must be fixed to split on `image_id` before any headline
  number is trusted. Once fixed, add the disjointness assertion output here.
- **Measured vs reported:** all COCO numbers in our tables are **measured by us**. Any
  compact-VLM comparison numbers (BLIP, etc.) cited from papers must be marked
  "author-reported" in the table footnote.

---

## On-device hardware (latency / memory)  — **Lead-filled**

- **Measured platforms:** Qualcomm AI Hub — Samsung Galaxy S21 (Snapdragon 888), Snapdragon X
  Elite CRD, Snapdragon X2 Elite CRD. Jetson Nano via `jetson_nano_bench.py` (student-side).
- **Latency field:** `estimated_inference_time` (**microseconds** → convert to ms).
  **Memory field:** `estimated_inference_peak_memory` (**bytes**). NOT `peak_memory_usage`
  (old bug; `jobs.json` backfilled).
- **Aggregation:** routing-weighted with training utilization `(f0,f1,f2)=(0.31,0.34,0.35)`
  applied identically to latency, peak memory, FLOPs.
- **Measured vs estimated:** any row derived by FLOPs-ratio scaling (e.g. the current
  N-Caltech101 Table-4 row) is **estimated**, not measured, and its caption must say so until
  E10 replaces it with a real AI Hub measurement (closes C3).
- **Provenance:** canonical job log `qai_hub_results/jobs.json`; paper aggregates
  `qai_hub_results/table5_measured.json`; job IDs cited in §4.5.

---

## DVS128 Gesture (E1)  — **PhD-filled (REQUIRED before E1 integration)**

> PhD: complete every field. E1 is now the paper's primary test of input-adaptive routing on
> real event data. The single most important field here is **frame sampling / voxelization** —
> the COCO failure was zero-event input, so document exactly how real event volumes are binned
> and **verify the `complexity_estimator` receives non-constant input** (attach the Week-1
> input-sanity diagnostic: per-sample std of event volumes > 0, and a routing histogram that is
> not 100%-one-branch).

```
### DVS128 Gesture
- Split source:            TODO (PhD)  — canonical 1342-sample / 11-class split + citation
- Split sizes:             TODO
- Preprocessing:           TODO
- Frame sampling / voxel:  TODO  — voxel-grid bins, time window, polarity, accumulation; MUST show non-zero per-sample variance
- Checkpoint provenance:   TODO  — cluster run, seeds {42..46}, commit, artifact path
- Metric implementation:   TODO  — top-1 acc + macro-F1 impl
- Leakage checks:          TODO  — subject-disjoint? trial-disjoint?
- Routing sanity (E1 wk1): TODO  — std(event_volume) per sample; per-branch routing histogram
```

---

## MSR-VTT video captioning (E2)  — **PhD-filled (REQUIRED before E2 integration)**

> PhD: this is the only validation of STTF's core token-reuse claim. Report the measured
> token-reuse rate (cached/recomputed fraction) per τ ∈ {0.70, 0.80, 0.90}, plus a `protocol.json`.

```
### MSR-VTT
- Split source:            TODO (PhD)  — canonical train/val/test split + citation
- Split sizes:             TODO
- Frame sampling:          TODO  — fps, #frames/clip, resize
- Tokenizer / vocab:       TODO
- Decoding settings:       TODO  — greedy/beam, max_length, dedupe
- STTF token-reuse:        TODO  — cached/recomputed fraction per τ ∈ {0.70,0.80,0.90}
- Checkpoint provenance:   TODO
- Metric implementation:   TODO  — CIDEr/BLEU-4/METEOR/ROUGE-L, pinned pycocoevalcap version
- Leakage checks:          TODO  — clip-id disjointness
```

---

## N-Caltech101 (E4 / E16)  — **PhD-filled (REQUIRED before integration)**

```
### N-Caltech101
- Split source:            TODO (PhD)  — 8246-sample / 101-class split + citation
- Split sizes:             TODO
- Preprocessing / voxel:   TODO
- Checkpoint provenance:   TODO
- Metric implementation:   TODO  — top-1 acc + macro-F1
- Leakage checks:          TODO
- Learning-curve frac (E16): TODO  — {25,50,75,100}% train fraction protocol
```

---

## Sign-off

| Section | Filled by | Lead sign-off |
|---|---|---|
| COCO | Lead | ⚠️ (2026-05-28) — split claim is FALSE (random_split, not Karpathy) + image-level leakage; **decision + code fix required before headline numbers trusted** |
| Hardware | Lead | ✅ (2026-05-28) |
| DVS128 (E1) | PhD | ☐ pending |
| MSR-VTT (E2) | PhD | ☐ pending |
| N-Caltech101 (E4/E16) | PhD | ☐ pending |

See also: `TNNLS_STATISTICAL_PLAN.md`, `TNNLS_REVISION_PLAN.md` (E25), `FINDINGS_TNNLS.md`.
