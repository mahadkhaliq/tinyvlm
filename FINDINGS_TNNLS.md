# FINDINGS_TNNLS.md — Per-phase outcomes for TNNLS Lead-side automation

Tracks results, headline numbers, and decision rationale for each phase executed via `TNNLS_AUTOMATION.md` on vast instance 37721982.

---

## 2026-05-25 — Cycle initialized

- `TNNLS_AUTOMATION.md` authored in nested tree (INSTRUCTION.md-style scaffolding).
- `TNNLS_AUTO_STATE.md` initialized; 7 phases pending (Phase 0 setup + E18, E19, E8, E9, E7, E15).
- S3 verified: `rclone v1.74.0` reachable, prefix `s3research:vastai-research/tinyvlm/tnnls/` writable.
- Instance 37721982 probed: RTX 5090 32 GB, `/workspace` 100 GB, mini image (no torch preinstalled).
- Code prerequisites verified — all required flags landed in nested `tinyvlm_vast.py` commit `6ebd6a1` (May 24): `--encoder_only`, `--eval_routing_mode`, `--baseline dense`, `DenseEncoderBaseline` class, hard-routing branch grouping. No further patches needed.

## Per-phase outcomes

(append a dated `### phase_E<id>` subsection after each phase completes with: headline numbers, decision rationale, S3 URI, GPU-hours spent, anomalies)
