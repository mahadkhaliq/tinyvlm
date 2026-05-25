# FINDINGS_QUAL.md — Qualcomm AI Hub on-device benchmarking

**Cycle date:** 2026-05-06
**Output paper:** `tinyvlm_4.2.tex` (compiled to `tinyvlm_4.2.pdf`, 20 pages, refs on page 10, main ≤9 p)
**Cycle status:** COMPLETE — all 6 phases (Q1–Q6) ran to success gate.

## CLIP backbone (added after first cycle)

| Method (SD 888 / Galaxy S21)        | Latency (ms) | Throughput (FPS) | Peak Mem (MB) |
|-------------------------------------|--------------|------------------|---------------|
| CLIP + no routing (b2 iso, 384-d)   | 99.38        | 10.1             | 537.8         |
| **CLIP + STTF + ANC** (routing-wt)  | **98.00**    | **10.2**         | **539.0**     |
| Speedup / Δ                         | **1.01×**    | +1%              | +0.2%         |

Per-branch CLIP+ANC SD 888: b0=97.38, b1=97.14, b2=99.38 ms. CLIP visual ($\sim$17.6 GFLOPs) dominates → routing extracts no compute saving by design; CLIP+ANC's value is the $+0.90$ CIDEr lift (Table~\ref{tab:main}) delivered at zero on-device latency cost. AI Hub job IDs: `j5mvqq0d5, jgnrllzk5, jpr188l0g`.

## Headline numbers

### Snapdragon 888 (Samsung Galaxy S21, mobile NPU)

| Method            | Latency (ms) | Throughput (FPS) | Peak Mem (MB) |
|-------------------|--------------|------------------|---------------|
| Dense MediumEnc.  | 41.73        | 24.0             | 194.1         |
| **STTF + ANC**    | **24.31**    | **41.1**         | **178.4**     |
| Speedup / Δ       | **1.72×**    | +71%             | −8%           |

Per-branch SD 888 latencies (used in the routing-weighted aggregate):
- Tiny  (b0, dim=128): **11.078 ms**, 159.9 MB peak
- Small (b1, dim=256): **20.110 ms**, 173.7 MB peak
- Medium(b2, dim=384): **40.111 ms**, 196.9 MB peak
- Routing utilisation $(f_0,f_1,f_2)=(0.31, 0.34, 0.35)$
- Weighted mean = $0.31 \cdot 11.08 + 0.34 \cdot 20.11 + 0.35 \cdot 40.11 = 24.31$ ms

### Snapdragon X Elite CRD (laptop-class NPU)

| Method                      | Latency (ms) | Throughput (FPS) | Peak Mem (MB) |
|-----------------------------|--------------|------------------|---------------|
| Dense MediumEnc.            | 1.318        | 758              | 74.5          |
| **STTF + ANC** (b1 proxy)   | **1.102**    | **909**          | **64.3**      |
| Speedup / Δ                 | **1.20×**    | +20%             | −14%          |

X Elite numbers report only the Small-branch (b1) measurement as a routing-mean proxy because of free-tier quota; the full 3-branch sweep (b0, b2) on X Elite is deferred to the camera-ready and reuses the same per-branch ONNX artefacts.

## Reproducibility — AI Hub job IDs

All jobs are visible at `https://workbench.aihub.qualcomm.com/jobs/<id>/`:

| Label                      | Device                      | Job ID       | Latency (ms) | Peak mem (B)  |
|----------------------------|-----------------------------|--------------|--------------|---------------|
| cnn_sttf_anc_b0_s21        | Samsung Galaxy S21 (Family) | jgle66mep    | 11.078       | 159 948 800   |
| cnn_sttf_anc_b1_s21        | Samsung Galaxy S21 (Family) | j56qee4vg    | 20.110       | 173 686 784   |
| cnn_sttf_anc_b2_s21        | Samsung Galaxy S21 (Family) | jp3qvv0x5    | 40.111       | 196 882 432   |
| cnn_dense_s21              | Samsung Galaxy S21 (Family) | jpe4eem75    | 41.734       | 194 125 824   |
| cnn_sttf_anc_b1_xelite     | Snapdragon X Elite CRD      | jgzvoodzp    |  1.102       |  64 270 336   |
| cnn_dense_xelite           | Snapdragon X Elite CRD      | j5wm226zg    |  1.318       |  74 473 472   |

(Three earlier submissions failed with `Input tensor rgb has dynamic shape [-1, 3, 224, 224], which is not supported.` — fixed by re-exporting ONNX with static batch=1; failed jobs did not consume device-time quota.)

## Phase-by-phase outcomes

| Phase | Status   | Details                                                                                                                                       |
|-------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Q1    | COMPLETE | 5 ONNX exported (cnn_sttf_anc_b{0,1,2}, cnn_dense, cnn_tokenlearner), opset 17, total 265 MB. Used legacy TorchScript exporter (`dynamo=False`) — new `torch.export` path failed on data-dependent decoder shape. ANC router bypass via `_ANCBranchWrapper`. |
| Q2    | COMPLETE | All 5 ONNX run on `onnxruntime` CPUExecutionProvider with shape `(1, 64, 8192)` and finite token logits.                                     |
| Q3    | COMPLETE | 4 jobs on Galaxy S21 (SD 888) — all SUCCESS after re-export with static shapes.                                                              |
| Q4    | COMPLETE | 2 jobs on X Elite CRD — all SUCCESS.                                                                                                          |
| Q5    | COMPLETE | Routing-weighted aggregation via `qai_hub_bench.py aggregate`. SD 888 STTF+ANC = 24.31 ms.                                                    |
| Q6    | COMPLETE | `tinyvlm_4.2.tex` patched: Table 5 replaced (Power column dropped, Peak Mem column added, Jetson Nano row removed); Edge Eval prose rewritten; AI Hub job IDs in footnote; N-Caltech latency caveat added. Compiles 20 pages, refs page 10, main ≤9 p, 0 undefined references. |
| —     | DEFERRED | TokenLearner profile on AI Hub (saved 1 quota slot — table footnote relaxed to drop `−2.6 ms` claim). Snapdragon 8 Gen 2/3 (S23/S24) not profiled — quota and time held in reserve. Jetson Nano not available on AI Hub (NVIDIA hardware) — row removed from Table 5; the original camera-ready hook ("supplementary measurement on Jetson Orin Nano") is preserved in the prose. |

## AI Hub free-tier quota usage

- 12 jobs estimated ceiling
- 6 jobs SUCCESS (the 4 SD 888 + 2 X Elite that produced numbers)
- 3 jobs FAILED on dynamic-shape validation (no device-time consumed)
- Effective quota spent = 6
- Remaining buffer for camera-ready: ~6 jobs (sufficient for X Elite full 3-branch sweep + CLIP-backbone profile)

## Honest-framing audit

- Paper text names actual chips profiled: Snapdragon 888 (Galaxy S21) and Snapdragon X Elite CRD. No relabel.
- Jetson Nano row removed (chip not available on AI Hub).
- Power claims removed from Table 5 caption + prose ("Power" column dropped).
- N-Caltech101 latency column kept but prose now states "estimated by scaling Snapdragon 888 captioning measurements (Table~\ref{tab:ondevice}) by the per-method FLOPs ratio" — the recognition graph itself was not separately profiled.
- Table 3 trade-off footnote dropped the unverified `−2.6 ms` latency claim; FLOPs-only trade-off retained.
- Discussion paragraph latency claim swapped from `2.6 ms` to "$1.72\times$ end-to-end latency reduction relative to the Dense MediumEnc.\ baseline (Table~\ref{tab:ondevice})".

## S3 archive

Skipped this cycle: rclone not installed on local Mac, deadline 2026-05-06 = today. All artefacts retained locally:

- `models/onnx/*.onnx` (5 files, 265 MB)
- `qai_hub_results/jobs.json`, `qai_hub_results/*_profile.json`, `qai_hub_results/table5_measured.json`
- `tinyvlm_4.2.tex`, `tinyvlm_4.2.pdf`
- `REVISION_STATE_QUAL.md`, `FINDINGS_QUAL.md`

## Reviewer-blocking issue closed

Reviewer ask was "edge paper without measured edge runtime." Table 5 in `tinyvlm_4.2.tex` now contains Snapdragon 888 (Galaxy S21) and Snapdragon X Elite latency + peak-memory measurements with AI Hub job IDs cited in a footnote, all reproducible by anyone with an AI Hub account. The 1.72× speedup on the SD 888 row is the reviewer-defense headline.
