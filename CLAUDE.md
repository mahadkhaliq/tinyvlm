# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`tinyvlm_vast.py` is a headless, checkpoint-enabled training script for **TinyVLM**, a compact Vision-Language Model that fuses RGB images with event-camera data to generate captions. It targets NeurIPS 2026 and is designed to run on local or remote (vast.ai) GPU instances.

Core ideas:
- **STTF** (Spatial-Temporal Token Fusion): temporal feature caching via adaptive threshold τ — reuse cached tokens for unchanged regions.
- **ANC** (Adaptive Neural Compression): three-branch encoder selected per-sample by a Gumbel-Softmax router. **Training** runs all branches (soft weighted sum for gradient flow); **inference** uses hard argmax routing — only the selected branch executes — realising the FLOPs savings.
- **CLIP backbone** (optional): frozen CLIP ViT-B/32 replaces the scratch CNN; three lightweight MLP heads project the 512-d CLS token to 128/256/384-d (same ANC structure).

## Architecture

### CNN backbone (default)
```
RGB [B,3,H,W] ──┐
                 ├──► complexity_estimator ──► GumbelSoftmaxRouter
Events [B,2,h,w]┘      (conv → 3-way logits)        │ weights [B,3]
                                                      │
             ┌──── TinyEncoder   (d=128, 2L, ~2 GFLOPs) ──┤
             ├──── SmallEncoder  (d=256, 4L, ~8 GFLOPs) ──┤ train: weighted sum
             └──── MediumEncoder (d=384, 6L, ~20 GFLOPs) ─┘ eval: argmax branch only
                                                      │
                                       encoded [B, 384]
                                                      │
                            ConditionalTransformer decoder (2-layer cross-attn)
                                                      │
                                 token_logits [B, T, vocab_size]
```

### CLIP backbone (`--clip_backbone`)
```
RGB [B,3,224,224] ──► CLIPStemEncoder (frozen ViT-B/32) ──► clip_feat [B, 512]
                                                              │
                           ┌── MLP head (512→128) ──┤
                           ├── MLP head (512→256) ──┤ same GumbelSoftmaxRouter
                           └── MLP head (512→384) ──┘ train soft / eval hard
                                                              │
                                               encoded [B, 384] → decoder
```

All three CLIP heads cost ~17.6 GFLOPs (dominated by ViT-B/32); routing selects representational capacity, not FLOPs.

## Key classes (with current line numbers)

| Class / function | Line | Purpose |
|---|---|---|
| `Config` | L119 | All hyperparameters as a dataclass; maps 1-to-1 with CLI flags |
| `CLIPStemEncoder` | L490 | Frozen CLIP ViT-B/32 visual stem; `preprocess` attribute used for dataset transforms |
| `AdaptiveNeuralCompression` | L570 | Main model: STTF + ANC (CNN or CLIP path) |
| `AdaptiveNeuralCompression._encoder_flops` | L645 | Returns per-branch FLOPs for comp_cost tracking |
| `TokenLearnerBaseline` | L749 | Comparison baseline; identical forward signature to ANC |
| `Trainer` | L1255 | Full loop: DDP, AMP, early stopping, checkpointing; model built *before* datasets so `clip_stem.preprocess` is available |
| `Trainer._validate` | L1644 | Reports `encoder_gflops` + `val_branch_k_hard_fraction` — the Phase 1 success gate |
| `CheckpointManager` | L1032 | Atomic writes via `.pt.tmp → os.replace`; rolling `keep_last_n` |
| `EarlyStopping` | L1100 | Patience-based; state serialised in checkpoints |
| `Vocabulary` | L296 | Frequency-ranked vocab; saves/loads `vocabulary.json` |
| `greedy_decode` | L361 | Autoregressive caption generation |
| `compute_caption_metrics` | L389 | CIDEr-D (pycocoevalcap) + BLEU-4 (nltk) |
| `build_datasets` | L980 | Accepts `clip_preprocess` kwarg; passes it to `CocoCaptionDataset` |
| `CocoCaptionDataset` | L885 | Accepts `clip_preprocess` kwarg; uses it instead of ImageNet normalization when provided |
| `run_multi_seed` | ~L1800 | W9: sequential runs over seeds, Welch t-test |
| `run_tau_sweep` | ~L1830 | W8: sequential runs over τ values |

## Reviewer-weakness mitigations

| Tag | Issue | Implementation |
|---|---|---|
| W1 | Overfitting | `dropout=0.2`, `weight_decay=0.05`, `label_smoothing=0.1`, `grad_clip=1.0`, `early_stop_patience=3` |
| W3 | ANC val gap | Seeded `random_split` (5% held-out val), per-epoch `_validate()` |
| W7 | Routing collapse | Switch-Transformer `load_balance_loss`, `UtilizationTracker` |
| W8 | τ generalization | `--tau_sweep 0.70,0.75,0.80,0.85,0.90` |
| W9 | Significance tests | `--seeds 42,43,44`, Welch t-test in `run_multi_seed` |

## Loss function

```
L = CE + λ_flops·flops_penalty(comp_cost, target_budget) + λ_entropy·RouterEntropy + λ_balance·LoadBalance
```

`flops_penalty = relu(comp_cost − target_budget).mean() / target_budget`

Defaults: `λ_flops=0.1`, `λ_entropy=0.01`, `λ_balance=0.01`, `target_budget=5e9`.
For CLIP backbone, set `--target_budget 18e9` (CLIP itself costs ~17.6 GFLOPs).

## Common commands

### Smoke test (no COCO needed)
```bash
python tinyvlm_vast.py --smoke_test --epochs 2
python tinyvlm_vast.py --smoke_test --epochs 2 --clip_backbone   # test CLIP path
```

### Full training on COCO 2017 — CNN backbone
```bash
python tinyvlm_vast.py \
    --coco_imgs /workspace/coco/train2017 \
    --coco_anns /workspace/coco/annotations/captions_train2017.json \
    --epochs 10 --batch_size 16 \
    --output_dir /workspace/runs/tinyvlm
```

### Full training — CLIP backbone (local RTX 3070, 8 GB)
```bash
python tinyvlm_vast.py \
    --coco_imgs /home/qubit/data/coco/train2017 \
    --coco_anns /home/qubit/data/coco/annotations/captions_train2017.json \
    --epochs 15 --batch_size 4 \
    --clip_backbone --lr 3e-4 --target_budget 18e9 \
    --eval_cider_freq 2 \
    --output_dir /tmp/tinyvlm_clip_seed42
```

### Multi-seed run (W9)
```bash
python tinyvlm_vast.py --seeds 42,43,44 --epochs 10
```

### τ sweep (W8)
```bash
python tinyvlm_vast.py --tau_sweep 0.70,0.75,0.80,0.85,0.90 --epochs 5
```

### Multi-GPU with torchrun
```bash
torchrun --nproc_per_node=4 tinyvlm_vast.py \
    --coco_imgs /workspace/coco/train2017 \
    --coco_anns /workspace/coco/annotations/captions_train2017.json \
    --epochs 10 --batch_size 8
```

### Resume (default) / force fresh start
```bash
python tinyvlm_vast.py --no_resume --epochs 10
```

### TokenLearner baseline
```bash
python tinyvlm_vast.py --baseline tokenlearner --smoke_test --epochs 5
```

## Key design decisions

**Hard routing at inference**: `AdaptiveNeuralCompression.forward` branches on `self.training`. In eval mode, `branch_idx = weights.argmax(dim=-1)` and only that encoder runs per sample. The paper's ~10.3 GFLOPs figure (`0.31×2 + 0.34×8 + 0.35×20`) is the hard-routing weighted average — only valid because of this code path.

**Model built before datasets**: `Trainer.__init__` builds the model first so `raw_model.clip_stem.preprocess` (the CLIP image transform) can be passed to `build_datasets` → `CocoCaptionDataset`. Changing this order breaks CLIP preprocessing.

**Vocabulary vs simple_tokenize**: The COCO path always uses `Vocabulary` (frequency-ranked, invertible). `simple_tokenize` (hash-based, non-invertible) is only for the smoke-test synthetic path. Using the wrong one collapses CIDEr to near-zero because decoded captions produce random tokens.

**comp_cost units**: Raw FLOPs (not GFLOPs). `encoder_gflops` in val metrics divides by 1e9. At eval, `comp_cost[b] = self._encoder_flops(k)` for the selected branch — confirm this equals ~17.6e9 for CLIP or ~2/8/20e9 for CNN.

## Output directory layout

```
<output_dir>/seed_42_tau_0.80/
├── config.json          # frozen config
├── vocabulary.json      # frequency-ranked word→id mapping (COCO runs only)
├── metrics.jsonl        # one JSON record per (epoch, split); val records include encoder_gflops
├── summary.json         # final aggregated results
├── checkpoints/
│   ├── best.pt          # kept forever
│   └── epoch_NNNN.pt    # rolling, last 3 only
└── plots/
    ├── curves.png
    └── anc_utilization.png
```

## Dependencies

| Package | When needed |
|---|---|
| `torch` / `torchvision` | Always |
| `open_clip_torch` | `--clip_backbone` |
| `Pillow`, `pycocotools` | COCO runs |
| `pycocoevalcap` | CIDEr metric |
| `nltk` | BLEU-4 metric |
| `matplotlib` | plots |
| `tensorboard` | optional TensorBoard logging |
| `scipy` | Welch t-test (falls back to numpy) |

Local environment: conda env `tinyvlm`, Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4, RTX 3070 8 GB.

## Configuration reference (key knobs)

| Flag | Default | Effect |
|---|---|---|
| `--clip_backbone` | False | Use frozen CLIP ViT-B/32 instead of scratch CNN |
| `--clip_model` | `ViT-B-32` | open_clip model name |
| `--clip_pretrained` | `openai` | open_clip pretrained weights |
| `--dropout` | 0.2 | Applied to all encoders and decoder |
| `--sttf_tau` | 0.80 | STTF temporal caching threshold |
| `--router_temperature` | 0.5 | Gumbel-Softmax temperature |
| `--target_budget` | 5e9 | FLOPs budget (penalised above this); use 18e9 for CLIP |
| `--eval_cider_freq` | 5 | Run CIDEr/BLEU-4 every N epochs (0 = final only) |
| `--baseline` | none | `none`=STTF+ANC, `tokenlearner`=baseline |
| `--no_onnx` | — | Skip ONNX export |

## Robustness notes

- **Atomic checkpoints**: `.pt.tmp` → `os.replace` — no partial writes survive a crash.
- **SIGINT/SIGTERM**: `GracefulExit` saves a checkpoint before exiting.
- **CUDA OOM**: OOM batches are skipped with a warning; training continues.
- **DDP**: rank 0 controls early-stopping; signal broadcast via `dist.broadcast`. Each rank loads checkpoints independently.
- **Resumption**: `--resume` (default on) picks up from `latest.pt` with RNG state restored.
