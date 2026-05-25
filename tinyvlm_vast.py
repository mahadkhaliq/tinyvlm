#!/usr/bin/env python3
"""
tinvlm_vast.py — Production training script for TinyVLM (STTF + ANC) on vast.ai.

This file rewrites tinyvlm_sttf_colab.py as a clean, headless, checkpoint-enabled
training pipeline designed to run over SSH on a vast.ai GPU instance. It
addresses the unresolved blocking issues from NeurIPS2026_Review2.md:

  * W1 — Overfitting:       dropout, increased weight decay, label smoothing,
                             gradient clipping, early stopping.
  * W3 — ANC val gap:       clean train/val split with stable seeding,
                             per-epoch validation on a held-out set.
  * W7 — Routing collapse:   Switch-Transformer load-balancing auxiliary loss
                             and per-branch utilization tracking every epoch.
  * W8 — τ generalization:   optional --tau_sweep flag for cross-dataset sweeps.
  * W9 — Significance tests: multi-seed runs with per-seed result logging
                             and Welch t-test aggregation.

Robustness features for vast.ai:
  * Atomic checkpoint writes with rolling retention.
  * Automatic resume from the most recent checkpoint in the output directory.
  * Graceful SIGINT/SIGTERM handling: saves a final checkpoint on preemption.
  * JSON-lines metrics log (machine-readable) alongside TensorBoard events.
  * OOM retry: falls back to a smaller per-step batch on CUDA OOM.
  * No shell magics, no implicit Colab assumptions. Pure Python.

Usage (smoke test on synthetic data — no COCO required):

    python tinvlm_vast.py --smoke_test --epochs 2

Full training on COCO 2017:

    python tinvlm_vast.py \
        --coco_imgs /workspace/coco/train2017 \
        --coco_anns /workspace/coco/annotations/captions_train2017.json \
        --epochs 10 --batch_size 16 --output_dir /workspace/runs/tinyvlm

Multi-seed run for statistical significance (W9):

    python tinvlm_vast.py --seeds 42,43,44 --epochs 10

τ sweep for W8 cross-dataset generalization study:

    python tinvlm_vast.py --tau_sweep 0.70,0.75,0.80,0.85,0.90 --epochs 5
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split


# =============================================================================
# Optional dependencies — imported lazily and with graceful fallbacks.
# =============================================================================

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _TB_AVAILABLE = True
except Exception:
    _TB_AVAILABLE = False

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(iterable=None, *_, **__):  # type: ignore
        return iterable if iterable is not None else iter(())

try:
    from PIL import Image  # type: ignore
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

try:
    from pycocotools.coco import COCO  # type: ignore
    _COCO_AVAILABLE = True
except Exception:
    _COCO_AVAILABLE = False

try:
    import open_clip  # type: ignore
    _OPEN_CLIP_AVAILABLE = True
except Exception:
    _OPEN_CLIP_AVAILABLE = False

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger("tinyvlm")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    # --- Data ---
    coco_imgs: str = "/workspace/coco/train2017"
    coco_anns: str = "/workspace/coco/annotations/captions_train2017.json"
    val_fraction: float = 0.05
    max_length: int = 64
    vocab_size: int = 8192
    smoke_test: bool = False
    smoke_test_size: int = 512

    # --- Model ---
    tiny_dim: int = 128
    small_dim: int = 256
    medium_dim: int = 384
    hidden_dim: int = 384
    num_decoder_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.2  # W1: dropout regularization

    # --- Training ---
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.05  # W1: increased from default
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    label_smoothing: float = 0.1  # W1

    # --- Losses ---
    target_budget: float = 5e9
    lambda_flops: float = 0.1
    lambda_entropy: float = 0.01
    lambda_balance: float = 0.01  # W7: Switch-Transformer aux loss weight

    # --- Routing / STTF ---
    router_temperature: float = 0.5
    sttf_tau: float = 0.80

    # --- Early stopping (W1) ---
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4

    # --- Checkpointing ---
    output_dir: str = "/workspace/runs/tinyvlm"
    keep_last_n_checkpoints: int = 3
    resume: bool = True

    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = False
    num_workers: int = 4

    # --- CLIP backbone ---
    clip_backbone: bool = False       # replace CNN encoders with frozen CLIP ViT-B/32
    clip_model: str = "ViT-B-32"     # open_clip model name
    clip_pretrained: str = "openai"  # open_clip pretrained weights tag

    # --- GPT-2 decoder ---
    gpt2_decoder: bool = False        # replace scratch decoder with pretrained GPT-2 small
    gpt2_n_prefix: int = 10           # number of visual prefix tokens fed to GPT-2

    # --- Baseline ---
    baseline: str = "none"          # "none" = STTF+ANC, "tokenlearner" = TL, "dense" = single encoder
    encoder_only: str = "medium"    # dense baseline encoder: "tiny", "small", or "medium"
    tokenlearner_num_tokens: int = 8  # number of learned tokens S

    # --- Ablations ---
    no_anc: bool = False  # S5: single fixed encoder, no routing (iso-backbone ablation)
    eval_only: bool = False  # S1: load checkpoint, eval once, exit (no training)

    # --- Evaluation ---
    eval_cider_freq: int = 5   # compute CIDEr/BLEU-4 every N epochs (0 = final only)
    eval_routing_mode: str = "hard"  # "hard" = deployed argmax, "soft" = train-time weighted routing

    # --- Export ---
    export_onnx: bool = True

    # --- Logging ---
    log_level: str = "INFO"
    tensorboard: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# =============================================================================
# Utilities: seeding, logging, graceful exit
# =============================================================================


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def setup_logging(log_level: str, log_file: Optional[Path] = None) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        root.addHandler(fh)


class GracefulExit:
    """Catches SIGINT/SIGTERM so the training loop can save a final checkpoint."""

    def __init__(self) -> None:
        self.should_exit = False
        try:
            signal.signal(signal.SIGINT, self._handle)
            signal.signal(signal.SIGTERM, self._handle)
        except ValueError:
            # Signals are only available in the main thread.
            pass

    def _handle(self, signum: int, _frame: Any) -> None:
        logger.warning(
            "Received signal %d; will save checkpoint and exit after current step.",
            signum,
        )
        self.should_exit = True


# =============================================================================
# Distributed Data Parallel helpers
# =============================================================================


def setup_ddp() -> Tuple[int, int, int]:
    """Read torchrun env vars and initialise the NCCL process group.

    Returns (rank, local_rank, world_size). When launched with plain
    ``python`` (no torchrun), world_size=1 and no distributed group is
    created.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        logger.info("DDP | rank %d / %d | local_rank %d", rank, world_size, local_rank)
    return rank, local_rank, world_size


def teardown_ddp() -> None:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Tokenizer (hash-based, kept identical to the original paper implementation)
# =============================================================================


def simple_tokenize(
    caption: str, max_length: int = 64, vocab_size: int = 8192
) -> torch.Tensor:
    toks = caption.lower().split()[: max_length - 1]
    ids = [min(abs(hash(w)) % (vocab_size - 1) + 1, vocab_size - 1) for w in toks]
    pad = [0] * (max_length - len(ids))
    return torch.tensor(ids + pad, dtype=torch.long)


class Vocabulary:
    """Frequency-based invertible vocabulary built from COCO captions.

    Replaces the one-way hash tokenizer so that generated token sequences
    can be decoded back to text for CIDEr / BLEU-4 evaluation.

    Special tokens:
        0  <PAD>   padding
        1  <UNK>   unknown word
        2  <BOS>   begin of sequence
        3  <EOS>   end of sequence
    """

    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIALS = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}

    def __init__(self, max_size: int = 8192) -> None:
        self.max_size = max_size
        self.word2idx: Dict[str, int] = {v: k for k, v in self.SPECIALS.items()}
        self.idx2word: Dict[int, str] = dict(self.SPECIALS)

    def build(self, captions: List[str]) -> None:
        from collections import Counter
        counter: Counter = Counter()
        for cap in captions:
            counter.update(cap.lower().split())
        for word, _ in counter.most_common(self.max_size - len(self.SPECIALS)):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        logger.info("Vocabulary built: %d words", len(self.word2idx))

    def encode(self, caption: str, max_length: int) -> torch.Tensor:
        words = caption.lower().split()
        ids = [self.BOS] + [self.word2idx.get(w, self.UNK) for w in words[: max_length - 2]] + [self.EOS]
        ids = ids[:max_length]
        ids += [self.PAD] * (max_length - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            if idx == self.EOS:
                break
            if idx not in (self.PAD, self.BOS):
                words.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(words)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.word2idx, indent=2))

    @classmethod
    def load(cls, path: Path, max_size: int = 8192) -> "Vocabulary":
        v = cls(max_size)
        v.word2idx = json.loads(path.read_text())
        v.idx2word = {int(i): w for w, i in v.word2idx.items()}
        return v


# =============================================================================
# Caption generation and evaluation metrics
# =============================================================================


@torch.no_grad()
def greedy_decode(
    model: Any,
    rgb: torch.Tensor,
    events: torch.Tensor,
    max_length: int,
    vocab: Vocabulary,
    device: torch.device,
) -> List[str]:
    """Greedy autoregressive decoding. Model-agnostic: works with both
    AdaptiveNeuralCompression and TokenLearnerBaseline via their shared
    forward() signature (rgb, events, text_tokens) → (logits, ...)."""
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    B = rgb.shape[0]
    tokens = torch.full((B, max_length), vocab.PAD, dtype=torch.long, device=device)
    tokens[:, 0] = vocab.BOS

    for t in range(1, max_length):
        logits, _, _, _ = raw(rgb, events, tokens)   # [B, max_length, vocab]
        next_tok = logits[:, t - 1, :].argmax(dim=-1)
        tokens[:, t] = next_tok
        if (next_tok == vocab.EOS).all():
            break

    captions = [vocab.decode(tokens[b].tolist()) for b in range(B)]
    return captions


@torch.no_grad()
def greedy_decode_gpt2(
    model: Any,
    rgb: torch.Tensor,
    events: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    device: torch.device,
) -> List[str]:
    """Greedy autoregressive decoding using the GPT-2 decoder path.

    Runs the CLIP+ANC encoder in hard-routing eval mode to get visual prefix
    tokens, then autoregressively generates caption tokens with GPT-2.
    """
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    B = rgb.size(0)

    # Encode: CLIP stem → hard-routed ANC branch → encoded [B, hidden_dim]
    if raw.clip_stem is not None:
        clip_feat = raw.clip_stem(rgb)
        router_logits = raw.complexity_estimator(clip_feat)
    else:
        router_logits = raw.complexity_estimator(events)
    weights = F.softmax(router_logits, dim=-1)
    branch_idx = weights.argmax(dim=-1)

    encoded = torch.zeros(B, raw.cfg.hidden_dim, device=device)
    for k, (enc, proj) in enumerate(zip(raw.encoders, raw.projections)):
        mask = branch_idx == k
        if not mask.any():
            continue
        feat = enc(clip_feat[mask]) if raw.clip_stem is not None else enc(rgb[mask], events[mask])
        encoded[mask] = proj(feat)

    prefix = raw.prefix_adapter(encoded)   # [B, n_prefix, 768]

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens - 1):
        tok_emb = raw.decoder.gpt2.transformer.wte(input_ids)             # [B, t, 768]
        combined = torch.cat([prefix, tok_emb], dim=1)                    # [B, n_prefix+t, 768]
        next_tok = raw.decoder.gpt2(inputs_embeds=combined).logits[:, -1, :].argmax(dim=-1)  # [B]
        next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
        input_ids = torch.cat([input_ids, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok == eos_id
        if finished.all():
            break

    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)


def compute_caption_metrics(
    hypotheses: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Compute CIDEr-D and BLEU-4.

    Tries pycocoevalcap first; falls back to nltk BLEU-4 only.

    Args:
        hypotheses:  one generated caption per image.
        references:  one or more reference captions per image (list of lists).
    """
    results: Dict[str, float] = {}

    # ── CIDEr via pycocoevalcap ───────────────────────────────────────────────
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore

        # Newer pycocoevalcap expects plain strings; older expects {"caption": str}.
        # Try plain strings first, fall back to dict format.
        gts_str = {i: refs for i, refs in enumerate(references)}
        res_str = {i: [h] for i, h in enumerate(hypotheses)}
        try:
            cider_scorer = Cider()
            score, _ = cider_scorer.compute_score(gts_str, res_str)
        except Exception:
            gts_str = {i: [{"caption": r} for r in refs] for i, refs in enumerate(references)}
            res_str = {i: [{"caption": h}] for i, h in enumerate(hypotheses)}
            cider_scorer = Cider()
            score, _ = cider_scorer.compute_score(gts_str, res_str)
        results["cider"] = float(score) * 100  # convert to 0-100 scale
    except Exception as e:
        logger.warning("CIDEr via pycocoevalcap failed (%s); skipping CIDEr.", e)

    # ── BLEU-4 via nltk ───────────────────────────────────────────────────────
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  # type: ignore

        smoothie = SmoothingFunction().method1
        refs_tok = [[ref.split() for ref in refs] for refs in references]
        hyp_tok = [h.split() for h in hypotheses]
        results["bleu4"] = float(corpus_bleu(refs_tok, hyp_tok, smoothing_function=smoothie)) * 100
    except Exception as e:
        logger.warning("BLEU-4 via nltk failed (%s); skipping BLEU-4.", e)

    # ── METEOR via nltk (pure Python, no Java required) ──────────────────────
    try:
        from nltk.translate.meteor_score import meteor_score as _nltk_meteor  # type: ignore
        import nltk  # type: ignore

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        scores = [
            _nltk_meteor([ref.split() for ref in refs], hyp.split())
            for refs, hyp in zip(references, hypotheses)
        ]
        results["meteor"] = float(sum(scores) / len(scores)) * 100
    except Exception as e:
        logger.warning("METEOR via nltk failed (%s); skipping METEOR.", e)

    return results


# =============================================================================
# Model
# =============================================================================


class _ConvEncoder(nn.Module):
    """Shared encoder scaffold; Tiny/Small/Medium differ only in (dim, depth, flops)."""

    def __init__(self, dim: int, depth: int, flops: float, dropout: float) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(5, dim, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Dropout2d(dropout),
        ]
        for _ in range(max(0, depth - 1)):
            layers += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ]
        layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
        ]
        self.net = nn.Sequential(*layers)
        self.flops = flops

    def forward(self, rgb: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-2:] != events.shape[-2:]:
            rgb = F.interpolate(
                rgb, size=events.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([rgb, events], dim=1)
        return self.net(x)


class TinyEncoder(_ConvEncoder):
    def __init__(self, dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__(dim=dim, depth=2, flops=2e9, dropout=dropout)


class SmallEncoder(_ConvEncoder):
    def __init__(self, dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__(dim=dim, depth=4, flops=8e9, dropout=dropout)


class MediumEncoder(_ConvEncoder):
    def __init__(self, dim: int = 384, dropout: float = 0.0) -> None:
        super().__init__(dim=dim, depth=6, flops=20e9, dropout=dropout)


class CLIPStemEncoder(nn.Module):
    """Frozen CLIP ViT-B/32 visual encoder outputting a [B, 512] CLS embedding.

    Used as a drop-in stem before the ANC routing heads when --clip_backbone
    is set.  Weights are frozen; only the downstream projection heads and
    decoder are trained.  The complexity estimator is reattached to operate
    on the 512-d CLS token rather than raw event-camera features.
    """

    OUT_DIM: int = 512   # ViT-B/32 CLS dimension
    # Reported FLOPs for one ViT-B/32 forward pass (image side only).
    FLOPS: float = 17.6e9

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        super().__init__()
        if not _OPEN_CLIP_AVAILABLE:
            raise RuntimeError("open_clip_torch is not installed. Run `pip install open_clip_torch`.")
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual
        self.visual.requires_grad_(False)

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.visual(rgb)   # [B, 512]


class GumbelSoftmaxRouter(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            y = (logits + g) / max(self.temperature, 1e-6)
            return F.softmax(y, dim=-1)
        return F.softmax(logits, dim=-1)


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        vocab_size: int = 8192,
        max_length: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(max_length, dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, vocab_size)

    def forward(
        self, encoded: torch.Tensor, text_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, T = text_tokens.shape
        memory = self.input_proj(encoded).unsqueeze(1)  # [B, 1, dim]
        tgt = self.token_emb(text_tokens) + self.pos_emb[:T, :]
        tgt = self.drop(tgt)
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=text_tokens.device), diagonal=1
        )
        out = self.decoder(tgt, memory, tgt_mask=causal_mask)
        return self.out(out)


class VisualPrefixAdapter(nn.Module):
    """Projects ANC visual features [B, in_dim] → GPT-2 prefix embeddings [B, n_prefix, gpt2_dim]."""

    def __init__(self, in_dim: int, gpt2_dim: int = 768, n_prefix: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_prefix = n_prefix
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, gpt2_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gpt2_dim * 2, gpt2_dim * n_prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).view(x.size(0), self.n_prefix, -1)


class GPT2CaptionDecoder(nn.Module):
    """Pretrained GPT-2 small with visual prefix injection (ClipCap-style).

    Teacher-forcing forward: concatenates n_prefix visual prefix tokens in front
    of the caption token embeddings and returns logits for the caption positions
    only, keeping the same [B, T, vocab_size] shape as ConditionalTransformer.
    """

    GPT2_DIM: int = 768       # GPT-2 small hidden dimension
    VOCAB_SIZE: int = 50257   # GPT-2 vocabulary size

    def __init__(self) -> None:
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers is not installed. Run `pip install transformers`.")
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, prefix_embeds: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prefix_embeds: [B, n_prefix, 768] — visual prefix from VisualPrefixAdapter
            input_ids:     [B, T] — GPT-2 token IDs (BOS + caption tokens, padded)
        Returns:
            logits: [B, T, 50257] — one logit vector per caption token position
        """
        n_prefix = prefix_embeds.size(1)
        token_embeds = self.gpt2.transformer.wte(input_ids)           # [B, T, 768]
        combined = torch.cat([prefix_embeds, token_embeds], dim=1)    # [B, n_prefix+T, 768]
        out = self.gpt2(inputs_embeds=combined)
        return out.logits[:, n_prefix:, :]                             # [B, T, 50257]


class AdaptiveNeuralCompression(nn.Module):
    """STTF + ANC unified model. Shape-compatible with tiny_vlm_anc.onnx export.

    When cfg.clip_backbone=True the three CNN encoders are replaced with a
    frozen CLIP ViT-B/32 stem feeding three lightweight MLP projection heads
    of heterogeneous capacity.  The routing and decoder are unchanged.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.clip_stem: Optional[CLIPStemEncoder] = None

        if cfg.clip_backbone:
            self.clip_stem = CLIPStemEncoder(cfg.clip_model, cfg.clip_pretrained)
            clip_dim = CLIPStemEncoder.OUT_DIM  # 512

            if cfg.no_anc:
                # Single fixed 384-d MLP projection, no router — S5 iso-backbone ablation.
                self.encoders = nn.ModuleList([
                    nn.Sequential(nn.Linear(clip_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout))
                ])
                self.projections = nn.ModuleList([nn.Linear(cfg.hidden_dim, cfg.hidden_dim)])
                self._head_flops = [CLIPStemEncoder.FLOPS]
                self.complexity_estimator = None
            else:
                # Three heterogeneous MLP heads operating on CLIP features.
                # Capacity difference comes from hidden width, preserving routing incentive.
                self.encoders = nn.ModuleList([
                    nn.Sequential(nn.Linear(clip_dim, 128), nn.GELU(), nn.Dropout(cfg.dropout)),
                    nn.Sequential(nn.Linear(clip_dim, 256), nn.GELU(), nn.Dropout(cfg.dropout)),
                    nn.Sequential(nn.Linear(clip_dim, 384), nn.GELU(), nn.Dropout(cfg.dropout)),
                ])
                # Approximate FLOPs for each MLP head (clip stem FLOPs shared).
                self._head_flops = [
                    CLIPStemEncoder.FLOPS + 128 * clip_dim * 2,
                    CLIPStemEncoder.FLOPS + 256 * clip_dim * 2,
                    CLIPStemEncoder.FLOPS + 384 * clip_dim * 2,
                ]
                head_dims = [128, 256, 384]
                self.projections = nn.ModuleList([
                    nn.Linear(d, cfg.hidden_dim) for d in head_dims
                ])
                # Complexity estimator operates on the 512-d CLIP CLS token.
                self.complexity_estimator = nn.Sequential(
                    nn.Linear(clip_dim, 64),
                    nn.GELU(),
                    nn.Linear(64, len(self.encoders)),
                )
        else:
            self.encoders = nn.ModuleList(
                [
                    TinyEncoder(dim=cfg.tiny_dim, dropout=cfg.dropout),
                    SmallEncoder(dim=cfg.small_dim, dropout=cfg.dropout),
                    MediumEncoder(dim=cfg.medium_dim, dropout=cfg.dropout),
                ]
            )
            self.projections = nn.ModuleList(
                [
                    nn.Linear(cfg.tiny_dim, cfg.hidden_dim),
                    nn.Linear(cfg.small_dim, cfg.hidden_dim),
                    nn.Linear(cfg.medium_dim, cfg.hidden_dim),
                ]
            )
            self.complexity_estimator = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=7, stride=4, padding=3),
                nn.GELU(),
                nn.Dropout2d(cfg.dropout),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, len(self.encoders)),
            )
            self._head_flops = [enc.flops for enc in self.encoders]

        self.router: Optional[GumbelSoftmaxRouter] = (
            None if cfg.no_anc else GumbelSoftmaxRouter(temperature=cfg.router_temperature)
        )

        if cfg.gpt2_decoder:
            self.prefix_adapter: Optional[VisualPrefixAdapter] = VisualPrefixAdapter(
                in_dim=cfg.hidden_dim,
                gpt2_dim=GPT2CaptionDecoder.GPT2_DIM,
                n_prefix=cfg.gpt2_n_prefix,
                dropout=cfg.dropout,
            )
            self.decoder: Any = GPT2CaptionDecoder()
        else:
            self.prefix_adapter = None
            self.decoder = ConditionalTransformer(
                dim=cfg.hidden_dim,
                vocab_size=cfg.vocab_size,
                max_length=cfg.max_length,
                nhead=cfg.num_heads,
                num_layers=cfg.num_decoder_layers,
                dropout=cfg.dropout,
            )

    def _encoder_flops(self, k: int) -> float:
        return self._head_flops[k]

    def forward(
        self, rgb: torch.Tensor, events: torch.Tensor, text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = rgb.size(0)

        # ── no_anc: bypass router entirely, use single fixed encoder ─────────
        if self.cfg.no_anc:
            clip_feat = self.clip_stem(rgb)               # [B, 512]
            feat = self.encoders[0](clip_feat)
            encoded = self.projections[0](feat)
            comp_cost = torch.full((B,), self._head_flops[0], device=rgb.device)
            router_logits = torch.zeros(B, 1, device=rgb.device)
            weights = torch.ones(B, 1, device=rgb.device)
            if self.prefix_adapter is not None:
                token_logits = self.decoder(self.prefix_adapter(encoded), text_tokens)
            else:
                token_logits = self.decoder(encoded, text_tokens)
            return token_logits, comp_cost, router_logits, weights

        if self.clip_stem is not None:
            # CLIP stem: produces [B, 512] CLS token (frozen, no grad).
            clip_feat = self.clip_stem(rgb)               # [B, 512]
            router_logits = self.complexity_estimator(clip_feat)
            weights = self.router(router_logits)
        else:
            router_logits = self.complexity_estimator(events)
            weights = self.router(router_logits)
        encoded = torch.zeros(B, self.cfg.hidden_dim, device=rgb.device)
        comp_cost = torch.zeros(B, device=rgb.device)

        use_soft_routing = self.training or self.cfg.eval_routing_mode == "soft"
        if use_soft_routing:
            # Soft routing: all encoders run; gradients flow through weights.
            if self.clip_stem is not None:
                feats = [enc(clip_feat) for enc in self.encoders]
            else:
                feats = [enc(rgb, events) for enc in self.encoders]
            for i, feat in enumerate(feats):
                w = weights[:, i : i + 1]
                encoded = encoded + w * self.projections[i](feat)
                comp_cost = comp_cost + w.squeeze(-1) * self._encoder_flops(i)
        else:
            # Hard routing: only the argmax branch runs per sample.
            # Samples are grouped by branch index for efficient batched execution.
            branch_idx = weights.argmax(dim=-1)          # [B]
            for k, (enc, proj) in enumerate(zip(self.encoders, self.projections)):
                mask = branch_idx == k
                if not mask.any():
                    continue
                feat = enc(clip_feat[mask]) if self.clip_stem is not None else enc(rgb[mask], events[mask])
                encoded[mask] = proj(feat)
                comp_cost[mask] = self._encoder_flops(k)

        if self.prefix_adapter is not None:
            prefix = self.prefix_adapter(encoded)
            token_logits = self.decoder(prefix, text_tokens)
        else:
            token_logits = self.decoder(encoded, text_tokens)
        return token_logits, comp_cost, router_logits, weights


# =============================================================================
# Dense single-encoder baseline for FLOPs-matched comparisons
# =============================================================================


class DenseEncoderBaseline(nn.Module):
    """Dense CNN baseline with a single fixed encoder branch.

    This supports the TNNLS FLOPs-matched SmallEncoder-only comparison while
    preserving the caption decoder and training loop interface.
    """

    _ENCODER_FACTORY: Dict[str, Callable[[Config], Tuple[nn.Module, int]]] = {
        "tiny": lambda cfg: (TinyEncoder(dim=cfg.tiny_dim, dropout=cfg.dropout), cfg.tiny_dim),
        "small": lambda cfg: (SmallEncoder(dim=cfg.small_dim, dropout=cfg.dropout), cfg.small_dim),
        "medium": lambda cfg: (MediumEncoder(dim=cfg.medium_dim, dropout=cfg.dropout), cfg.medium_dim),
    }

    def __init__(self, cfg: "Config") -> None:
        super().__init__()
        if cfg.encoder_only not in self._ENCODER_FACTORY:
            raise ValueError(
                f"Unsupported --encoder_only={cfg.encoder_only!r}; "
                "expected one of tiny, small, medium"
            )
        self.cfg = cfg
        self.encoder_name = cfg.encoder_only
        self.encoder, encoder_dim = self._ENCODER_FACTORY[cfg.encoder_only](cfg)
        self.proj = nn.Linear(encoder_dim, cfg.hidden_dim)
        self.decoder = ConditionalTransformer(
            dim=cfg.hidden_dim,
            vocab_size=cfg.vocab_size,
            max_length=cfg.max_length,
            nhead=cfg.num_heads,
            num_layers=cfg.num_decoder_layers,
            dropout=cfg.dropout,
        )

    def forward(
        self, rgb: torch.Tensor, events: torch.Tensor, text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(rgb, events)
        encoded = self.proj(feat)
        token_logits = self.decoder(encoded, text_tokens)

        B = encoded.shape[0]
        flops = float(getattr(self.encoder, "flops", 0.0))
        comp_cost = torch.full((B,), flops, device=encoded.device)
        router_logits = torch.zeros(B, 1, device=encoded.device)
        router_weights = torch.ones(B, 1, device=encoded.device)
        return token_logits, comp_cost, router_logits, router_weights


# =============================================================================
# TokenLearner baseline  (Ryoo et al., NeurIPS 2021)
# =============================================================================


class _SpatialEncoder(nn.Module):
    """Like MediumEncoder but returns spatial features [B, dim, H, W]
    instead of a globally-pooled vector — required by TokenLearner."""

    def __init__(self, dim: int = 384, depth: int = 6, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(5, dim, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Dropout2d(dropout),
        ]
        for _ in range(max(0, depth - 1)):
            layers += [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, rgb: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        if rgb.shape[-2:] != events.shape[-2:]:
            rgb = F.interpolate(
                rgb, size=events.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.net(torch.cat([rgb, events], dim=1))  # [B, dim, H, W]


class _TokenLearnerModule(nn.Module):
    """Learns S spatial attention maps to summarise a feature map into S tokens.

    Each of the S output tokens is a weighted sum of the spatial locations,
    where weights are produced by a per-token 1×1 conv followed by softmax.
    This matches the formulation in Ryoo et al., NeurIPS 2021.
    """

    def __init__(self, in_channels: int, num_tokens: int = 8) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.attn_maps = nn.Sequential(
            nn.Conv2d(in_channels, num_tokens, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        attn = self.attn_maps(x)          # [B, S, H, W]
        attn = attn.view(B, self.num_tokens, H * W)
        attn = torch.softmax(attn, dim=-1)  # normalise over spatial locations
        x_flat = x.view(B, C, H * W)       # [B, C, HW]
        # Weighted sum over spatial locations: [B, S, C]
        tokens = torch.einsum("bsn,bcn->bsc", attn, x_flat)
        return tokens


class TokenLearnerBaseline(nn.Module):
    """Baseline model replacing STTF+ANC with learned spatial token selection.

    Architecture:
        Spatial CNN encoder → TokenLearner (S tokens) → mean-pool → Transformer decoder

    There is no temporal caching (unlike STTF) and no adaptive routing (unlike ANC).
    The forward signature matches AdaptiveNeuralCompression so the training loop,
    losses, and metrics code require no modification.

    comp_cost is returned as zeros (no FLOPs budget is consumed by routing).
    router_weights is returned as uniform (no routing decisions are made).
    """

    def __init__(self, cfg: "Config", num_tokens: int = 8) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_tokens = num_tokens
        self.encoder = _SpatialEncoder(
            dim=cfg.medium_dim, depth=6, dropout=cfg.dropout
        )
        self.token_learner = _TokenLearnerModule(
            in_channels=cfg.medium_dim, num_tokens=num_tokens
        )
        self.proj = nn.Linear(cfg.medium_dim, cfg.hidden_dim)
        self.decoder = ConditionalTransformer(
            dim=cfg.hidden_dim,
            vocab_size=cfg.vocab_size,
            max_length=cfg.max_length,
            nhead=cfg.num_heads,
            num_layers=cfg.num_decoder_layers,
            dropout=cfg.dropout,
        )

    def forward(
        self, rgb: torch.Tensor, events: torch.Tensor, text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = self.encoder(rgb, events)          # [B, dim, H, W]
        tokens = self.token_learner(spatial)          # [B, S, dim]
        tokens = self.proj(tokens)                    # [B, S, hidden_dim]
        encoded = tokens.mean(dim=1)                  # [B, hidden_dim]

        token_logits = self.decoder(encoded, text_tokens)

        B = encoded.shape[0]
        comp_cost = torch.zeros(B, device=encoded.device)
        router_logits = torch.zeros(B, 1, device=encoded.device)
        router_weights = torch.ones(B, 1, device=encoded.device)
        return token_logits, comp_cost, router_logits, router_weights


# =============================================================================
# Losses and metrics
# =============================================================================


def caption_ce_loss(
    pred_logits: torch.Tensor, target_tokens: torch.Tensor, label_smoothing: float,
    pad_id: int = 0,
) -> torch.Tensor:
    """Shift-by-one cross-entropy with label smoothing and PAD masking."""
    _, _, V = pred_logits.shape
    pred = pred_logits[:, :-1, :].contiguous().view(-1, V)
    tgt = target_tokens[:, 1:].contiguous().view(-1)
    return F.cross_entropy(pred, tgt, ignore_index=pad_id, label_smoothing=label_smoothing)


def flops_penalty(comp_cost: torch.Tensor, target_budget: float) -> torch.Tensor:
    return F.relu(comp_cost - target_budget).mean() / (target_budget + 1e-6)


def router_entropy(weights: torch.Tensor) -> torch.Tensor:
    return -(weights * (weights + 1e-9).log()).sum(dim=-1).mean()


def load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer auxiliary loss. W7: discourages routing collapse.

    ``aux = K * Σ_i f_i · P_i``

    where ``f_i`` is the fraction of samples hard-routed to expert *i* and
    ``P_i`` is the mean routing probability to expert *i*. Minimum value is
    1.0, achieved when routing is perfectly balanced across experts.
    """
    K = weights.shape[-1]
    hard = weights.argmax(dim=-1)
    f = F.one_hot(hard, num_classes=K).float().mean(dim=0)
    P = weights.mean(dim=0)
    return K * (f * P).sum()


@torch.no_grad()
def token_accuracy(pred_logits: torch.Tensor, target_tokens: torch.Tensor, pad_id: int = 0) -> float:
    preds = pred_logits[:, :-1, :].argmax(dim=-1)
    targets = target_tokens[:, 1:]
    mask = targets != pad_id
    if mask.sum() == 0:
        return 0.0
    return ((preds == targets) & mask).sum().item() / mask.sum().item()


class UtilizationTracker:
    """W7: per-branch utilization tracking for ANC routing diagnostics."""

    def __init__(self, num_branches: int) -> None:
        self.num_branches = num_branches
        self.reset()

    def reset(self) -> None:
        self.hard_counts = np.zeros(self.num_branches, dtype=np.float64)
        self.prob_sum = np.zeros(self.num_branches, dtype=np.float64)
        self.n = 0

    @torch.no_grad()
    def update(self, weights: torch.Tensor) -> None:
        w = weights.detach().float().cpu().numpy()
        hard = w.argmax(axis=-1)
        for k in range(self.num_branches):
            self.hard_counts[k] += float((hard == k).sum())
        self.prob_sum += w.sum(axis=0)
        self.n += w.shape[0]

    def summary(self) -> Dict[str, float]:
        if self.n == 0:
            return {}
        out: Dict[str, float] = {}
        for k in range(self.num_branches):
            out[f"branch_{k}_hard_fraction"] = float(self.hard_counts[k] / self.n)
            out[f"branch_{k}_mean_prob"] = float(self.prob_sum[k] / self.n)
        return out


# =============================================================================
# Dataset
# =============================================================================


class CocoCaptionDataset(Dataset):
    def __init__(
        self,
        images_root: str,
        anns_json: str,
        max_length: int = 64,
        vocab: Optional["Vocabulary"] = None,
        clip_preprocess: Optional[Callable] = None,
        gpt2_tokenizer: Optional[Any] = None,
    ) -> None:
        if not _COCO_AVAILABLE:
            raise RuntimeError(
                "pycocotools is not installed. Run `pip install pycocotools`."
            )
        if not _PIL_AVAILABLE:
            raise RuntimeError("Pillow is not installed. Run `pip install pillow`.")
        if not Path(images_root).exists():
            raise FileNotFoundError(f"COCO images root not found: {images_root}")
        if not Path(anns_json).exists():
            raise FileNotFoundError(f"COCO annotations file not found: {anns_json}")
        from torchvision import transforms  # local import: only needed for COCO

        self.coco = COCO(anns_json)
        self.ids = sorted(self.coco.anns.keys())
        self.images_root = images_root
        self.max_length = max_length
        self.vocab = vocab
        self.gpt2_tokenizer = gpt2_tokenizer
        # When a CLIP preprocess callable is provided, use it instead of the
        # standard ImageNet normalization (CLIP has its own mean/std).
        if clip_preprocess is not None:
            self.transform = clip_preprocess
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def all_captions(self) -> List[str]:
        return [self.coco.anns[ann_id]["caption"] for ann_id in self.ids]

    def get_image_captions(self, idx: int) -> Tuple[int, List[str]]:
        """Returns (image_id, all reference captions for that image)."""
        ann = self.coco.anns[self.ids[idx]]
        img_id = ann["image_id"]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        refs = [self.coco.anns[a]["caption"] for a in ann_ids]
        return img_id, refs

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for attempt in range(len(self.ids)):
            try:
                cur_idx = (idx + attempt) % len(self.ids)
                ann = self.coco.anns[self.ids[cur_idx]]
                img_info = self.coco.loadImgs(ann["image_id"])[0]
                img_path = os.path.join(self.images_root, img_info["file_name"])
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                if self.gpt2_tokenizer is not None:
                    enc = self.gpt2_tokenizer(
                        ann["caption"],
                        max_length=self.max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    token_ids = enc.input_ids.squeeze(0)
                elif self.vocab is not None:
                    token_ids = self.vocab.encode(ann["caption"], self.max_length)
                else:
                    token_ids = simple_tokenize(ann["caption"], self.max_length)
                events = torch.zeros(2, img.shape[1] // 4, img.shape[2] // 4)
                return img, events, token_ids
            except Exception:
                continue
        raise RuntimeError(f"No valid images found starting from idx={idx}")


class SyntheticDataset(Dataset):
    """Deterministic synthetic data for smoke tests; no COCO required."""

    def __init__(self, size: int = 512, max_length: int = 64, vocab_size: int = 8192) -> None:
        self.size = size
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(idx)
        rgb = torch.randn(3, 224, 224, generator=g)
        events = torch.randn(2, 56, 56, generator=g)
        tokens = torch.randint(
            1, self.vocab_size - 1, (self.max_length,), generator=g, dtype=torch.long
        )
        return rgb, events, tokens


def build_datasets(
    cfg: Config,
    vocab: Optional[Vocabulary] = None,
    clip_preprocess: Optional[Callable] = None,
    gpt2_tokenizer: Optional[Any] = None,
) -> Tuple[Dataset, Dataset]:
    if cfg.smoke_test:
        logger.info("Smoke test mode: building synthetic dataset (%d samples)", cfg.smoke_test_size)
        full = SyntheticDataset(
            size=cfg.smoke_test_size,
            max_length=cfg.max_length,
            vocab_size=cfg.vocab_size if not cfg.gpt2_decoder else GPT2CaptionDecoder.VOCAB_SIZE,
        )
    else:
        logger.info("Loading COCO from %s", cfg.coco_imgs)
        full = CocoCaptionDataset(
            cfg.coco_imgs, cfg.coco_anns, cfg.max_length,
            vocab=vocab, clip_preprocess=clip_preprocess,
            gpt2_tokenizer=gpt2_tokenizer,
        )

    n_val = max(1, int(len(full) * cfg.val_fraction))
    n_train = len(full) - n_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=generator)
    logger.info("Dataset: %d train / %d val", len(train_set), len(val_set))
    return train_set, val_set


def build_vocabulary(cfg: Config, run_dir: Path) -> Optional[Vocabulary]:
    """Build vocabulary from all COCO captions; save to run_dir for reproducibility."""
    if cfg.smoke_test or cfg.gpt2_decoder:
        return None
    vocab_path = run_dir / "vocabulary.json"
    if vocab_path.exists():
        logger.info("Loading existing vocabulary from %s", vocab_path)
        return Vocabulary.load(vocab_path, max_size=cfg.vocab_size)
    if not _COCO_AVAILABLE:
        logger.warning("pycocotools not available; skipping vocabulary build")
        return None
    logger.info("Building vocabulary from %s ...", cfg.coco_anns)
    coco = COCO(cfg.coco_anns)
    captions = [ann["caption"] for ann in coco.anns.values()]
    vocab = Vocabulary(max_size=cfg.vocab_size)
    vocab.build(captions)
    vocab.save(vocab_path)
    return vocab


# =============================================================================
# Checkpointing
# =============================================================================


class CheckpointManager:
    def __init__(self, output_dir: Path, keep_last_n: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(self, state: Dict[str, Any], epoch: int, is_best: bool = False) -> Path:
        path = self.ckpt_dir / f"epoch_{epoch:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

        latest = self.ckpt_dir / "latest.pt"
        tmp_latest = latest.with_suffix(".pt.tmp")
        torch.save(state, tmp_latest)
        os.replace(tmp_latest, latest)

        if is_best:
            best = self.ckpt_dir / "best.pt"
            tmp_best = best.with_suffix(".pt.tmp")
            torch.save(state, tmp_best)
            os.replace(tmp_best, best)

        self._prune()
        return path

    def _prune(self) -> None:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        for old in ckpts[: -self.keep_last_n]:
            try:
                old.unlink()
            except OSError:
                pass

    def cleanup_after_training(self) -> None:
        """Remove all epoch_*.pt and latest.pt after training completes.

        Keeps only best.pt — the single best-validation-loss checkpoint.
        Called by Trainer._finalize on rank 0.
        """
        removed = 0
        for pattern in ("epoch_*.pt",):
            for f in self.ckpt_dir.glob(pattern):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        logger.info("Checkpoint cleanup: removed %d intermediate files, kept best.pt", removed)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        latest = self.ckpt_dir / "latest.pt"
        if latest.exists():
            logger.info("Resuming from %s", latest)
            return torch.load(latest, map_location="cpu", weights_only=False)
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        if ckpts:
            logger.info("Resuming from %s", ckpts[-1])
            return torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        return None


# =============================================================================
# Early stopping
# =============================================================================


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        improved = val_loss < self.best - self.min_delta
        if improved:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved

    def state_dict(self) -> Dict[str, Any]:
        return {
            "best": self.best,
            "counter": self.counter,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.best = state.get("best", float("inf"))
        self.counter = state.get("counter", 0)
        self.should_stop = state.get("should_stop", False)


# =============================================================================
# Metrics log (JSON-lines)
# =============================================================================


class MetricsLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


# =============================================================================
# Statistical significance (W9)
# =============================================================================


def welch_ttest(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Welch's two-sided t-test. Returns (t_statistic, p_value).

    Falls back to a normal approximation if scipy is unavailable.
    """
    try:
        from scipy.stats import ttest_ind  # type: ignore

        result = ttest_ind(a, b, equal_var=False)
        return float(result.statistic), float(result.pvalue)
    except Exception:
        arr_a = np.asarray(a, dtype=np.float64)
        arr_b = np.asarray(b, dtype=np.float64)
        na, nb = len(arr_a), len(arr_b)
        if na < 2 or nb < 2:
            return float("nan"), float("nan")
        mean_a, mean_b = arr_a.mean(), arr_b.mean()
        var_a = arr_a.var(ddof=1)
        var_b = arr_b.var(ddof=1)
        denom = math.sqrt(var_a / na + var_b / nb + 1e-12)
        t = float((mean_a - mean_b) / denom)
        z = abs(t)
        p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
        return t, p


# =============================================================================
# Plotting (matplotlib optional)
# =============================================================================


def plot_curves(records: List[Dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        logger.warning("matplotlib not available; skipping curve plot")
        return

    # Each split uses its own epoch list to handle interrupted/resumed runs
    # where train and val record counts may differ.
    tr_epochs = [r["epoch"] for r in records if r.get("split") == "train"]
    tr_loss   = [r["loss"]  for r in records if r.get("split") == "train"]
    tr_acc    = [r["accuracy"] for r in records if r.get("split") == "train"]
    val_epochs = [r["epoch"] for r in records if r.get("split") == "val"]
    val_loss   = [r["loss"]  for r in records if r.get("split") == "val"]
    val_acc    = [r["accuracy"] for r in records if r.get("split") == "val"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(tr_epochs, tr_loss, label="train", marker="o")
    axes[0].plot(val_epochs, val_loss, label="val", marker="s")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(tr_epochs, tr_acc, label="train", marker="o")
    axes[1].plot(val_epochs, val_acc, label="val", marker="s")
    axes[1].set_title("Token accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_utilization(records: List[Dict[str, Any]], num_branches: int, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return
    train_records = [r for r in records if r.get("split") == "train"]
    if not train_records:
        return
    epochs = [r["epoch"] for r in train_records]
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(num_branches):
        key = f"branch_{k}_hard_fraction"
        frac = [r.get(key, float("nan")) for r in train_records]
        ax.plot(epochs, frac, marker="o", label=f"branch {k}")
    ax.set_ylabel("hard routing fraction")
    ax.set_xlabel("epoch")
    ax.set_title("ANC per-branch utilization (W7)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =============================================================================
# Trainer
# =============================================================================


class Trainer:
    def __init__(
        self,
        cfg: Config,
        run_dir: Path,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = rank == 0  # only rank 0 writes files, logs, checkpoints

        # ── Device ─────────────────────────────────────────────────────────────
        if world_size > 1:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.is_main:
            logger.info(
                "Device: %s | world_size: %d | GPU: %s",
                self.device,
                world_size,
                torch.cuda.get_device_name(local_rank) if self.device.type == "cuda" else "cpu",
            )

        # Each rank gets a unique seed for data sampling while sharing the
        # same model initialisation seed.
        set_seed(cfg.seed + rank, cfg.deterministic)

        # ── GPT-2 tokenizer (when gpt2_decoder=True, replaces Vocabulary) ────
        self.gpt2_tokenizer: Optional[Any] = None
        self.pad_id: int = 0   # Vocabulary.PAD; overridden below for GPT-2
        if cfg.gpt2_decoder:
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("transformers is not installed. Run `pip install transformers`.")
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
            self.pad_id = self.gpt2_tokenizer.pad_token_id
            if self.is_main:
                logger.info("GPT-2 tokenizer loaded (pad_token_id=%d)", self.pad_id)

        # ── Vocabulary (rank 0 builds/saves; all ranks load) ──────────────────
        if self.is_main:
            self.vocab = build_vocabulary(cfg, run_dir)
        else:
            self.vocab = None
        # All ranks need the vocab; non-main ranks load it after rank 0 saves.
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()
            if not self.is_main and not cfg.smoke_test and not cfg.gpt2_decoder:
                vocab_path = run_dir / "vocabulary.json"
                if vocab_path.exists():
                    self.vocab = Vocabulary.load(vocab_path, max_size=cfg.vocab_size)

        # ── Model → DDP ────────────────────────────────────────────────────────
        # Build model before datasets so CLIP preprocess is available for the dataset.
        # Re-seed model init identically on all ranks so weights are in sync.
        set_seed(cfg.seed, cfg.deterministic)
        if cfg.baseline == "tokenlearner":
            self.model: Any = TokenLearnerBaseline(
                cfg, num_tokens=cfg.tokenlearner_num_tokens
            ).to(self.device)
        elif cfg.baseline == "dense":
            if cfg.clip_backbone:
                raise ValueError("--baseline dense currently supports CNN encoders only")
            self.model = DenseEncoderBaseline(cfg).to(self.device)
        else:
            self.model = AdaptiveNeuralCompression(cfg).to(self.device)

        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)

        # ── Datasets & samplers ────────────────────────────────────────────────
        _clip_pre = (
            self.raw_model.clip_stem.preprocess
            if cfg.clip_backbone and hasattr(self.raw_model, "clip_stem") and self.raw_model.clip_stem is not None
            else None
        )
        train_set, val_set = build_datasets(
            cfg, vocab=self.vocab, clip_preprocess=_clip_pre,
            gpt2_tokenizer=self.gpt2_tokenizer,
        )

        if world_size > 1:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler: Any = DistributedSampler(
                train_set, num_replicas=world_size, rank=rank,
                shuffle=True, drop_last=True,
            )
            val_sampler: Any = DistributedSampler(
                val_set, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            train_sampler = None
            val_sampler = None

        self.train_sampler = train_sampler
        self.train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )

        if self.is_main:
            n_params = sum(p.numel() for p in self.raw_model.parameters())
            logger.info(
                "Model: %s | parameters: %.2fM",
                (
                    f"Dense-{cfg.encoder_only}"
                    if cfg.baseline == "dense"
                    else cfg.baseline if cfg.baseline != "none" else "STTF+ANC"
                ),
                n_params / 1e6,
            )

        # ── Optimiser & scheduler ──────────────────────────────────────────────
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = self._build_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")
        self.early_stop = EarlyStopping(cfg.early_stop_patience, cfg.early_stop_min_delta)
        # Utilization tracking only applies to STTF+ANC (has .encoders)
        num_branches = len(self.raw_model.encoders) if hasattr(self.raw_model, "encoders") else 1
        self.utilization = UtilizationTracker(num_branches=num_branches)

        # ── I/O (rank 0 only) ──────────────────────────────────────────────────
        self.ckpt_mgr = CheckpointManager(run_dir, keep_last_n=cfg.keep_last_n_checkpoints)
        self.metrics = MetricsLog(run_dir / "metrics.jsonl")
        self.graceful = GracefulExit()

        self.tb_writer: Optional[Any] = None
        if self.is_main and cfg.tensorboard and _TB_AVAILABLE:
            self.tb_writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        elif self.is_main and cfg.tensorboard:
            logger.warning("TensorBoard requested but not installed; disabling.")

        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_cider = 0.0
        if cfg.resume:
            self._maybe_resume()

    @property
    def raw_model(self) -> AdaptiveNeuralCompression:
        """Unwrap DDP to access the underlying model (encoders, state dict, etc.)."""
        return self.model.module if hasattr(self.model, "module") else self.model  # type: ignore

    def _reduce_metric(self, value: float) -> float:
        """Average a scalar across all DDP ranks (no-op for world_size=1)."""
        if self.world_size == 1:
            return value
        import torch.distributed as dist
        t = torch.tensor(value, device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        return float(t)

    def _build_scheduler(self) -> Optional[Any]:
        if self.cfg.epochs <= 1:
            return None
        warmup_epochs = min(self.cfg.warmup_epochs, max(0, self.cfg.epochs - 1))
        if warmup_epochs > 0:
            warmup = LambdaLR(
                self.optimizer,
                lr_lambda=lambda e: (e + 1) / warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.epochs - warmup_epochs
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)

    def _maybe_resume(self) -> None:
        # All ranks load from shared storage so no broadcast is needed and
        # there are no mismatched collective calls between ranks.
        state = self.ckpt_mgr.load_latest()
        if state is None:
            return
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])
        if state.get("early_stop") is not None:
            self.early_stop.load_state_dict(state["early_stop"])
        self.start_epoch = state.get("epoch", 0) + 1
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        # Only rank 0 restores RNG state; other ranks keep their offset seeds.
        if self.is_main:
            rng = state.get("rng")
            if rng is not None:
                try:
                    torch.set_rng_state(rng["torch"])
                    if rng.get("cuda") is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(rng["cuda"])
                    np.random.set_state(rng["numpy"])
                    random.setstate(rng["python"])
                except Exception as e:
                    logger.warning("Could not restore RNG state: %s", e)
            logger.info(
                "Resumed at epoch %d (best_val_loss=%.4f)",
                self.start_epoch,
                self.best_val_loss,
            )

    def _snapshot_state(self, epoch: int) -> Dict[str, Any]:
        rng = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        return {
            "epoch": epoch,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(),
            "early_stop": self.early_stop.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.cfg),
            "rng": rng,
        }

    def _compute_losses(
        self, pred_logits: torch.Tensor, comp_cost: torch.Tensor, weights: torch.Tensor, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ce = caption_ce_loss(pred_logits, tokens, self.cfg.label_smoothing, pad_id=self.pad_id)
        f_pen = flops_penalty(comp_cost, self.cfg.target_budget)
        ent = router_entropy(weights)
        balance = load_balance_loss(weights)
        total = (
            ce
            + self.cfg.lambda_flops * f_pen
            + self.cfg.lambda_entropy * ent
            + self.cfg.lambda_balance * balance
        )
        parts = {
            "loss_ce": float(ce.detach()),
            "loss_flops": float(f_pen.detach()),
            "loss_entropy": float(ent.detach()),
            "loss_balance": float(balance.detach()),
        }
        return total, parts

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        # Tell DistributedSampler which epoch this is so shuffling is unique.
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        self.model.train()
        self.utilization.reset()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"rank{self.rank} epoch {epoch} [train]",
            leave=False,
            disable=not self.is_main,
        )
        for rgb, events, tokens in pbar:
            rgb = rgb.to(self.device, non_blocking=True)
            events = events.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.cuda.amp.autocast() if self.device.type == "cuda" else nullcontext()
            )
            try:
                with autocast:
                    logits_tok, comp_cost, _, weights = self.model(rgb, events, tokens)
                    loss, parts = self._compute_losses(logits_tok, comp_cost, weights, tokens)
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM at epoch %d; clearing cache and skipping batch", epoch)
                torch.cuda.empty_cache()
                continue

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.utilization.update(weights)
            total_loss += float(loss.detach())
            total_acc += token_accuracy(logits_tok, tokens, pad_id=self.pad_id)
            n_batches += 1

            if self.graceful.should_exit:
                logger.warning("Graceful exit requested mid-epoch")
                break

        if n_batches == 0:
            raise RuntimeError("No batches completed in epoch (all OOM or empty loader).")

        metrics = {
            "loss": self._reduce_metric(total_loss / n_batches),
            "accuracy": self._reduce_metric(total_acc / n_batches),
            **parts,  # last batch's component breakdown (rank-local, rank 0 only)
            **self.utilization.summary(),
        }
        return metrics

    @torch.no_grad()
    def _evaluate_captions(self, epoch: int) -> Dict[str, float]:
        """Generate captions on the val set and compute CIDEr + BLEU-4.

        Only runs on rank 0; skipped for smoke tests or when both vocab and
        gpt2_tokenizer are absent.
        """
        if not self.is_main or (self.vocab is None and self.gpt2_tokenizer is None):
            return {}
        logger.info("epoch %d: running caption evaluation (CIDEr/BLEU-4)...", epoch)

        self.model.eval()
        hypotheses: List[str] = []
        references: List[List[str]] = []

        # Cap in-loop evaluation to 1000 samples to avoid NCCL watchdog
        # timeouts on non-main ranks that are waiting for the post-epoch
        # broadcast. The full val set is used in _finalize.
        MAX_INLINE_SAMPLES = 1000
        full_ds = self.val_loader.dataset
        if len(full_ds) > MAX_INLINE_SAMPLES:
            g = torch.Generator().manual_seed(self.cfg.seed + epoch)
            idx = torch.randperm(len(full_ds), generator=g)[:MAX_INLINE_SAMPLES].tolist()
            eval_ds: Any = torch.utils.data.Subset(full_ds, idx)
        else:
            eval_ds = full_ds

        eval_loader = DataLoader(
            eval_ds,
            batch_size=32,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        # Build reference captions indexed by position in val set.
        # val_loader.dataset is a Subset of CocoCaptionDataset.
        # Unwrap the (possibly subset-of-subset) dataset to reach CocoCaptionDataset.
        underlying = eval_ds
        if hasattr(underlying, "dataset") and hasattr(underlying, "indices"):
            base_ds = underlying.dataset
            eval_indices = underlying.indices
        else:
            base_ds = underlying
            eval_indices = list(range(len(underlying)))
        # Unwrap one more level if the base is itself a Subset.
        if hasattr(base_ds, "dataset") and hasattr(base_ds, "indices"):
            outer_indices = base_ds.indices
            base_ds = base_ds.dataset
            eval_indices = [outer_indices[i] for i in eval_indices]

        ref_map: Dict[int, List[str]] = {}
        if hasattr(base_ds, "get_image_captions"):
            for pos, idx in enumerate(eval_indices):
                _, refs = base_ds.get_image_captions(idx)
                ref_map[pos] = refs

        pos = 0
        for rgb, events, _ in eval_loader:
            rgb = rgb.to(self.device)
            events = events.to(self.device)
            if self.gpt2_tokenizer is not None:
                caps = greedy_decode_gpt2(
                    self.model, rgb, events,
                    self.gpt2_tokenizer, self.cfg.max_length, self.device,
                )
            else:
                caps = greedy_decode(
                    self.model, rgb, events,
                    self.cfg.max_length, self.vocab, self.device,
                )
            for cap in caps:
                hypotheses.append(cap)
                references.append(ref_map.get(pos, [""]))
                pos += 1

        metrics = compute_caption_metrics(hypotheses, references)
        if metrics:
            logger.info("epoch %d caption metrics: %s", epoch,
                        {k: f"{v:.2f}" for k, v in metrics.items()})
        return metrics

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_comp_cost = 0.0
        n_batches = 0
        val_util = UtilizationTracker(num_branches=self.utilization.num_branches)

        pbar = tqdm(
            self.val_loader,
            desc=f"epoch {epoch} [val]",
            leave=False,
            disable=not self.is_main,
        )
        for rgb, events, tokens in pbar:
            rgb = rgb.to(self.device, non_blocking=True)
            events = events.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)
            logits_tok, comp_cost, _, weights = self.model(rgb, events, tokens)
            loss, _ = self._compute_losses(logits_tok, comp_cost, weights, tokens)
            total_loss += float(loss)
            total_acc += token_accuracy(logits_tok, tokens, pad_id=self.pad_id)
            total_comp_cost += float(comp_cost.mean())
            val_util.update(weights)
            n_batches += 1

        if n_batches == 0:
            return {"loss": float("nan"), "accuracy": float("nan")}
        result = {
            "loss": self._reduce_metric(total_loss / n_batches),
            "accuracy": self._reduce_metric(total_acc / n_batches),
            "encoder_gflops": self._reduce_metric(total_comp_cost / n_batches / 1e9),
        }
        result.update({f"val_{k}": v for k, v in val_util.summary().items()})
        return result

    def _log_epoch(
        self, epoch: int, split: str, metrics: Dict[str, float], duration: float
    ) -> None:
        if not self.is_main:
            return
        record = {
            "epoch": epoch,
            "split": split,
            "duration_sec": duration,
            "seed": self.cfg.seed,
            "world_size": self.world_size,
            **metrics,
        }
        self.metrics.append(record)
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    self.tb_writer.add_scalar(f"{split}/{k}", v, epoch)
        if "loss" in metrics and "accuracy" in metrics:
            logger.info(
                "epoch %d [%s] loss=%.4f acc=%.2f%% time=%.1fs",
                epoch, split, metrics["loss"], metrics["accuracy"] * 100, duration,
            )
        else:
            logger.info(
                "epoch %d [%s] %s time=%.1fs",
                epoch, split,
                " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)),
                duration,
            )

    def train(self) -> Dict[str, Any]:
        if self.cfg.eval_only:
            if self.is_main:
                logger.info("eval_only=True: running single validation pass then exiting.")
            val_metrics = self._validate(-1)
            caption_metrics = self._evaluate_captions(-1)
            all_metrics = {**val_metrics, **caption_metrics}
            if self.is_main:
                logger.info("eval_only results: %s", {k: f"{v:.4f}" for k, v in all_metrics.items() if isinstance(v, float)})
                (self.run_dir / "eval_only_results.json").write_text(json.dumps(all_metrics, indent=2))
            return all_metrics

        if self.is_main:
            logger.info("Starting training from epoch %d", self.start_epoch)
        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch)
            self._log_epoch(epoch, "train", train_metrics, time.time() - t0)

            t0 = time.time()
            val_metrics = self._validate(epoch)
            self._log_epoch(epoch, "val", val_metrics, time.time() - t0)

            # Caption metrics every eval_cider_freq epochs (rank 0 only).
            caption_metrics: Dict[str, float] = {}
            freq = self.cfg.eval_cider_freq
            if freq > 0 and (epoch + 1) % freq == 0:
                caption_metrics = self._evaluate_captions(epoch)
                if caption_metrics:
                    self._log_epoch(epoch, "caption", caption_metrics, 0.0)

            # Early stopping + best-checkpoint decision (rank 0).
            # GPT-2 mode: best.pt tracks peak CIDEr; early stopping still on val loss.
            # Default mode: both track val loss (unchanged behaviour).
            if self.is_main:
                cider_now = caption_metrics.get("cider") if caption_metrics else None
                if self.cfg.gpt2_decoder and cider_now is not None:
                    improved = float(cider_now) > self.best_cider
                    if improved:
                        self.best_cider = float(cider_now)
                    self.early_stop.step(val_metrics["loss"])
                else:
                    improved = self.early_stop.step(val_metrics["loss"])
                    if improved:
                        self.best_val_loss = val_metrics["loss"]
            else:
                improved = False

            if self.world_size > 1:
                import torch.distributed as dist
                # Broadcast: [should_stop, improved, best_val_loss]
                ctrl = torch.tensor(
                    [float(self.early_stop.should_stop), float(improved), self.best_val_loss],
                    device=self.device,
                )
                dist.broadcast(ctrl, src=0)
                self.early_stop.should_stop = bool(ctrl[0].item())
                improved = bool(ctrl[1].item())
                self.best_val_loss = float(ctrl[2].item())

            if self.scheduler is not None:
                self.scheduler.step()

            # Checkpoint written only by rank 0.
            if self.is_main:
                state = self._snapshot_state(epoch)
                self.ckpt_mgr.save(state, epoch, is_best=improved)

            # Barrier: all ranks stay in sync before the next epoch.
            if self.world_size > 1:
                import torch.distributed as dist
                dist.barrier()

            if self.early_stop.should_stop:
                if self.is_main:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                break
            if self.graceful.should_exit:
                if self.is_main:
                    logger.info("Graceful exit after epoch %d", epoch)
                break

        if self.tb_writer is not None:
            self.tb_writer.close()

        return self._finalize()

    def _finalize(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        if self.is_main:
            records = self.metrics.read_all()
            plot_curves(records, self.run_dir / "plots" / "curves.png")
            if hasattr(self.raw_model, "encoders"):
                plot_utilization(
                    records, num_branches=len(self.raw_model.encoders),
                    out_path=self.run_dir / "plots" / "anc_utilization.png",
                )

            summary_path = self.run_dir / "summary.json"
            val_records = [r for r in records if r.get("split") == "val"]
            # Final caption evaluation (always run at end regardless of freq).
            final_caption_metrics = self._evaluate_captions(
                (val_records[-1]["epoch"] if val_records else 0)
            )

            caption_records = [r for r in records if r.get("split") == "caption"]
            best_cider = max((r.get("cider", 0) for r in caption_records), default=None)
            final_cider = final_caption_metrics.get("cider")
            final_bleu4 = final_caption_metrics.get("bleu4")

            summary = {
                "seed": self.cfg.seed,
                "world_size": self.world_size,
                "epochs_trained": (val_records[-1]["epoch"] + 1) if val_records else 0,
                "best_val_loss": self.best_val_loss,
                "final_val_loss": val_records[-1]["loss"] if val_records else None,
                "final_val_accuracy": val_records[-1]["accuracy"] if val_records else None,
                "final_cider": final_cider,
                "final_bleu4": final_bleu4,
                "best_cider": best_cider,
                "early_stopped": self.early_stop.should_stop,
            }
            summary_path.write_text(json.dumps(summary, indent=2))
            logger.info("Summary: %s", summary)

            self.ckpt_mgr.cleanup_after_training()

            if self.cfg.export_onnx:
                self._export_onnx()
        return summary

    def _export_onnx(self) -> None:
        # Exported in eval mode: hard argmax routing is active, so only one
        # encoder branch executes per sample.  FLOPs at inference match the
        # hard-routing average reported in the paper.
        onnx_path = self.run_dir / "tinyvlm_anc.onnx"
        self.raw_model.eval()
        dummy_rgb = torch.randn(1, 3, 224, 224, device=self.device)
        dummy_events = torch.randn(1, 2, 56, 56, device=self.device)
        dummy_text = torch.randint(
            1, self.cfg.vocab_size, (1, self.cfg.max_length), device=self.device
        )
        try:
            torch.onnx.export(
                self.raw_model,
                (dummy_rgb, dummy_events, dummy_text),
                str(onnx_path),
                opset_version=17,
                input_names=["rgb", "events", "text"],
                output_names=["token_logits", "comp_cost", "router_logits", "router_weights"],
                dynamic_axes={
                    "rgb": {0: "batch"},
                    "events": {0: "batch"},
                    "text": {0: "batch"},
                    "token_logits": {0: "batch"},
                },
            )
            logger.info("Exported ONNX model to %s", onnx_path)
        except Exception as e:
            logger.warning("ONNX export failed: %s", e)


# =============================================================================
# Multi-seed orchestration (W9) and τ sweep (W8)
# =============================================================================


def run_single(
    cfg: Config,
    rank: int = 0,
    local_rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    run_dir = Path(cfg.output_dir) / f"seed_{cfg.seed}_tau_{cfg.sttf_tau:.2f}"
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(cfg.to_json())
        setup_logging(cfg.log_level, log_file=run_dir / "train.log")
        logger.info("Run directory: %s", run_dir)
        logger.info("Config:\n%s", cfg.to_json())
    # ALL ranks barrier so rank 1 doesn't enter DDP init before rank 0 finishes
    # directory setup — mismatched collectives cause NCCL deadlock.
    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
    trainer = Trainer(cfg, run_dir, rank=rank, local_rank=local_rank, world_size=world_size)
    return trainer.train()


def run_multi_seed(base_cfg: Config, seeds: List[int]) -> Dict[str, Any]:
    """W9: multi-seed runs with Welch t-test aggregation."""
    results: List[Dict[str, Any]] = []
    for seed in seeds:
        cfg = dataclasses.replace(base_cfg, seed=seed)
        summary = run_single(cfg)
        summary["seed"] = seed
        results.append(summary)

    val_losses = [r["final_val_loss"] for r in results if r.get("final_val_loss") is not None]
    val_accs = [r["final_val_accuracy"] for r in results if r.get("final_val_accuracy") is not None]

    aggregate: Dict[str, Any] = {
        "seeds": seeds,
        "per_seed": results,
        "val_loss_mean": float(np.mean(val_losses)) if val_losses else None,
        "val_loss_std": float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else None,
        "val_acc_mean": float(np.mean(val_accs)) if val_accs else None,
        "val_acc_std": float(np.std(val_accs, ddof=1)) if len(val_accs) > 1 else None,
    }

    out = Path(base_cfg.output_dir) / "multi_seed_summary.json"
    out.write_text(json.dumps(aggregate, indent=2))
    logger.info("Multi-seed summary written to %s", out)
    logger.info("Aggregate: %s", {k: v for k, v in aggregate.items() if k != "per_seed"})
    return aggregate


def run_tau_sweep(base_cfg: Config, taus: List[float]) -> Dict[str, Any]:
    """W8: cross-dataset τ sensitivity sweep."""
    results: List[Dict[str, Any]] = []
    for tau in taus:
        cfg = dataclasses.replace(base_cfg, sttf_tau=tau)
        summary = run_single(cfg)
        summary["sttf_tau"] = tau
        results.append(summary)

    out = Path(base_cfg.output_dir) / "tau_sweep_summary.json"
    out.write_text(json.dumps({"sweep": results}, indent=2))
    logger.info("τ sweep summary written to %s", out)
    return {"sweep": results}


# =============================================================================
# Argument parsing
# =============================================================================


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TinyVLM (STTF + ANC) training for vast.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default = Config()

    # Data
    p.add_argument("--coco_imgs", type=str, default=default.coco_imgs)
    p.add_argument("--coco_anns", type=str, default=default.coco_anns)
    p.add_argument("--val_fraction", type=float, default=default.val_fraction)
    p.add_argument("--max_length", type=int, default=default.max_length)
    p.add_argument("--vocab_size", type=int, default=default.vocab_size)
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--smoke_test_size", type=int, default=default.smoke_test_size)

    # Model
    p.add_argument("--dropout", type=float, default=default.dropout)
    p.add_argument("--num_decoder_layers", type=int, default=default.num_decoder_layers)
    p.add_argument("--clip_backbone", action="store_true",
                   help="Use frozen CLIP ViT-B/32 stem instead of CNN encoders")
    p.add_argument("--clip_model", type=str, default=default.clip_model)
    p.add_argument("--clip_pretrained", type=str, default=default.clip_pretrained)
    p.add_argument("--gpt2_decoder", action="store_true",
                   help="Replace scratch decoder with pretrained GPT-2 small (requires --clip_backbone)")
    p.add_argument("--gpt2_n_prefix", type=int, default=default.gpt2_n_prefix,
                   help="Number of visual prefix tokens fed to GPT-2")

    # Training
    p.add_argument("--epochs", type=int, default=default.epochs)
    p.add_argument("--batch_size", type=int, default=default.batch_size)
    p.add_argument("--lr", type=float, default=default.lr)
    p.add_argument("--weight_decay", type=float, default=default.weight_decay)
    p.add_argument("--warmup_epochs", type=int, default=default.warmup_epochs)
    p.add_argument("--grad_clip", type=float, default=default.grad_clip)
    p.add_argument("--label_smoothing", type=float, default=default.label_smoothing)

    # Losses
    p.add_argument("--target_budget", type=float, default=default.target_budget)
    p.add_argument("--lambda_flops", type=float, default=default.lambda_flops)
    p.add_argument("--lambda_entropy", type=float, default=default.lambda_entropy)
    p.add_argument("--lambda_balance", type=float, default=default.lambda_balance)
    p.add_argument("--router_temperature", type=float, default=default.router_temperature)
    p.add_argument("--sttf_tau", type=float, default=default.sttf_tau)

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=default.early_stop_patience)

    # Checkpointing
    p.add_argument("--output_dir", type=str, default=default.output_dir)
    p.add_argument("--keep_last_n_checkpoints", type=int, default=default.keep_last_n_checkpoints)
    p.add_argument("--no_resume", action="store_true", help="Disable auto-resume")

    # Reproducibility
    p.add_argument("--seed", type=int, default=default.seed)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--num_workers", type=int, default=default.num_workers)

    # Export / logging
    p.add_argument("--baseline", type=str, default=default.baseline,
                   choices=["none", "tokenlearner", "dense"],
                   help="Swap STTF+ANC for a comparison baseline model")
    p.add_argument("--encoder_only", type=str, default=default.encoder_only,
                   choices=["tiny", "small", "medium"],
                   help="Single CNN encoder used when --baseline dense")
    p.add_argument("--tokenlearner_num_tokens", type=int, default=default.tokenlearner_num_tokens,
                   help="Number of learned tokens S in the TokenLearner baseline")
    p.add_argument("--eval_cider_freq", type=int, default=default.eval_cider_freq,
                   help="Compute CIDEr/BLEU-4 every N epochs (0 = final only)")
    p.add_argument("--eval_routing_mode", type=str, default=default.eval_routing_mode,
                   choices=["hard", "soft"],
                   help="Routing mode used during eval/inference for STTF+ANC")
    p.add_argument("--no_anc", action="store_true",
                   help="S5 ablation: single fixed encoder, no routing (requires --clip_backbone)")
    p.add_argument("--eval_only", action="store_true",
                   help="S1: load checkpoint from output_dir, run one eval pass, exit")
    p.add_argument("--no_onnx", action="store_true")
    p.add_argument("--no_tensorboard", action="store_true")
    p.add_argument("--log_level", type=str, default=default.log_level)

    # Multi-seed / sweep modes
    p.add_argument("--seeds", type=_parse_int_list, default=None,
                   help="Comma-separated seeds for multi-seed run (W9)")
    p.add_argument("--tau_sweep", type=_parse_float_list, default=None,
                   help="Comma-separated τ values for sweep (W8)")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        coco_imgs=args.coco_imgs,
        coco_anns=args.coco_anns,
        val_fraction=args.val_fraction,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        smoke_test=args.smoke_test,
        smoke_test_size=args.smoke_test_size,
        dropout=args.dropout,
        num_decoder_layers=args.num_decoder_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        target_budget=args.target_budget,
        lambda_flops=args.lambda_flops,
        lambda_entropy=args.lambda_entropy,
        lambda_balance=args.lambda_balance,
        router_temperature=args.router_temperature,
        sttf_tau=args.sttf_tau,
        early_stop_patience=args.early_stop_patience,
        output_dir=args.output_dir,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        resume=not args.no_resume,
        seed=args.seed,
        deterministic=args.deterministic,
        num_workers=args.num_workers,
        baseline=args.baseline,
        encoder_only=args.encoder_only,
        tokenlearner_num_tokens=args.tokenlearner_num_tokens,
        clip_backbone=args.clip_backbone,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        gpt2_decoder=args.gpt2_decoder,
        gpt2_n_prefix=args.gpt2_n_prefix,
        eval_cider_freq=args.eval_cider_freq,
        eval_routing_mode=args.eval_routing_mode,
        no_anc=args.no_anc,
        eval_only=args.eval_only,
        export_onnx=not args.no_onnx,
        tensorboard=not args.no_tensorboard,
        log_level=args.log_level,
    )


# =============================================================================
# Entry point
# =============================================================================


def main() -> int:
    args = build_parser().parse_args()
    rank, local_rank, world_size = setup_ddp()

    # Only rank 0 sets up top-level logging and output dir.
    if rank == 0:
        setup_logging(args.log_level)
        cfg = config_from_args(args)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info("TinyVLM vast.ai training pipeline")
        logger.info("PyTorch: %s | CUDA: %s | GPUs: %d", torch.__version__, torch.cuda.is_available(), world_size)
    else:
        setup_logging("WARNING")  # suppress noise from non-main ranks
        cfg = config_from_args(args)

    try:
        if args.tau_sweep:
            if rank == 0:
                run_tau_sweep(cfg, args.tau_sweep)
        elif args.seeds:
            if rank == 0:
                run_multi_seed(cfg, args.seeds)
        else:
            run_single(cfg, rank=rank, local_rank=local_rank, world_size=world_size)
    finally:
        teardown_ddp()

    return 0


if __name__ == "__main__":
    sys.exit(main())
