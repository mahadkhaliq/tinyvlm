"""
Phase-2 pivotal experiment: token-budget routing on a frozen CLIP ViT-B/32.

Motivation (see PUBLICATION_READINESS_2026-07-03.md / OPUS48_PLAN_A Phase 2):
the shipped CLIP-ANC variant routes among three MLP heads (128/256/384-d) that all
sit on top of the SAME full ViT-B/32 forward. The stem (~4.29 G) dominates, so the
per-branch FLOPs differ by ~0.26 M -> branches are iso-cost -> routing is provably
inert (Karpathy-test dense-control tie, Welch p=0.72). This module makes the three
branches process DIFFERENT visual-token budgets INSIDE the frozen ViT, so the stem
FLOPs themselves differ per branch:

    Branch 0 : CLS + 24 strided patch tokens  (~0.50x stem -> ~2.13 GFLOPs)
    Branch 1 : CLS + 36 strided patch tokens  (~0.74x stem -> ~3.17 GFLOPs)
    Branch 2 : CLS + 49 patch tokens (full)   (~1.00x stem -> ~4.29 GFLOPs)

Token selection is a DETERMINISTIC even stride over the 49 patches (static indices,
no data-dependent gather) so the per-branch graph is ONNX-exportable with static
shapes (AI Hub requirement; see CLAUDE.md ONNX gotchas).

The router / complexity estimator consumes a CHEAP RGB-derived signal (a small conv
net on a downsampled image), NOT the full CLS embedding -- otherwise the expensive
stem would have to run before routing, defeating the saving. This also removes the
E8 failure mode (the CNN router read a zeroed event tensor -> constant signal ->
100% Medium collapse); here the signal is the actual image.

This file is standalone and does not mutate the shared training script. See
integrate.md for the exact patch that wires it into tinyvlm_vast.py behind a
--clip_token_budget flag. Run `python clip_token_budget.py` for the smoke test.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
    _OPEN_CLIP = True
except Exception:  # pragma: no cover
    _OPEN_CLIP = False

# Reported full-stem FLOPs, kept identical to CLIPStemEncoder.FLOPS in tinyvlm_vast.py
# so paper numbers stay on one convention. Per-branch costs are scaled to this anchor.
# CORRECTED 2026-07-16. This was 17.6e9, which is the canonical ViT-B/**16** figure; this model is
# ViT-B/**32**. The anchor was ~4.1x too large, inflating every ABSOLUTE FLOPs number derived from it.
# Relative savings were unaffected (the constant cancels in stem_flops ratios), so no conclusion moves.
# 4.29e9 = (12*N*d^2 + 2*N^2*d)*L at N=50, d=768, L=12 -- multiply-accumulates counted as FLOPs, the
# convention under which ViT-B/16 is 17.6 G and ViT-B/32 is 4.4 G. Verified: the same formula returns
# 17.45 G at N=197 (B/16). Do NOT substitute an fvcore count here -- fvcore does not trace
# nn.MultiheadAttention and silently omits the attention block (returns 2.95 G).
FULL_STEM_FLOPS: float = 4.29e9
KEEP_BUDGETS = (24, 36, 49)  # branch 0 / 1 / 2 patch-token budgets (49 = full)


def strided_keep_indices(n_patches: int, k: int) -> list[int]:
    """CLS (index 0) + k patch tokens evenly strided over [1, n_patches].

    Deterministic and input-independent -> static ONNX indices. For k >= n_patches
    all patches are kept (full forward)."""
    if k >= n_patches:
        return list(range(n_patches + 1))
    if k <= 1:
        return [0, 1]
    step = (n_patches - 1) / (k - 1)
    patch_idx = sorted({1 + round(i * step) for i in range(k)})
    return [0] + patch_idx


class TokenBudgetCLIPStem(nn.Module):
    """Frozen ViT-B/32 whose forward runs the transformer over CLS + keep_k patch
    tokens, so stem FLOPs scale with keep_k. Reconstructs the exact open_clip
    VisionTransformer forward (verified bit-identical at keep_k=full).

    Two token-selection rules (`select=`):

      "stride"  (default, the pre-registered rule) -- deterministic even stride over the
                49 patches. Input-independent -> static ONNX indices -> exportable.
                Cost: 12 layers at (keep_k+1) tokens.

      "attn"    (E21) -- EViT-style CLS-attention top-k. Runs block 0 at FULL width,
                ranks patches by the CLS query's attention to them, keeps the top-k, and
                runs blocks 1..11 on the reduced set. This is the literature-standard
                informed selector (cf. EViT/DynamicViT) and the falsification test for the
                paper's C4 claim.

    IMPORTANT -- "attn" is NOT free and NOT a drop-in cost match for "stride":
      * Cost is 1 layer at full width + 11 layers at (keep_k+1), i.e. strictly MORE than
        stride's 12 at (keep_k+1). At keep_k=24 that is ~2.31 G vs stride's ~2.13 G (+8%).
        `stem_flops(keep_k, select=...)` returns the right number for each rule -- do not
        reuse stride's figure for an attn run.
      * Selection is data-dependent (a per-sample gather), so an "attn" branch is NOT
        ONNX-exportable with static shapes. That is fine: E21 is a quality experiment, not
        a latency one. Do not try to export it (see CLAUDE.md ONNX gotchas).
      * At keep_k >= n_patches both rules are identical no-ops and remain bit-identical to
        the untouched ViT.
    """

    OUT_DIM: int = 512

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        super().__init__()
        if not _OPEN_CLIP:
            raise RuntimeError("open_clip_torch not installed.")
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        v = model.visual
        self.conv1 = v.conv1
        self.class_embedding = v.class_embedding
        self.positional_embedding = v.positional_embedding
        self.ln_pre = v.ln_pre
        self.transformer = v.transformer
        self.ln_post = v.ln_post
        self.proj = v.proj
        # open_clip ViT-B-32 transformer is batch_first=True (verified) -> no LND permute.
        self._batch_first = bool(getattr(v.transformer, "batch_first", True))
        self.width = int(self.class_embedding.shape[0])                 # 768
        self.n_patches = int(self.positional_embedding.shape[0]) - 1    # 49
        self.n_layers = len(self.transformer.resblocks)                 # 12
        self.requires_grad_(False)

    def _embed_tokens(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(rgb)                                # [B, width, g, g]
        b, w, g1, g2 = x.shape
        x = x.reshape(b, w, g1 * g2).permute(0, 2, 1)      # [B, n_patches, width]
        cls = self.class_embedding.to(x.dtype) + torch.zeros(
            b, 1, w, dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)                     # [B, n_patches+1, width]
        x = x + self.positional_embedding.to(x.dtype)
        return self.ln_pre(x)

    def _cls_attn_topk_gather(self, x: torch.Tensor, keep_k: int) -> torch.Tensor:
        """Run block 0 at full width, rank patches by CLS attention, gather top-k.

        Returns [B, keep_k+1, width] with CLS at position 0, patch order preserved
        (we sort the kept indices so positional semantics stay monotone)."""
        blk = self.transformer.resblocks[0]
        h = blk.ln_1(x)
        # need_weights=True -> [B, T, T] averaged over heads; row 0 is the CLS query.
        attn_out, attn_w = blk.attn(h, h, h, need_weights=True, average_attn_weights=True)
        x = x + blk.ls_1(attn_out)
        x = x + blk.ls_2(blk.mlp(blk.ln_2(x)))             # block 0 complete, full width

        cls_to_patch = attn_w[:, 0, 1:]                    # [B, n_patches]
        topk = cls_to_patch.topk(keep_k, dim=-1).indices   # [B, keep_k]
        topk, _ = torch.sort(topk, dim=-1)                 # keep patch order monotone
        idx = torch.cat([torch.zeros_like(topk[:, :1]), topk + 1], dim=1)  # prepend CLS
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    @torch.no_grad()
    def forward(
        self, rgb: torch.Tensor, keep_k: int | None = None, select: str = "stride"
    ) -> torch.Tensor:
        x = self._embed_tokens(rgb)                        # [B, 50, width]
        prune = keep_k is not None and keep_k < self.n_patches

        if prune and select == "attn":
            # EViT-style: block 0 at full width supplies the ranking, then prune.
            x = self._cls_attn_topk_gather(x, keep_k)      # [B, keep_k+1, width]
            blocks = self.transformer.resblocks[1:]
            if not self._batch_first:
                x = x.permute(1, 0, 2)
            for blk in blocks:
                x = blk(x)
            if not self._batch_first:
                x = x.permute(1, 0, 2)
        else:
            if prune:
                if select != "stride":
                    raise ValueError(f"unknown select={select!r}; use 'stride' or 'attn'")
                idx = torch.tensor(
                    strided_keep_indices(self.n_patches, keep_k), device=x.device
                )
                x = x.index_select(1, idx)                 # [B, keep_k+1, width] (static idx)
            if not self._batch_first:
                x = x.permute(1, 0, 2)
            x = self.transformer(x)
            if not self._batch_first:
                x = x.permute(1, 0, 2)

        x = self.ln_post(x[:, 0, :])                       # CLS
        if self.proj is not None:
            x = x @ self.proj
        return x                                           # [B, 512]

    def _relative_transformer_flops(self, keep_k: int) -> float:
        n = min(keep_k, self.n_patches) + 1
        d = self.width
        per_layer = 12.0 * n * d * d + 2.0 * n * n * d     # qkv+proj+mlp (12 N d^2) + attn (2 N^2 d)
        return self.n_layers * per_layer

    def _per_layer_relative(self, n_tokens: int) -> float:
        d = self.width
        return 12.0 * n_tokens * d * d + 2.0 * n_tokens * n_tokens * d

    def stem_flops(self, keep_k: int, select: str = "stride") -> float:
        """Per-branch stem FLOPs, MACs-as-FLOPs convention (scaled so keep_k=full == FULL_STEM_FLOPS).

        select="stride": 12 layers at (keep_k+1) tokens.
        select="attn"  : 1 layer at FULL width (it produces the ranking) + 11 at (keep_k+1).
                         Strictly more than stride -- an attn branch is NOT cost-matched to
                         the stride branch of the same keep_k. Report the number this
                         function gives for the rule you actually ran."""
        full_relative = self._relative_transformer_flops(self.n_patches)
        if keep_k >= self.n_patches:
            return FULL_STEM_FLOPS
        if select == "stride":
            rel = self._relative_transformer_flops(keep_k)
        elif select == "attn":
            n_full = self.n_patches + 1
            rel = self._per_layer_relative(n_full) + (self.n_layers - 1) * self._per_layer_relative(keep_k + 1)
        else:
            raise ValueError(f"unknown select={select!r}; use 'stride' or 'attn'")
        return FULL_STEM_FLOPS * rel / full_relative


class RGBComplexityEstimator(nn.Module):
    """Cheap RGB-derived router signal: 2-conv net on a downsampled image -> K logits.
    Runs BEFORE the budgeted stem, so routing does not require the full ViT. Its non-
    constant (image-dependent) output is what removes the E8 zero-signal collapse."""

    def __init__(self, n_branches: int = 3, in_res: int = 32) -> None:
        super().__init__()
        self.in_res = in_res
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GELU(),   # 32->16
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),  # 16->8
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, n_branches),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(rgb, size=(self.in_res, self.in_res), mode="bilinear", align_corners=False)
        return self.net(x)                                 # [B, n_branches]

    def flops(self) -> float:
        r = self.in_res
        c1 = (r // 2) ** 2 * 16 * 3 * 9
        c2 = (r // 4) ** 2 * 32 * 16 * 9
        return 2.0 * (c1 + c2)                              # ~0.01 GFLOPs, negligible vs stem


def branch_flops(stem: TokenBudgetCLIPStem, budgets=KEEP_BUDGETS) -> list[float]:
    return [stem.stem_flops(k) for k in budgets]


# --------------------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import sys

    print("=== Phase-2 token-budget CLIP stem smoke test ===")
    if not _OPEN_CLIP:
        print("open_clip not available; cannot smoke-test."); sys.exit(1)

    dev = "cpu"
    stem = TokenBudgetCLIPStem().to(dev).eval()
    router = RGBComplexityEstimator(n_branches=len(KEEP_BUDGETS)).to(dev).eval()
    rgb = torch.randn(2, 3, 224, 224, device=dev)

    # 1) full budget reproduces the untouched CLIP forward (correctness anchor)
    import open_clip as _oc
    ref = _oc.create_model_and_transforms("ViT-B-32", pretrained="openai")[0].visual.to(dev).eval()
    with torch.no_grad():
        full = stem(rgb, keep_k=49)
        ref_out = ref(rgb)
    diff = (full - ref_out).abs().max().item()
    print(f"[1] keep_k=49 vs untouched ViT: max|diff|={diff:.2e}  ->  {'OK' if diff < 1e-4 else 'FAIL'}")

    # 2) each budget runs, correct output dim, different token counts
    print("[2] per-branch forward + FLOPs (MACs-as-FLOPs convention, 4.29 G full stem):")
    fl = branch_flops(stem)
    ok_dim = True
    for k, f in zip(KEEP_BUDGETS, fl):
        with torch.no_grad():
            out = stem(rgb, keep_k=k)
        idx = strided_keep_indices(stem.n_patches, k)
        ok_dim &= tuple(out.shape) == (2, 512)
        print(f"    keep_k={k:2d}  tokens={len(idx):2d}  out={tuple(out.shape)}  stem={f/1e9:5.2f} GFLOPs")

    # 3) branches genuinely differ in FLOPs (the whole point) + gap vs full
    gap = (fl[2] - fl[0]) / fl[2]
    balanced = sum(fl) / len(fl)
    print(f"[3] FLOPs spread: b0={fl[0]/1e9:.2f}  b1={fl[1]/1e9:.2f}  b2={fl[2]/1e9:.2f} G")
    print(f"    b0 is {100*gap:.0f}% below full; balanced-avg={balanced/1e9:.2f} G "
          f"({100*(1-balanced/fl[2]):.0f}% below full)")

    # 4) router is cheap + image-dependent (non-constant -> no E8 collapse)
    with torch.no_grad():
        logits = router(rgb)
        logits2 = router(torch.randn(2, 3, 224, 224, device=dev))
    varies = (logits - logits2).abs().max().item()
    print(f"[4] router logits shape={tuple(logits.shape)} flops={router.flops()/1e9:.4f} G  "
          f"image-dependent(max|d|)={varies:.3f}  ->  {'OK' if varies > 1e-3 else 'FAIL'}")

    passed = diff < 1e-4 and ok_dim and fl[0] < fl[1] < fl[2] and gap > 0.18 and varies > 1e-3
    print(f"\nSMOKE TEST: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
