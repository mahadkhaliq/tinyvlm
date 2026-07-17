# Phase-2 integration patch — wiring token-budget routing into `tinyvlm_vast.py`

Applies the smoke-tested `clip_token_budget.py` into the training script behind a
`--clip_token_budget` flag. Target file: `tinyvlm_student_pkg/tinyvlm/tinyvlm_vast.py`
(identical to `tinyvlm/tinyvlm_vast.py`). **Do not edit Mark's file in place on the
shared tree** — copy the training script + `clip_token_budget.py` into the Phase-2 run
dir on vast and patch the copy (run_all.sh does this).

Six edit sites. Line numbers are approximate (from the 2026-07 student build); match on
the anchor strings. After applying, the mandatory gate is `python tinyvlm_vast.py
--smoke_test --clip_token_budget --clip_backbone --gpt2_decoder --epochs 1` — it must run
without error and log three distinct per-branch FLOPs.

Prereq: `clip_token_budget.py` sits next to `tinyvlm_vast.py` (so `from clip_token_budget
import ...` resolves), or add its dir to `PYTHONPATH`.

---

### 1. Config — add two fields (near L140 / L186, in the `@dataclass class Config`)

```python
    clip_token_budget: bool = False   # Phase-2: per-branch visual-token budget (real per-branch FLOPs)
    token_budgets: str = "24,36,49"   # CLS + K patch tokens per branch (49 = full ViT-B/32)
```

### 2. Top of file — import the module (after the existing open_clip import guard)

```python
try:
    from clip_token_budget import TokenBudgetCLIPStem, RGBComplexityEstimator
    _TOKEN_BUDGET = True
except Exception:
    _TOKEN_BUDGET = False
```

### 3. `AdaptiveNeuralCompression.__init__` — replace the `if cfg.clip_backbone:` block head (L707–733)

Insert a token-budget branch *before* the existing `clip_backbone` MLP-head block:

```python
        if cfg.clip_backbone and cfg.clip_token_budget:
            assert _TOKEN_BUDGET, "clip_token_budget.py not importable"
            budgets = [int(x) for x in cfg.token_budgets.split(",")]
            self.clip_stem = TokenBudgetCLIPStem(cfg.clip_model, cfg.clip_pretrained)
            self._budgets = budgets
            clip_dim = TokenBudgetCLIPStem.OUT_DIM
            head_dims = [128, 256, 384]
            self.encoders = nn.ModuleList(
                [nn.Sequential(nn.Linear(clip_dim, d), nn.GELU(), nn.Dropout(cfg.dropout)) for d in head_dims]
            )
            self.projections = nn.ModuleList([nn.Linear(d, cfg.hidden_dim) for d in head_dims])
            # cheap RGB router (image-dependent -> fixes the E8 zero-signal collapse)
            self.complexity_estimator = RGBComplexityEstimator(n_branches=len(head_dims))
            # per-branch cost = budgeted STEM FLOPs (differ!) + small head FLOPs
            self._head_flops = [
                self.clip_stem.stem_flops(k) + 2.0 * d * clip_dim
                for k, d in zip(budgets, head_dims)
            ]
        elif cfg.clip_backbone:
            # ... existing iso-FLOPs MLP-head block unchanged ...
```

Note: the existing block sets `self.clip_stem = CLIPStemEncoder(...)`. Keep it as the
`elif`. Also add `self._budgets = None` in the non-token-budget paths (or guard with
`getattr(self, "_budgets", None)` in forward).

### 4. `forward` — token-budget path (L786–820)

Replace the CLIP routing/encoding with a branch that (a) routes from RGB and (b) runs a
per-branch budgeted stem. Only the CLIP+token-budget case changes; CNN and iso-FLOPs CLIP
paths stay as-is.

```python
        budgets = getattr(self, "_budgets", None)
        if self.clip_stem is not None and budgets is not None:
            router_logits = self.complexity_estimator(rgb)      # cheap RGB signal
            weights = self.router(router_logits)
            B = rgb.size(0)
            encoded = torch.zeros(B, self.cfg.hidden_dim, device=rgb.device)
            comp_cost = torch.zeros(B, device=rgb.device)
            if self.training or self.cfg.eval_routing_mode == "soft":
                for i, (enc, proj, k) in enumerate(zip(self.encoders, self.projections, budgets)):
                    feat = enc(self.clip_stem(rgb, keep_k=k))   # branch-specific budgeted stem
                    w = weights[:, i:i+1]
                    encoded = encoded + w * proj(feat)
                    comp_cost = comp_cost + w.squeeze(-1) * self._encoder_flops(i)
            else:
                branch_idx = weights.argmax(dim=-1)
                for i, (enc, proj, k) in enumerate(zip(self.encoders, self.projections, budgets)):
                    mask = branch_idx == i
                    if not mask.any():
                        continue
                    feat = enc(self.clip_stem(rgb[mask], keep_k=k))  # only selected stem runs
                    encoded[mask] = proj(feat)
                    comp_cost[mask] = self._encoder_flops(i)
            # ... fall through to the existing prefix_adapter / decoder tail ...
```

Wire this in front of the existing `if self.clip_stem is not None:` (full-stem) block so
token-budget takes precedence, then let both converge on the shared decoder tail
(L822–827). `_encoder_flops` (L780) already returns `self._head_flops[k]`, which now holds
`stem_flops(k)+head` — no change needed there.

### 5. `greedy_decode_gpt2` — eval routing (L426–441)

The eval decoder recomputes routing; give it the same token-budget path:

```python
    budgets = getattr(raw, "_budgets", None)
    if raw.clip_stem is not None and budgets is not None:
        router_logits = raw.complexity_estimator(rgb)           # RGB signal
        weights = F.softmax(router_logits, dim=-1)
        branch_idx = weights.argmax(dim=-1)
        encoded = torch.zeros(B, raw.cfg.hidden_dim, device=device)
        for i, (enc, proj, k) in enumerate(zip(raw.encoders, raw.projections, budgets)):
            mask = branch_idx == i
            if not mask.any():
                continue
            feat = enc(raw.clip_stem(rgb[mask], keep_k=k))
            encoded[mask] = proj(feat)
    elif raw.clip_stem is not None:
        # ... existing full-stem eval path unchanged ...
```

Also log `branch_idx.bincount(minlength=3)` here (or in the eval loop) — the per-branch
inference utilisation is the A3 acceptance number.

### 6. `build_parser` (L2450+) and Config-from-args (L2554)

```python
    p.add_argument("--clip_token_budget", action="store_true",
                   help="Phase-2: per-branch visual-token budget (real per-branch FLOPs on frozen CLIP)")
    p.add_argument("--token_budgets", type=str, default=default.token_budgets)
```
and in the `Config(...)` construction from args:
```python
        clip_token_budget=args.clip_token_budget,
        token_budgets=args.token_budgets,
```

---

## Instrumentation the analysis needs (add to the eval loop)

- **Per-branch inference utilisation** — `branch_idx` histogram over the whole Karpathy-test
  set (A3). Write to `summary.json` as `inference_branch_fraction`.
- **Realised routing-weighted FLOPs** — `sum(frac[i] * stem_flops(budget[i]))` using the
  *measured inference* fractions above, NOT the training fractions (A2). Write as
  `realised_encoder_gflops`.
- **Complexity–branch correlation** — Spearman(router_logit_margin, branch_idx) or the
  E8-style audit (A4). Reuse `tnnls_results/E8/tnnls_routing_audit.py` as the template.

These three plus CIDEr/BLEU on Karpathy-test are exactly the A1–A4 pre-registered metrics.
