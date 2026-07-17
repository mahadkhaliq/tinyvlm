"""
Reference routing model for the Phase-2 token-budget experiment.

This mirrors EXACTLY the change that integrate.md applies to
AdaptiveNeuralCompression in tinyvlm_vast.py: a token-budget CLIP stem where
each branch runs a different visual-token budget, routed by a cheap RGB signal.
It exists to prove the full soft/hard routing path (forward, backward, per-branch
comp_cost) works end-to-end on random data -- no COCO / GPU needed -- so the wiring
into the real training script is low-risk. Run: `python phase2_model_ref.py`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from clip_token_budget import (
    TokenBudgetCLIPStem,
    RGBComplexityEstimator,
    KEEP_BUDGETS,
)


class TokenBudgetANC(nn.Module):
    """Frozen token-budget CLIP stem + RGB router + 3 heterogeneous heads.

    Routing semantics match the shipped model: soft weighted-sum over all branches
    at train time (gradients flow through the router); hard argmax at inference
    (only the selected branch's budgeted stem runs -> realised FLOPs = that branch).
    The ONLY difference from the shipped CLIP-ANC is that each branch calls the stem
    with its own keep_k, so per-branch stem FLOPs genuinely differ.
    """

    def __init__(self, hidden_dim: int = 256, budgets=KEEP_BUDGETS, temperature: float = 0.5) -> None:
        super().__init__()
        self.budgets = list(budgets)
        self.temperature = temperature
        self.stem = TokenBudgetCLIPStem()
        clip_dim = TokenBudgetCLIPStem.OUT_DIM
        head_dims = [128, 256, 384]
        self.encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(clip_dim, d), nn.GELU()) for d in head_dims]
        )
        self.projections = nn.ModuleList([nn.Linear(d, hidden_dim) for d in head_dims])
        self.complexity_estimator = RGBComplexityEstimator(n_branches=len(self.budgets))
        # per-branch cost = budgeted stem FLOPs + (small) head FLOPs
        self._stem_flops = [self.stem.stem_flops(k) for k in self.budgets]
        self._head_flops = [2.0 * d * clip_dim for d in head_dims]

    def _encoder_flops(self, k: int) -> float:
        return self._stem_flops[k] + self._head_flops[k]

    def _route(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            return torch.softmax((logits + g) / max(self.temperature, 1e-6), dim=-1)
        return torch.softmax(logits, dim=-1)

    def forward(self, rgb: torch.Tensor, eval_hard: bool = False):
        B = rgb.size(0)
        router_logits = self.complexity_estimator(rgb)          # cheap, image-dependent
        weights = self._route(router_logits)
        hidden = self.projections[0].out_features
        encoded = torch.zeros(B, hidden, device=rgb.device)
        comp_cost = torch.zeros(B, device=rgb.device)

        use_soft = self.training or not eval_hard
        if use_soft:
            for i, (enc, proj, k) in enumerate(zip(self.encoders, self.projections, self.budgets)):
                feat = enc(self.stem(rgb, keep_k=k))            # branch-specific budgeted stem
                w = weights[:, i : i + 1]
                encoded = encoded + w * proj(feat)
                comp_cost = comp_cost + w.squeeze(-1) * self._encoder_flops(i)
        else:
            branch_idx = weights.argmax(dim=-1)
            for i, (enc, proj, k) in enumerate(zip(self.encoders, self.projections, self.budgets)):
                mask = branch_idx == i
                if not mask.any():
                    continue
                feat = enc(self.stem(rgb[mask], keep_k=k))      # only selected branch's stem runs
                encoded[mask] = proj(feat)
                comp_cost[mask] = self._encoder_flops(i)
        return encoded, comp_cost, router_logits, weights


if __name__ == "__main__":
    import sys

    print("=== Phase-2 end-to-end routing smoke test ===")
    torch.manual_seed(0)
    model = TokenBudgetANC(hidden_dim=256)
    rgb = torch.randn(4, 3, 224, 224)

    # train-mode soft routing: forward + backward + gradients on trainable params only
    model.train()
    enc, comp, logits, w = model(rgb)
    loss = enc.pow(2).mean() + 1e-11 * comp.mean()
    loss.backward()
    stem_grad = any(p.grad is not None for p in model.stem.parameters())
    head_grad = all(p.grad is not None for p in model.encoders.parameters())
    router_grad = all(p.grad is not None for p in model.complexity_estimator.parameters())
    print(f"[train/soft] encoded={tuple(enc.shape)} comp_cost(mean)={comp.mean()/1e9:.2f}G "
          f"weights_sum={w.sum(-1).mean():.3f}")
    print(f"    grads: stem_frozen={not stem_grad} heads={head_grad} router={router_grad}")

    # eval hard routing: realised comp_cost must equal one of the discrete branch costs
    model.eval()
    with torch.no_grad():
        enc2, comp2, _, w2 = model(rgb, eval_hard=True)
    branch_costs = [model._encoder_flops(i) for i in range(3)]
    # float32 stores ~1e10 FLOPs with ~1e3 abs error -> compare with a relative tolerance
    all_discrete = all(any(abs(c.item() - bc) <= 1e-3 * bc for bc in branch_costs) for c in comp2)
    util = torch.bincount(model.complexity_estimator(rgb).argmax(-1), minlength=3).tolist()
    print(f"[eval/hard] per-branch stem FLOPs (G): {[round(bc/1e9,2) for bc in branch_costs]}")
    print(f"    realised comp_cost is one-of-branch: {all_discrete}  util(this batch)={util}")

    passed = (
        tuple(enc.shape) == (4, 256)
        and not stem_grad          # frozen CLIP stem: no grads
        and head_grad and router_grad
        and abs(float(w.sum(-1).mean()) - 1.0) < 1e-4
        and branch_costs[0] < branch_costs[1] < branch_costs[2]
        and all_discrete
    )
    print(f"\nE2E SMOKE TEST: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)
