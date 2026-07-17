# GPU Work Order — Mark (Hellbender / SLURM)

**Written 2026-07-16 by Max.** Supersedes the earlier vast.ai version of this work order —
you're running on Hellbender, so there is **no billing and no teardown discipline to worry about**.
Ignore any instruction you've seen from me about `vastai destroy`; it does not apply.

Three experiments, ranked. **All three are implemented and smoke-tested — you should not have to write
model code.** E21 is the one that matters most; run it first.

| | Cost | Status |
|---|---|---|
| **E21** attention selection | 3 runs, ~2–3 h each | ✅ `sbatch hellbender/e21_attn.slurm` |
| **E22** oracle bound | eval only, ~1 h | ✅ `sbatch hellbender/e22_oracle.slurm` (needs E21/phase-2 checkpoints to exist first) |
| **E20** single-branch sweep | 9 runs | ✅ `sbatch hellbender/e20_single_branch.slurm` |

---

## Setup (once)

```bash
cd $SCRATCH
git clone https://github.com/mahadkhaliq/tinyvlm.git
cd tinyvlm
bash hellbender_setup.sh           # conda env `tinyvlm` + COCO -> $SCRATCH/coco
```

Everything is on **`main`** as of 2026-07-16 — the phase-2 code used to live outside version
control, so an older clone will not have it. If you already have a checkout, `git pull` first.

**The code you need is at `vast/phase2/`** (repo root). Despite the directory name it is not
vast-specific — it is just the training script. Submit all jobs from the repo root; the SLURM
scripts `cd` into `vast/phase2/` themselves.

Sanity check before you submit anything:

```bash
cd $SCRATCH/tinyvlm/vast/phase2 && python clip_token_budget.py   # prints bit-identity diff 0.0
cd $SCRATCH/tinyvlm                                              # submit from the root
```

---

## Priority 1 — E21: attention-ranked token selection ⭐

**~3 runs × ~2–3 h = one afternoon.** This decides whether the paper's central claim is true.

The paper says dropping visual tokens from the frozen ViT costs ~5 CIDEr that no routing recovers
(condition C4). **But our cheap branches pick tokens by a deterministic even stride** — the least
informed rule that exists. ToMe/EViT/DynamicViT exist precisely because *informed* selection drops
40–50% of tokens at near-zero quality cost. So a reviewer's one-line rebuttal is: *"that's the cost of
stride sampling, not of cost differentiation."* Right now we cannot answer it, and the paper has to
scope its main claim down to "static stride selection" as a result.

I implemented the selector for you (`--token_select attn`, EViT-style: block 0 runs at full width and
supplies the CLS-attention ranking, then blocks 1–11 run on the top-k). It is smoke-tested and
bit-identical to the untouched ViT at full budget.

```bash
sbatch hellbender/e21_attn.slurm        # job array, seeds 42/43/44
```

**Read the result like this — decide now, before you look:**

| E21 outcome | Meaning |
|---|---|
| attn also loses ~5 CIDEr (lands ~75) | **C4 is structural.** The claim hardens from "stride selection" to genuinely "cost differentiation on a frozen stem." This is the outcome that finishes the paper. |
| attn recovers most of the gap (lands ~78–80) | **Our diagnosis is wrong.** Informed selection escapes the wall, and the paper needs reframing. Better we find this than a referee. |

**You cannot get a bad result here.** Both outcomes are publishable and I want whichever one is true.

⚠️ **Two things about `attn` you must not miss:**

1. **It is NOT cost-matched to stride.** Block 0 runs at full width to produce the ranking, so cost is
   `1×full + 11×(k+1)` instead of `12×(k+1)`: **2.31 / 3.26 / 4.29 G** vs stride's **2.13 / 3.17 /
   4.29 G** (+8.4% at k=24, +3.0% at k=36). `stem_flops()` already returns the right number per rule —
   just don't hand-copy stride's figures into an attn result.
2. **Pass `--no_onnx`.** Attn does a data-dependent gather, so ONNX export cannot succeed. It's caught
   and non-fatal, but without the flag you get an alarming traceback that looks like a crash. The
   provided SLURM script already passes it.

---

## Priority 2 — E22: oracle routing bound

**Eval-only, no training, ~1 h.** `sbatch hellbender/e22_oracle.slurm` (set `RUN_ROOT` to the trained
token-budget runs you want bounded; defaults to the pre-registered λ=0.01 run).

The paper claims "no routing policy recovers" the 5 CIDEr — currently supported by exactly three
learned λ values. An oracle bounds *every* policy: evaluate all three existing per-branch checkpoints
on Karpathy-test and take the best branch per image.

- Oracle ≈ 75 → "no policy recovers" becomes airtight and policy-independent.
- Oracle ≈ 79+ → real routing headroom exists and our claim is too strong.

**Report it as an upper bound and never as an operating point.** It picks the per-image winner using
the test metric itself, which no deployable router can do — that optimism is precisely the point: if
even the oracle can't clear the wall, nothing can. The JSON writes this caveat into an
`interpretation` field so it travels with the number.

Implementation note: the oracle pins every input to one branch via a `_force_branch` override on the
model. That override **only applies on the hard-argmax path** — if the model isn't in `eval()` with
`eval_routing_mode=hard`, soft routing would silently mix all branches and the "pinned" result would be
a plausible-looking lie. I made that case raise a `RuntimeError` rather than no-op, so you cannot hit
it silently. Checkpoints are indexed in `RUNS.md`.

---

## Priority 3 — E20: single-branch budget sweep

**9 runs (3 budgets × 3 seeds), ~1 day of queue.**

Separates *token-dropping* cost from the fixed cost of the routed 3-branch setup. The paper currently
admits it cannot attribute the ~5 CIDEr and says so in print; this resolves it and yields a
budget–quality *curve*, which is the citable artifact.

Why it matters: at λ=0.1, **45% of inputs already route to the 49-token branch — the bit-identical
untouched ViT** — yet the model still scores 75.16, not the ~77 a pure token-dropping story predicts.
So the penalty may be a fixed cost of the routed configuration (three heads each trained on ~1/3 of
the data + router interference), paid before any token is dropped.

```bash
sbatch hellbender/e20_single_branch.slurm    # 3 budgets x 3 seeds job array
```

Read: if single-branch **keep_k=49** (drops nothing, no router) lands ≈79.8, the architecture is sound
and E20-b vs the routed λ=0.01 run isolates the rest. If it lands ≈75, the finding *upgrades* — the
routing apparatus itself costs ~5 CIDEr. You win either way.

Implementation notes: `--single_branch_keep_k K` keeps exactly the branch that owned budget K inside
the routed model — same stem, same head dim (24→128, 36→256, 49→384), same decoder — and drops the
router entirely. That's what makes it a clean attribution rather than a different model. The script
passes `--lambda_balance 0 --lambda_entropy 0`: both terms are inert with a single expert (balance is
exactly 1.0, entropy is 0), so leaving them on would just add noise to the loss. The routing audit
reports `single_branch: true` and marks A3/A4 **not applicable** rather than fabricating a routing
distribution for a model that has no router.

---

## Optional freebie — E23: token-budget branches on AI Hub

$0, not a Hellbender job. Our one realised saving (~19% encoder FLOPs) never became a latency number —
the on-device table profiles the *old iso-cost* CLIP heads, not the token-budget branches the argument
rests on. `--token_select stride` branches ARE exportable (static indices), so this works.
See `qai_hub_bench.py export-onnx / submit / poll`. Field gotcha: latency =
`estimated_inference_time` (µs), memory = `estimated_inference_peak_memory` (bytes) — **not**
`peak_memory_usage`.

---

## ⚠️ FLOPs anchor was wrong — read before quoting any FLOPs number

Fixed 2026-07-16. `FULL_STEM_FLOPS` was `17.6e9`, which is the canonical **ViT-B/16** figure; our model
is **ViT-B/32**. Every absolute FLOPs number was ~4.1× inflated. Corrected to `4.29e9` in
`vast/phase2/clip_token_budget.py` and `vast/phase2/tinyvlm_vast.py`.

- Correct stride costs: **2.13 / 3.17 / 4.29 G** (was 8.75 / 12.99 / 17.60).
- **All relative savings unchanged** — the constant cancels in every ratio, so no conclusion moved.
  That is exactly why it hid for months.
- **Do not "verify" with fvcore.** It doesn't trace `nn.MultiheadAttention` and silently omits the whole
  attention block (reports 2.95 G). It looks authoritative and is wrong.
- Old result JSONs deliberately keep the old anchor — see `tnnls_results/README_FLOPS_ANCHOR.md`.

If you pull an old checkpoint or script, check which anchor it used before reporting anything.

---

## Protocol — non-negotiable or the comparison is void

- **Split:** Karpathy, image-disjoint, leak-free. Select on val, report on **test**.
- **Decoder:** `ConditionalTransformer` (NOT `--gpt2_decoder`) — must match `E_karpathy_clean`.
- **Backbone:** frozen CLIP ViT-B/32 (`--clip_backbone`).
- **Seeds:** 42, 43, 44. Report mean ± std over all three.
- **No hyperparameter search.** One config per arm, three seeds, report as-is. If an arm
  underperforms, that IS the result — do not tune it up. This is the same commitment our
  pre-registration made and honored, and it is a large part of why the paper is credible.

## SLURM notes

- **8 h wall limit** on the `gpu` partition. A 15-epoch CLIP run is ~2–3 h, so one run fits comfortably.
- The script **checkpoints and auto-resumes** from `<output_dir>/<run>/checkpoints/latest.pt`, and traps
  SIGTERM to save on preemption. If a job dies, resubmitting the same command resumes — it does not
  start over. Use `--no_resume` only if you deliberately want a fresh start.
- Watch: `squeue -u $USER`, `tail -f tinyvlm_e21_*.out`. Cancel: `scancel <jobid>`.
- Data lives at `$SCRATCH/coco` (set `COCO_DIR` to override). Runs write to `$SCRATCH/tinyvlm_runs/`.

## Deliverables

Per experiment: `tnnls_results/E2x_<name>/aggregate.json` (per-seed CIDEr + BLEU4, mean ± std, realised
GFLOPs **for the rule you ran**, Welch tests vs dense 79.80 ± 1.51 and vs routed λ=0.01 74.93 ± 0.25),
plus `NOTES.md` with the job IDs and GPU-hours. Update `TNNLS_STATE.json`; add a `C-E2x-*` row to
`tnnls_claim_ledger.csv`.

**Do not edit the manuscript.** Hand off via `aggregate.json` — Max reads it directly.

Report whatever you get, including an unflattering result. Every experiment above is designed so that
both outcomes are publishable.
