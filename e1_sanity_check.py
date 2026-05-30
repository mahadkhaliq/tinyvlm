"""E1 Week-1 input-sanity check for DVS128 Gesture.

Two-stage check:
  Stage 1 (always runs): verify DVS128 event volumes have non-zero per-sample
    variance — the COCO failure root cause was std=0 (zero-tensor events).
  Stage 2 (requires --checkpoint): load CNN STTF+ANC model, run
    complexity_estimator on a batch, confirm routing histogram is not
    100%-one-branch.

Saves results to --output_dir (default: ./tnnls_results/E1_sanity/):
  event_stats.json   — per-sample std + summary statistics
  routing_hist.json  — per-branch fraction (Stage 2 only)
  event_vol_grid.png — mosaic of 16 event volumes (visual inspection)
  routing_hist.png   — bar chart of branch fractions (Stage 2 only)

Usage:
  # Stage 1 only (data check, no GPU/checkpoint needed)
  python e1_sanity_check.py --data_dir $SCRATCH/dvs128

  # Stage 1 + 2 (full check with CNN model)
  python e1_sanity_check.py \
      --data_dir $SCRATCH/dvs128 \
      --checkpoint $SCRATCH/tinyvlm_ckpts/cnn_seed42.pt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── event-volume helpers ──────────────────────────────────────────────────────

def events_to_volume(events, H: int = 56, W: int = 56) -> np.ndarray:
    """Accumulate structured event array into a (2, H, W) polarity volume.

    Channel 0 = positive events, channel 1 = negative events.
    Coordinates are scaled from the 128×128 DVS128 sensor to H×W.
    """
    vol = np.zeros((2, H, W), dtype=np.float32)
    if len(events) == 0:
        return vol
    xs = np.clip((events["x"].astype(np.float32) * W / 128).astype(int), 0, W - 1)
    ys = np.clip((events["y"].astype(np.float32) * H / 128).astype(int), 0, H - 1)
    ps = events["p"].astype(int)
    np.add.at(vol[0], (ys[ps == 1], xs[ps == 1]), 1.0)
    np.add.at(vol[1], (ys[ps == 0], xs[ps == 0]), 1.0)
    return vol


def load_dvs128(data_dir: str, n_samples: int = 200, train: bool = True):
    """Load DVS128 Gesture samples via tonic, return list of (volume, label)."""
    try:
        import tonic
        import tonic.transforms as transforms
    except ImportError:
        sys.exit("tonic not installed — run: pip install tonic")

    sensor_size = tonic.datasets.DVSGesture.sensor_size  # (128, 128, 2)

    dataset = tonic.datasets.DVSGesture(
        save_to=data_dir,
        train=train,
    )

    samples = []
    indices = np.random.default_rng(42).choice(
        len(dataset), size=min(n_samples, len(dataset)), replace=False
    )
    for idx in indices:
        events, label = dataset[int(idx)]
        vol = events_to_volume(events)
        samples.append((vol, label))
    return samples


# ── Stage 1: data variance check ─────────────────────────────────────────────

def stage1_data_check(samples, output_dir: Path):
    vols = np.stack([v for v, _ in samples])          # (N, 2, 56, 56)

    # per-sample std over all spatial/channel dimensions
    per_sample_std = vols.reshape(len(vols), -1).std(axis=1)  # (N,)

    stats = {
        "n_samples": len(samples),
        "per_sample_std_mean": float(per_sample_std.mean()),
        "per_sample_std_min":  float(per_sample_std.min()),
        "per_sample_std_max":  float(per_sample_std.max()),
        "fraction_nonzero_std": float((per_sample_std > 0).mean()),
        "verdict": "PASS" if per_sample_std.mean() > 0 else "FAIL_zero_events",
    }

    print("\n── Stage 1: event-volume variance ──────────────────────────")
    print(f"  samples checked : {stats['n_samples']}")
    print(f"  mean std        : {stats['per_sample_std_mean']:.4f}")
    print(f"  min std         : {stats['per_sample_std_min']:.4f}")
    print(f"  fraction std>0  : {stats['fraction_nonzero_std']:.3f}")
    print(f"  VERDICT         : {stats['verdict']}")

    (output_dir / "event_stats.json").write_text(json.dumps(stats, indent=2))

    # visual: mosaic of first 16 event volumes (pos - neg for display)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, (vol, lbl) in zip(axes.flat, samples[:16]):
        display = vol[0] - vol[1]   # positive minus negative events
        ax.imshow(display, cmap="RdBu_r", vmin=-display.std()*2, vmax=display.std()*2)
        ax.set_title(f"cls {lbl}", fontsize=7)
        ax.axis("off")
    fig.suptitle("DVS128 Gesture — event volumes (red=positive, blue=negative)")
    fig.tight_layout()
    fig.savefig(output_dir / "event_vol_grid.png", dpi=100)
    plt.close(fig)
    print(f"  saved: {output_dir / 'event_vol_grid.png'}")

    return stats


# ── Stage 2: routing-histogram check ─────────────────────────────────────────

def stage2_routing_check(samples, checkpoint_path: str, output_dir: Path):
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from tinyvlm_vast import AdaptiveNeuralCompression, Config
    except ImportError:
        sys.exit("tinyvlm_vast.py not found in current directory")

    print("\n── Stage 2: routing histogram ──────────────────────────────")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    cfg = Config()
    model = AdaptiveNeuralCompression(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  warning: {len(missing)} missing keys")
    model.eval().to(device)

    branch_counts = np.zeros(3, dtype=int)

    with torch.no_grad():
        for vol, _ in samples:
            rgb_t   = torch.zeros(1, 3, 224, 224, device=device)
            event_t = torch.tensor(vol[None], device=device)   # (1, 2, 56, 56)

            # extract complexity_estimator output and argmax routing
            rgb_feats = model.complexity_estimator(rgb_t, event_t) \
                if hasattr(model, "complexity_estimator") \
                else model._complexity(rgb_t, event_t)
            logits = model.router(rgb_feats) if hasattr(model, "router") \
                else _extract_router_logits(model, rgb_t, event_t)
            branch_idx = int(logits.argmax(dim=-1).squeeze())
            branch_counts[branch_idx] += 1

    fracs = branch_counts / branch_counts.sum()
    routing = {
        "branch_counts": branch_counts.tolist(),
        "branch_fractions": fracs.tolist(),
        "dominant_branch": int(branch_counts.argmax()),
        "verdict": "PASS_routing_varied" if fracs.max() < 0.95 else "WARN_routing_collapsed",
    }

    print(f"  branch fractions: {fracs.round(3).tolist()}")
    print(f"  dominant branch : {routing['dominant_branch']}")
    print(f"  VERDICT         : {routing['verdict']}")

    (output_dir / "routing_hist.json").write_text(json.dumps(routing, indent=2))

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Tiny (2G)", "Small (8G)", "Medium (20G)"], fracs,
           color=["#4CAF50", "#2196F3", "#F44336"])
    ax.set_ylabel("Fraction of samples routed")
    ax.set_ylim(0, 1.05)
    ax.set_title("DVS128 routing histogram (hard argmax, CNN STTF+ANC)")
    for i, (f, c) in enumerate(zip(fracs, branch_counts)):
        ax.text(i, f + 0.02, f"{f:.2f} (n={c})", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "routing_hist.png", dpi=100)
    plt.close(fig)
    print(f"  saved: {output_dir / 'routing_hist.png'}")

    return routing


def _extract_router_logits(model, rgb, events):
    """Fallback: hook complexity_estimator + GumbelSoftmaxRouter directly."""
    # Works for AdaptiveNeuralCompression where the router is a sub-module
    with torch.no_grad():
        # Try common attribute patterns
        for attr in ("gumbel_router", "anc_router", "branch_router"):
            if hasattr(model, attr):
                feat = model.complexity_estimator(rgb, events)
                return getattr(model, attr)(feat)
        # Last resort: run full forward in eval (hard routing), capture branch_idx
        # via a forward hook on the router
        logits_captured = {}
        hooks = []

        def _hook(module, inp, out):
            logits_captured["logits"] = out.detach()

        for name, module in model.named_modules():
            if "router" in name.lower() or "gumbel" in name.lower():
                hooks.append(module.register_forward_hook(_hook))
                break

        model(rgb, events, torch.zeros(1, 10, dtype=torch.long, device=rgb.device))
        for h in hooks:
            h.remove()

        if "logits" in logits_captured:
            return logits_captured["logits"]
        raise RuntimeError("Could not locate router module — inspect model architecture")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E1 DVS128 input-sanity check")
    parser.add_argument("--data_dir", required=True,
                        help="Directory for DVS128 Gesture data (tonic will download here)")
    parser.add_argument("--checkpoint", default=None,
                        help="CNN STTF+ANC checkpoint for Stage 2 routing check")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to check (default 200)")
    parser.add_argument("--output_dir", default="tnnls_results/E1_sanity",
                        help="Directory for output JSON and PNG files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.n_samples} DVS128 samples from {args.data_dir} ...")
    samples = load_dvs128(args.data_dir, n_samples=args.n_samples)
    print(f"Loaded {len(samples)} samples.")

    stage1_stats = stage1_data_check(samples, output_dir)

    if args.checkpoint:
        stage2_routing = stage2_routing_check(samples, args.checkpoint, output_dir)
    else:
        print("\n── Stage 2 skipped (no --checkpoint provided) ──────────────")
        print("  To run Stage 2: add --checkpoint <path/to/cnn_seed42.pt>")
        stage2_routing = None

    summary = {
        "stage1": stage1_stats,
        "stage2": stage2_routing,
        "go_nogo": (
            "GO"
            if stage1_stats["verdict"] == "PASS"
            and (stage2_routing is None or "PASS" in stage2_routing["verdict"])
            else "NO_GO"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*55}")
    print(f"  GO/NO-GO : {summary['go_nogo']}")
    print(f"  artifacts: {output_dir}/")
    print(f"{'='*55}")
    print("Send Max: event_stats.json + routing_hist.png + summary.json")


if __name__ == "__main__":
    main()
