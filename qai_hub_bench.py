"""Qualcomm AI Hub benchmark driver for TinyVLM.

Submits ONNX models to AI Hub for on-device profiling on Snapdragon devices,
polls until completion, downloads per-run latency CSV + summary JSON, and
optionally merges results into Table 5 numbers for tinyvlm_4.tex.

Auth: requires `qai_hub` Python package + a personal API token from
aihub.qualcomm.com (Settings -> API Token). Set:

    export QAI_HUB_API_TOKEN=<token>
    qai-hub configure --api_token "$QAI_HUB_API_TOKEN"

Usage:
    python qai_hub_bench.py export-onnx \
        --variant sttf_anc \
        --checkpoint <path>.pt --output models/onnx/cnn_sttf_anc_b0.onnx --branch 0
    python qai_hub_bench.py submit \
        --onnx models/onnx/cnn_sttf_anc_b0.onnx \
        --device "Samsung Galaxy S21 (Family)" \
        --label cnn_sttf_anc_b0_s21
    python qai_hub_bench.py poll --label cnn_sttf_anc_b0_s21
    python qai_hub_bench.py aggregate --output qai_hub_results/table5_measured.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path("qai_hub_results")
JOBS_FILE = RESULTS_DIR / "jobs.json"


def load_jobs():
    if JOBS_FILE.exists():
        return json.loads(JOBS_FILE.read_text())
    return {}


def save_jobs(jobs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))


class _ANCBranchWrapper(nn.Module):
    """Wraps AdaptiveNeuralCompression to expose a single-branch forward path.

    Skips Gumbel-softmax router (non-exportable due to randomness) and runs
    encoder[branch] -> projection[branch] -> decoder only. Returns just
    token_logits (single-output) for clean AI Hub profiling.
    """

    def __init__(self, anc_model, branch: int):
        super().__init__()
        self.encoder = anc_model.encoders[branch]
        self.projection = anc_model.projections[branch]
        self.decoder = anc_model.decoder

    def forward(self, rgb: torch.Tensor, events: torch.Tensor, text: torch.Tensor):
        feat = self.encoder(rgb, events)
        encoded = self.projection(feat)
        return self.decoder(encoded, text)


class _TokenLearnerWrapper(nn.Module):
    """Wraps TokenLearnerBaseline for single-output ONNX export."""

    def __init__(self, tl_model):
        super().__init__()
        self.inner = tl_model

    def forward(self, rgb: torch.Tensor, events: torch.Tensor, text: torch.Tensor):
        spatial = self.inner.encoder(rgb, events)
        tokens = self.inner.token_learner(spatial)
        tokens = self.inner.proj(tokens)
        encoded = tokens.mean(dim=1)
        return self.inner.decoder(encoded, text)


class _CLIPBranchWrapper(nn.Module):
    """Single-branch CLIP+ANC path: frozen ViT-B/32 -> MLP head[k] -> proj[k] -> decoder.

    Latency is weight-value-independent for ONNX profiling, so random head/decoder
    initialisation is acceptable.
    """

    def __init__(self, anc_model, branch: int):
        super().__init__()
        self.clip_visual = anc_model.clip_stem.visual
        self.head = anc_model.encoders[branch]
        self.projection = anc_model.projections[branch]
        self.decoder = anc_model.decoder

    def forward(self, rgb: torch.Tensor, text: torch.Tensor):
        clip_feat = self.clip_visual(rgb)
        feat = self.head(clip_feat)
        encoded = self.projection(feat)
        return self.decoder(encoded, text)


def _build_model(variant: str, checkpoint_path: str, branch: int):
    """Construct model + load checkpoint state_dict, return profiling wrapper."""
    if variant == "clip_anc":
        # Use nested tinyvlm/ which has CLIP code (root copy lacks it).
        sys.path.insert(0, str(Path(__file__).parent / "tinyvlm"))
        try:
            from tinyvlm.tinyvlm_vast import AdaptiveNeuralCompression, Config  # type: ignore
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent / "tinyvlm"))
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_clip_tinyvlm_vast", str(Path(__file__).parent / "tinyvlm" / "tinyvlm_vast.py")
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            AdaptiveNeuralCompression = mod.AdaptiveNeuralCompression
            Config = mod.Config
        cfg = Config()
        cfg.clip_backbone = True
        model = AdaptiveNeuralCompression(cfg)
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=False)
        model.eval()
        return _CLIPBranchWrapper(model, branch), cfg

    from tinyvlm_vast import (
        AdaptiveNeuralCompression,
        TokenLearnerBaseline,
        Config,
    )

    cfg = Config()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    if variant == "tokenlearner":
        model = TokenLearnerBaseline(cfg, num_tokens=cfg.tokenlearner_num_tokens
                                     if hasattr(cfg, "tokenlearner_num_tokens") else 8)
        model.load_state_dict(state, strict=False)
        model.eval()
        return _TokenLearnerWrapper(model), cfg

    model = AdaptiveNeuralCompression(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    return _ANCBranchWrapper(model, branch), cfg


def cmd_export_onnx(args):
    """Export a single-branch (or full TokenLearner) graph to ONNX opset 17."""
    if args.variant == "clip_anc":
        # Disable fused MHA (not exportable at opset 17)
        try:
            torch.backends.mha.set_fastpath_enabled(False)
        except Exception:
            pass
    wrapper, cfg = _build_model(args.variant, args.checkpoint, args.branch)

    dummy_rgb = torch.randn(1, 3, 224, 224)
    dummy_text = torch.randint(1, cfg.vocab_size, (1, cfg.max_length))
    dummy_events = torch.zeros(1, 2, 56, 56)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.variant == "clip_anc":
        torch.onnx.export(
            wrapper,
            (dummy_rgb, dummy_text),
            str(out_path),
            opset_version=17,
            input_names=["rgb", "text"],
            output_names=["token_logits"],
            dynamo=False,
        )
    else:
        torch.onnx.export(
            wrapper,
            (dummy_rgb, dummy_events, dummy_text),
            str(out_path),
            opset_version=17,
            input_names=["rgb", "events", "text"],
            output_names=["token_logits"],
            dynamo=False,
        )
    import onnx
    onnx.checker.check_model(str(out_path))
    print(f"exported: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def cmd_submit(args):
    import qai_hub as hub

    model = hub.upload_model(args.onnx)
    job = hub.submit_profile_job(
        model=model,
        device=hub.Device(args.device),
        name=args.label,
    )
    jobs = load_jobs()
    jobs[args.label] = {
        "job_id": job.job_id,
        "device": args.device,
        "onnx": args.onnx,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "submitted",
    }
    save_jobs(jobs)
    print(f"submitted: {args.label} -> {job.job_id} on {args.device}")


def cmd_poll(args):
    import qai_hub as hub

    jobs = load_jobs()
    label = args.label
    if label not in jobs:
        sys.exit(f"unknown label: {label}")
    job = hub.get_job(jobs[label]["job_id"])
    status = job.get_status()
    sym = getattr(status, "symbol", str(status))
    msg = getattr(status, "message", "") or ""
    jobs[label]["status"] = sym
    save_jobs(jobs)
    print(f"{label}: {sym} ({msg})")
    if getattr(status, "success", False):
        try:
            profile = job.download_profile()
        except Exception as e:
            print(f"  download_profile failed: {e}")
            return
        out = RESULTS_DIR / f"{label}_profile.json"
        out.write_text(json.dumps(profile, indent=2, default=str))
        exec_summary = profile.get("execution_summary", {}) if isinstance(profile, dict) else {}
        ms_us = exec_summary.get("estimated_inference_time", None)
        peak_mem = exec_summary.get("estimated_inference_peak_memory", None)
        jobs[label].update({
            "latency_ms": ms_us / 1e3 if ms_us else None,
            "peak_mem_bytes": peak_mem,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        save_jobs(jobs)
        print(f"  latency: {jobs[label]['latency_ms']} ms, peak_mem: {peak_mem}")


def cmd_aggregate(args):
    """Merge per-branch + per-device latencies into Table 5 numbers."""
    jobs = load_jobs()
    by_device = {}
    for label, j in jobs.items():
        if j.get("latency_ms") is None:
            continue
        dev = j["device"]
        by_device.setdefault(dev, {})[label] = {
            "latency_ms": j["latency_ms"],
            "peak_mem_bytes": j.get("peak_mem_bytes"),
            "job_id": j.get("job_id"),
        }

    routing = {"b0": 0.31, "b1": 0.34, "b2": 0.35}
    derived = {}
    for dev, runs in by_device.items():
        weighted = 0.0
        ok = True
        for b, w in routing.items():
            key = next((k for k in runs if f"_{b}_" in k or k.endswith(f"_{b}")), None)
            if key is None:
                ok = False
                break
            weighted += w * runs[key]["latency_ms"]
        derived[dev] = {
            "sttf_anc_routing_weighted_ms": weighted if ok else None,
            "raw_runs": runs,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(derived, indent=2))
    print(f"wrote: {out_path}")
    for dev, d in derived.items():
        w = d["sttf_anc_routing_weighted_ms"]
        print(f"\n{dev}: STTF+ANC weighted = {w:.2f} ms" if w else f"\n{dev}: incomplete")
        for k, v in d["raw_runs"].items():
            print(f"  {k}: {v['latency_ms']:.2f} ms (job {v['job_id']})")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export-onnx")
    e.add_argument("--variant", choices=["sttf_anc", "dense", "tokenlearner", "clip_anc"], required=True)
    e.add_argument("--checkpoint", default="")
    e.add_argument("--output", required=True)
    e.add_argument("--branch", type=int, default=2,
                   help="branch idx for ANC variants (0=Tiny, 1=Small, 2=Medium); ignored for tokenlearner")
    e.set_defaults(fn=cmd_export_onnx)

    s = sub.add_parser("submit")
    s.add_argument("--onnx", required=True)
    s.add_argument("--device", required=True)
    s.add_argument("--label", required=True)
    s.set_defaults(fn=cmd_submit)

    pl = sub.add_parser("poll")
    pl.add_argument("--label", required=True)
    pl.set_defaults(fn=cmd_poll)

    a = sub.add_parser("aggregate")
    a.add_argument("--output", default="qai_hub_results/table5_measured.json")
    a.set_defaults(fn=cmd_aggregate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
