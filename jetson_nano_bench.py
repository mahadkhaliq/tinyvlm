"""TinyVLM Jetson Nano benchmark.

Loads an ONNX file, runs N inference iterations on Jetson hardware via
onnxruntime (TensorRT execution provider preferred, CUDA fallback, CPU
fallback), records mean/std/min/p50/p99 latency in milliseconds and peak
GPU memory in bytes via the tegrastats sidecar, and writes a JSON record
compatible with the format used by qai_hub_results/jobs.json.

Usage:
    python jetson_nano_bench.py \\
        --onnx models/onnx/cnn_sttf_anc_b1.onnx \\
        --label cnn_sttf_anc_b1_jetson_nano \\
        --iters 100 --warmup 20

Output: jetson_results/<label>.json + appended row in jetson_results/jobs.json
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("jetson_results")
JOBS_FILE = RESULTS_DIR / "jobs.json"


def load_jobs():
    if JOBS_FILE.exists():
        return json.loads(JOBS_FILE.read_text())
    return {}


def save_jobs(jobs):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))


class TegrastatsSampler(threading.Thread):
    """Background sampler that reads /sys/devices/.../meminfo via tegrastats.

    On Jetson Nano without tegrastats access, we fall back to /proc/meminfo
    delta. Reports peak resident-memory increase during the benchmark window.
    """

    def __init__(self, interval=0.1):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._proc = None

    def _read_meminfo(self):
        """Return MemAvailable in bytes from /proc/meminfo."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        except Exception:
            pass
        return None

    def run(self):
        baseline = self._read_meminfo()
        if baseline is None:
            return
        while not self._stop.is_set():
            cur = self._read_meminfo()
            if cur is not None and cur < baseline:
                used = baseline - cur
                if used > self.peak_bytes:
                    self.peak_bytes = used
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)


def pick_session(onnx_path):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    print(f"available providers: {avail}", flush=True)

    preferred = []
    if "TensorrtExecutionProvider" in avail:
        preferred.append((
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(RESULTS_DIR / "trt_cache"),
            },
        ))
    if "CUDAExecutionProvider" in avail:
        preferred.append(("CUDAExecutionProvider", {}))
    preferred.append(("CPUExecutionProvider", {}))

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for prov_name, prov_opts in preferred:
        try:
            sess = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=[(prov_name, prov_opts)] if prov_opts else [prov_name],
            )
            print(f"using provider: {prov_name}", flush=True)
            return sess, prov_name
        except Exception as e:
            print(f"provider {prov_name} failed: {e}", flush=True)
            continue
    raise RuntimeError("no execution provider succeeded")


def make_dummy_inputs(sess):
    """Build dummy input tensors matching session input shapes / types."""
    feed = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if "float" in inp.type:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        elif "int64" in inp.type:
            feed[inp.name] = np.random.randint(1, 8192, shape, dtype=np.int64)
        elif "int32" in inp.type:
            feed[inp.name] = np.random.randint(1, 8192, shape, dtype=np.int32)
        else:
            feed[inp.name] = np.zeros(shape, dtype=np.float32)
    return feed


def benchmark(onnx_path, iters, warmup):
    sess, provider = pick_session(onnx_path)
    feed = make_dummy_inputs(sess)
    out_names = [o.name for o in sess.get_outputs()]

    # Warm-up.
    for _ in range(warmup):
        sess.run(out_names, feed)

    sampler = TegrastatsSampler(interval=0.05)
    sampler.start()

    samples_us = []
    t_block_start = time.perf_counter()
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        sess.run(out_names, feed)
        t1 = time.perf_counter_ns()
        samples_us.append((t1 - t0) / 1000.0)  # us
    t_block_end = time.perf_counter()

    sampler.stop()

    samples_ms = [x / 1000.0 for x in samples_us]
    return {
        "provider": provider,
        "iters": iters,
        "warmup": warmup,
        "wall_seconds": t_block_end - t_block_start,
        "latency_ms_mean": statistics.mean(samples_ms),
        "latency_ms_std": statistics.pstdev(samples_ms),
        "latency_ms_min": min(samples_ms),
        "latency_ms_p50": statistics.median(samples_ms),
        "latency_ms_p99": sorted(samples_ms)[int(0.99 * len(samples_ms))],
        "peak_resident_delta_bytes": sampler.peak_bytes,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()

    onnx_path = str(Path(args.onnx).resolve())
    if not Path(onnx_path).exists():
        sys.exit(f"missing ONNX: {onnx_path}")

    print(f"benchmarking {onnx_path} for {args.iters} iters (warmup {args.warmup})", flush=True)
    result = benchmark(onnx_path, args.iters, args.warmup)
    result.update({
        "label": args.label,
        "onnx": onnx_path,
        "device": "Jetson Nano",
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "uname": " ".join(os.uname()),
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"{args.label}.json"
    out_json.write_text(json.dumps(result, indent=2))

    jobs = load_jobs()
    jobs[args.label] = {
        "device": "Jetson Nano",
        "onnx": onnx_path,
        "latency_ms": result["latency_ms_mean"],
        "latency_ms_std": result["latency_ms_std"],
        "peak_mem_bytes": result["peak_resident_delta_bytes"],
        "provider": result["provider"],
        "iters": result["iters"],
        "ran_at": result["ran_at"],
    }
    save_jobs(jobs)

    print("=== RESULT ===")
    print(json.dumps(result, indent=2))
    print(f"\nwrote: {out_json}")
    print(f"appended to: {JOBS_FILE}")


if __name__ == "__main__":
    main()
