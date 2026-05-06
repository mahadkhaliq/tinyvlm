# Jetson Nano Benchmark Instructions

Hi — these are step-by-step instructions for running the TinyVLM on-device latency benchmark on an NVIDIA Jetson Nano. Total wall-clock time on the device: ~30-45 minutes once setup is done.

You need:
- 1× Jetson Nano (4 GB) flashed with JetPack 4.6.1 or 4.6.4 (Ubuntu 18.04, CUDA 10.2, TensorRT 8.2)
- microSD ≥ 32 GB
- Wi-Fi or Ethernet
- 5V/4A power supply (do not use USB power; the GPU throttles or undervolts)
- This repository checked out on the Jetson (or just the files listed in step 3)

## 0. Performance mode

```bash
sudo nvpmodel -m 0       # MAX-N (use the 4A barrel-jack supply)
sudo jetson_clocks       # lock GPU/CPU clocks at max
```

Verify: `cat /etc/nvpmodel.conf | head -3` should show `< MAX-N >`. If your supply is USB-only, use `sudo nvpmodel -m 1` (5W mode) — note this in your results.

## 1. Install dependencies

The default Python on JetPack 4.6 is 3.6, which is too old for modern onnxruntime wheels. Install Python 3.8 + a Jetson-specific onnxruntime-gpu wheel:

```bash
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3-pip libopenblas-base libopenmpi-dev libomp-dev

python3.8 -m venv ~/tinyvlm-bench
source ~/tinyvlm-bench/bin/activate
pip install --upgrade pip wheel

# Jetson-specific onnxruntime-gpu (1.16.0 is the last with JetPack 4.6 support)
pip install numpy onnx
wget https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
```

Sanity check:

```bash
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

You should see `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`. If TensorRT is missing, the script falls back to CUDA — that is OK for this benchmark.

## 2. Copy benchmark files to the Jetson

From your laptop (Mac):

```bash
# Replace <jetson-ip> with the Jetson's IP (e.g. 192.168.1.42)
scp jetson_nano_bench.py jetson@<jetson-ip>:~/
scp -r models/onnx jetson@<jetson-ip>:~/onnx
```

Or use a USB stick — copy `jetson_nano_bench.py` and the entire `models/onnx/` folder.

## 3. Run the benchmark

On the Jetson, with the venv active:

```bash
cd ~
mkdir -p jetson_results

# CNN STTF+ANC: profile each branch separately
python jetson_nano_bench.py --onnx onnx/cnn_sttf_anc_b0.onnx --label cnn_sttf_anc_b0_jetson_nano
python jetson_nano_bench.py --onnx onnx/cnn_sttf_anc_b1.onnx --label cnn_sttf_anc_b1_jetson_nano
python jetson_nano_bench.py --onnx onnx/cnn_sttf_anc_b2.onnx --label cnn_sttf_anc_b2_jetson_nano

# CNN baselines
python jetson_nano_bench.py --onnx onnx/cnn_dense.onnx        --label cnn_dense_jetson_nano
python jetson_nano_bench.py --onnx onnx/cnn_tokenlearner.onnx --label cnn_tokenlearner_jetson_nano

# CLIP backbone (large; only run if RAM allows — see "Out of memory" below)
python jetson_nano_bench.py --onnx onnx/clip_anc_b0.onnx --label clip_anc_b0_jetson_nano
python jetson_nano_bench.py --onnx onnx/clip_anc_b1.onnx --label clip_anc_b1_jetson_nano
python jetson_nano_bench.py --onnx onnx/clip_anc_b2.onnx --label clip_anc_b2_jetson_nano
```

Each call takes ~1-3 minutes (most of it is the TRT engine build for the first call on a given ONNX; subsequent calls reuse the cache in `jetson_results/trt_cache/`).

The script defaults to `--iters 100 --warmup 20`. Increase to `--iters 200` for tighter standard deviation if you have time.

## 4. Collect outputs

When all runs are done:

```bash
ls jetson_results/
# expect: jobs.json + one <label>.json per run + trt_cache/

cat jetson_results/jobs.json
```

Send the entire `jetson_results/` folder back (zip + email, or scp back to the lab Mac):

```bash
tar czf jetson_results.tar.gz jetson_results
```

Or scp from Mac:

```bash
scp jetson@<jetson-ip>:~/jetson_results.tar.gz ./
```

## 5. Record run conditions

In the email / commit message, please include:

- JetPack version (`head -1 /etc/nv_tegra_release`)
- nvpmodel mode used (0 = MAX-N, 1 = 5W, 2 = 10W)
- Whether `jetson_clocks` was active
- Ambient temperature (rough — the Nano throttles above ~85 °C die temp)
- Whether the heatsink fan was running

These matter — Jetson Nano latency varies up to 2× depending on power mode and thermal state.

## Troubleshooting

**"Out of memory"** when loading a CLIP ONNX.
The Nano has only 4 GB RAM and the CLIP graph is ~389 MB on disk + activations. If TRT FP32 OOMs, the script's TRT options already enable FP16 (`trt_fp16_enable=True`) which roughly halves activation memory. If FP16 still OOMs, drop the CLIP rows and report only the CNN rows.

**"libcudart.so.10.2 not found"** when importing onnxruntime.
JetPack version mismatch. Confirm `dpkg -l | grep cuda-cudart` shows 10.2. If you flashed JetPack 5.x, you need a different onnxruntime wheel — ask the lab.

**TensorRT engine build hangs > 10 min.**
Normal for first run on a CLIP graph (it's a 12-layer ViT plus decoder, lots of fusion candidates). Let it finish; subsequent runs use the cache.

**`TensorrtExecutionProvider` missing from providers list.**
Means onnxruntime-gpu wheel was built without TRT support. Switch to the wheel link in step 1 — you may have grabbed a CUDA-only build.

**Latency variance > 50 % across runs.**
Power supply or thermal throttling. Check `tegrastats` in another terminal — look for `CPU@... C`, `GPU@... C` columns. If they exceed 80 °C, add a fan.

**Power numbers?**
The Jetson Nano dev board does not expose per-rail power telemetry over the standard interface. If you need power, the lab has an INA3221 breakout — but for this round we only need latency + peak resident memory.

## What we will use the numbers for

These measurements will become the third column of Table 5 in our NeurIPS 2026 submission. The headline comparison is STTF+ANC vs. the dense baseline (and vs. TokenLearner). We expect roughly:

- Dense MediumEnc.: ~120-180 ms
- STTF+ANC (routing-weighted across b0/b1/b2): ~60-90 ms
- Speedup: 1.5-2×

Your real numbers will replace those expectations. Don't tune them. Just run, send back, and we'll do the aggregation.

Thanks!
