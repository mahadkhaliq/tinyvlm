# Repository Guidelines

## Project Structure & Module Organization

This workspace contains the TinyVLM NeurIPS submission, training scripts, benchmark tooling, and paper artifacts. Use `tinyvlm_vast.py` for production training work; `tinyvlm_sttf_colab.py` is the legacy smoke-test path, and `sttf.py` is a throwaway Colab dump. Paper sources live at the root (`tinyvlm_4.3.tex` is the active submission) with figures in `Figure/`. Qualcomm benchmark outputs are in `qai_hub_results/`; Jetson handoff files are in `jetson_nano/`. The nested `tinyvlm/` directory is a separate checkout with its own scripts and history; confirm the intended tree before editing files that exist in both places.

## Build, Test, and Development Commands

Install root dependencies manually; the pinned list is in `tinyvlm/requirements.txt`.

```bash
pip install torch torchvision transformers tensorboard pycocotools pillow tqdm scipy matplotlib onnx open_clip_torch pycocoevalcap
python -m nltk.downloader punkt punkt_tab
python tinyvlm_vast.py --smoke_test --epochs 2
python tinyvlm_vast.py --seeds 42,43,44 --epochs 10 ...
pdflatex tinyvlm_4.3.tex && bibtex tinyvlm_4.3 && pdflatex tinyvlm_4.3.tex && pdflatex tinyvlm_4.3.tex
python qai_hub_bench.py --help
python jetson_nano_bench.py --help
```

The smoke test validates the production training path without COCO. The LaTeX command rebuilds the active paper; verify the main text stays within the NeurIPS page limit.

## Coding Style & Naming Conventions

Use Python 3.10+ and standard 4-space indentation. Keep hyperparameters in the `Config` dataclass in `tinyvlm_vast.py`, then expose matching CLI flags. Prefer explicit JSON artifacts for benchmark provenance. Name result directories by purpose and preserve `.nosync` suffixes for large local runs excluded from iCloud.

## Testing Guidelines

There is no formal pytest suite. Before changing training code, run the smoke test and inspect generated `summary.json` / `metrics.jsonl`. For benchmark changes, preserve `qai_hub_results/jobs.json` schema and verify table aggregates against `qai_hub_results/table5_measured.json`. Do not invent latency, memory, FLOPs, CIDEr, or power values.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative summaries such as `add Qualcomm AI Hub on-device benchmark cycle` and `Phase CLIP done; state + findings updated`. Keep commits scoped to one phase or artifact set. PRs should describe the experiment or paper change, list commands run, cite job IDs or result files for measured numbers, and mention page-count impact for paper edits.

## Security & Configuration Tips

Never commit `*.pt`, checkpoints, `runs_*`, `.nosync/`, API tokens, or ONNX blobs unless explicitly intended. Keep AI Hub credentials in local config only, and redact token references in committed state files.
