# v0.14.0 Release Notes

## Summary

PathVLM-LiteBench v0.14.0 is a statistical-credibility and maintainability release. It strengthens how results are reported and how the toolkit is structured, without changing the core patch-level evaluation workflows or the project's low-compute, laptop-friendly positioning.

The headline additions are 95% bootstrap confidence intervals on accuracy and a new `linear-probe` command for few-shot evaluation on frozen embeddings. Accuracy figures from `compare-models` and the linear probe now ship with percentile bootstrap confidence intervals, so users can judge whether a difference between models is real or within noise — especially important on the small, laptop-sized samples this toolkit targets. Result metadata now records the runtime environment (Python, torch, transformers, scikit-learn versions) for reproducibility. This release also resolves license, version, and dependency inconsistencies, and splits the monolithic `cli.py` into a structured `cli/` package.

This release does not add whole-slide image reading, slide tiling, tissue detection, bundled pathology data, bundled model weights, required CONCH access, Hugging Face token handling, clinical diagnosis, or clinical decision support. It remains a research and educational toolkit for patch-level evaluation and visualization.

## What Is Included

Bootstrap confidence intervals and environment metadata:

- adds percentile bootstrap confidence intervals for accuracy (default 95%, configurable via `--confidence`, `--bootstrap-resamples`, `--seed`)
- `compare-models` now prints and saves a confidence interval per model, and draws error bars on the comparison chart
- records the runtime environment (Python, torch, transformers, scikit-learn versions) in result metadata for reproducibility
- keeps all computation deterministic under a fixed seed

`linear-probe` CLI subcommand:

- adds `pathvlm-litebench linear-probe` to train a logistic-regression probe on frozen image embeddings and evaluate it on a held-out split
- reuses the zero-shot report format, writing `predictions.csv`, `errors.csv`, and `metrics.json`
- reports accuracy with a 95% bootstrap confidence interval, balanced accuracy, and macro F1
- records the probe configuration (`logistic_regression`, C, max_iter, normalize, seed) and the runtime environment in metadata
- tunable via `--C`, `--max-iter`, `--no-normalize`, `--confidence`, `--bootstrap-resamples`, and `--seed`
- supports `--dry-run` for inspecting the planned run without loading a model

License, version, and dependency consistency:

- the README license section now states MIT, matching the MIT LICENSE file and the packaging metadata
- the package version is single-sourced: `pyproject.toml` derives it dynamically from `pathvlm_litebench.version` instead of duplicating the string
- `requirements.txt` is trimmed to runtime dependencies only; development tooling (`pytest`, `jupyter`) moves to the existing `[dev]` extra, with `pyproject.toml` noted as the canonical dependency list

CLI package refactor:

- splits the single-file `cli.py` into a `cli/` package: `parser.py` for argument parsing, `app.py` for dispatch, and a `commands/` subpackage grouping handlers by domain (info, manifest, reports, heatmap, config, model evaluation)
- preserves the public imports `main`, `build_parser`, and `_apply_zero_shot_grid_overrides`
- preserves the `python -m pathvlm_litebench.cli` entry point and the `pathvlm-litebench` console script
- preserves the invariant that importing the CLI never loads `torch` or `transformers`, via lazy imports inside handlers

Continuous integration:

- the CI workflow now installs the package with its development extra (`pip install -e ".[dev]"`) so `pytest` is present after `requirements.txt` stopped shipping it

## What Remains Unchanged

The toolkit remains focused on:

- frozen CLIP-style model inference for retrieval, classification, and patch scoring workflows
- patch-level images
- coordinate-aware patch manifests
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- local CSV, JSON, Markdown, HTML, and PNG artifacts
- offline CI tests that avoid model downloads

Existing workflows remain available:

- CLIP, PLIP, and optional CONCH model wrappers
- patch-text retrieval
- zero-shot classification
- prompt sensitivity analysis
- zero-shot prompt-grid runs
- multi-model zero-shot comparison via `compare-models`
- manifest conversion, balanced sampling, and ImageFolder manifest building
- summary and comparison Markdown reports
- embedding cache and visualization utilities
- artifact-only `render-coordinate-heatmap`
- model-backed `score-coordinate-heatmap` and `score-coordinate-heatmap-prompt-set`
- artifact-only `compare-coordinate-heatmap-scores`
- synthetic patch-coordinate heatmap demo

CONCH remains optional and gated. v0.14.0 does not make CONCH a required dependency.

## What Is Intentionally Not Included

v0.14.0 does not include:

- whole-slide image file reading
- slide tiling or tissue detection
- stain normalization or slide pyramid rendering
- tumor boundary identification
- clinical diagnosis or clinical decision support
- model training or fine-tuning beyond a linear probe on frozen embeddings
- downloaded or bundled public datasets
- bundled model weights
- required model downloads in CI
- required CONCH access
- Hugging Face token handling
- generated heatmaps, score CSV files, metadata files, comparison files, reports, or prediction files committed to the repository
- real pathology images or whole-slide image files committed to the repository

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, generated patches, heatmaps, score CSV files, metadata JSON files, comparison artifacts, reports, prediction files, or run logs.

## Verification

Recommended verification before tagging on Windows:

```powershell
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m pathvlm_litebench.cli version
.venv\Scripts\python.exe -m pathvlm_litebench.cli models
.venv\Scripts\python.exe -m pathvlm_litebench.cli linear-probe --help
```

The inspection commands above do not download models. Model-based commands load weights only when an evaluation is actually run.

## Example Commands

Compare models with confidence intervals on a balanced sample:

```bash
pathvlm-litebench compare-models --manifest dataset/CRC_VAL_HE_100_sample_manifest.csv --models clip plip --class-names "adipose tissue" background debris lymphocytes mucus "smooth muscle" "normal colon mucosa" "cancer-associated stroma" "colorectal adenocarcinoma epithelium" --output-dir outputs/model_comparison
```

Train and evaluate a linear probe on frozen embeddings:

```bash
pathvlm-litebench linear-probe --manifest dataset/CRC_VAL_HE_100_sample_manifest.csv --model plip --output-dir outputs/linear_probe
```

## Release Highlights

PathVLM-LiteBench v0.14.0 makes results more trustworthy and the codebase easier to maintain:

- accuracy now ships with 95% bootstrap confidence intervals, so model differences can be judged against noise
- a new `linear-probe` command adds few-shot evaluation on frozen embeddings, alongside zero-shot
- result metadata records the runtime environment for reproducibility
- license, version, and dependency metadata are now consistent and single-sourced
- the CLI is reorganized into a structured `cli/` package without changing its behavior or import surface
- CI installs the development extra so the test suite runs reliably
