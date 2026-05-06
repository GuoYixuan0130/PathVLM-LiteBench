# v0.3.1 Release Notes

## Summary

PathVLM-LiteBench v0.3.1 adds Markdown comparison summaries for multiple saved report directories.

This patch release builds on the v0.3.0 post-run reporting workflow. It helps users compare several runs from existing local artifacts without rerunning model inference, loading images, downloading model weights, or adding new model wrappers.

This release is intended for research and educational use only.

## What Is Included

Report comparison workflow:

- added `pathvlm-litebench compare-reports`
- added zero-shot comparison summaries
- added retrieval comparison summaries
- added prompt sensitivity comparison summaries
- added optional run labels through `--run_names`
- added artifact presence tables for compared report directories
- kept comparison generation local and artifact-only

Supported report directories:

- zero-shot directories containing `metrics.json`, optionally with `predictions.csv` and `errors.csv`
- retrieval directories containing `retrieval_metrics.json`, optionally with `retrieval_results.csv`
- prompt sensitivity directories containing `prompt_sensitivity_metrics.json`, optionally with `prompt_sensitivity_summary.csv` and `prompt_sensitivity_details.csv`

## CLI Usage

Zero-shot report comparison:

```bash
pathvlm-litebench compare-reports \
  --task zero-shot \
  --report_dirs outputs/zero_shot_clip outputs/zero_shot_plip \
  --run_names clip plip \
  --output outputs/zero_shot_comparison.md
```

Retrieval report comparison:

```bash
pathvlm-litebench compare-reports \
  --task retrieval \
  --report_dirs outputs/retrieval_clip outputs/retrieval_plip \
  --run_names clip plip \
  --output outputs/retrieval_comparison.md
```

Prompt sensitivity report comparison:

```bash
pathvlm-litebench compare-reports \
  --task prompt-sensitivity \
  --report_dirs outputs/prompt_sensitivity_clip outputs/prompt_sensitivity_plip \
  --run_names clip plip \
  --output outputs/prompt_sensitivity_comparison.md
```

## What Remains Unchanged

The toolkit remains focused on:

- frozen model inference
- patch-level images
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- lightweight tests and CI
- local CSV/JSON/Markdown reporting

Existing v0.3.0 summary workflows remain available through `pathvlm-litebench summarize-report`.

## What Is Intentionally Not Included

v0.3.1 does not include:

- dashboards or hosted tracking integrations
- MLflow, W&B, database, or server-backed logging support
- new model wrappers
- CONCH implementation
- full WSI high-throughput processing
- large-scale VLM pretraining
- clinical diagnosis or clinical decision support
- bundled public datasets
- real pathology images
- model weights
- generated output reports

Generated comparisons are local outputs. Do not commit comparisons if they contain dataset-specific paths, metrics, or other local experiment details.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
```

The lightweight CLI commands and CI tests do not download models by default.

## Release Highlights

PathVLM-LiteBench v0.3.1 improves post-run result synthesis:

- one CLI command for multi-run Markdown comparisons
- same-task comparisons for zero-shot, retrieval, and prompt sensitivity reports
- no model inference, image loading, or model downloads during comparison generation
- continued low-compute, patch-level, research-only positioning
