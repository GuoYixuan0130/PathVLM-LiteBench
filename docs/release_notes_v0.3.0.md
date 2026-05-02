# v0.3.0 Release Notes

## Summary

PathVLM-LiteBench v0.3.0 adds Markdown experiment summaries for saved evaluation outputs.

This release focuses on result review after a run has already produced local report artifacts. The new summary utility reads existing CSV/JSON files and writes an `experiment_summary.md` file. It does not run model inference, load images, download model weights, or add new model wrappers.

This release is intended for research and educational use only.

## What Is Included

Report summary workflow:

- added `pathvlm-litebench summarize-report`
- added zero-shot report summaries
- added retrieval report summaries
- added prompt sensitivity report summaries
- added Markdown tables for run metadata, key metrics, artifact paths, and report row counts
- normalized Markdown numeric formatting for readability
- normalized generated summary paths to forward-slash style for GitHub-friendly Markdown

Supported report directories:

- zero-shot directories containing `metrics.json`, `predictions.csv`, and optionally `errors.csv`
- retrieval directories containing `retrieval_metrics.json` and optionally `retrieval_results.csv`
- prompt sensitivity directories containing `prompt_sensitivity_metrics.json`, optionally with `prompt_sensitivity_summary.csv` and `prompt_sensitivity_details.csv`

## CLI Usage

Zero-shot report summary:

```bash
pathvlm-litebench summarize-report \
  --task zero-shot \
  --report_dir outputs/zero_shot_demo
```

Retrieval report summary:

```bash
pathvlm-litebench summarize-report \
  --task retrieval \
  --report_dir outputs/retrieval_demo
```

Prompt sensitivity report summary:

```bash
pathvlm-litebench summarize-report \
  --task prompt-sensitivity \
  --report_dir outputs/prompt_sensitivity_demo
```

Each command writes `experiment_summary.md` in the report directory by default. Use `--output` to choose a different Markdown path.

## What Remains Unchanged

The toolkit remains focused on:

- frozen model inference
- patch-level images
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- lightweight tests and CI

Existing workflows remain available:

- CLIP and PLIP model wrappers
- CONCH placeholder status
- patch-text retrieval
- manifest-based retrieval evaluation with Recall@K
- zero-shot patch classification
- classification metrics beyond accuracy
- prediction distribution and collapse warnings
- prompt sensitivity analysis
- manifest conversion and sampling
- embedding cache
- visualization and HTML retrieval reports
- config-driven retrieval, zero-shot, and prompt sensitivity demos

## What Is Intentionally Not Included

v0.3.0 does not include:

- multi-run comparison tables
- dashboard or tracking-server integrations
- MLflow, W&B, database, or hosted logging support
- new model wrappers
- CONCH implementation
- full WSI high-throughput processing
- large-scale VLM pretraining
- clinical diagnosis or clinical decision support
- bundled public datasets
- real pathology images
- model weights
- generated output reports

Generated summaries are local outputs. Do not commit summaries if they contain dataset-specific paths, metrics, or other local experiment details.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli summarize-report --help
```

The lightweight CLI commands and CI tests do not download models by default.

Optional local summary checks:

```bash
pathvlm-litebench summarize-report --task zero-shot --report_dir outputs/zero_shot_demo
pathvlm-litebench summarize-report --task retrieval --report_dir outputs/retrieval_demo
pathvlm-litebench summarize-report --task prompt-sensitivity --report_dir outputs/prompt_sensitivity_demo
```

These commands require matching local report artifacts under `outputs/`.

## Release Highlights

PathVLM-LiteBench v0.3.0 improves the post-run reporting workflow:

- one CLI command for Markdown summaries
- support for zero-shot, retrieval, and prompt sensitivity reports
- readable metric and artifact summaries
- no new runtime service, database, or model dependency
- continued low-compute, patch-level, research-only positioning
