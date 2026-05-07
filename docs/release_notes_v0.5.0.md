# v0.5.0 Release Notes

## Summary

PathVLM-LiteBench v0.5.0 packages the existing patch-level tools into a clearer public benchmark workflow. The release focuses on reproducible local runs: prepare a public patch dataset manifest, sample it, run zero-shot experiments, run prompt grids, validate configs, and compare saved reports.

This release does not add model training, bundled datasets, bundled weights, or clinical decision support. It remains a research and educational toolkit for frozen model evaluation.

## What Is Included

Public patch benchmark workflow:

- added `docs/public_patch_benchmark_workflow.md`
- documented a local MHIST-style workflow from annotation conversion to saved report comparison
- documented naming conventions for local `outputs/` runs
- clarified how to interpret accuracy, balanced accuracy, macro-F1, predicted-label counts, confusion matrices, and prompt-grid warnings without making clinical claims

Prompt-grid workflow documentation:

- added `docs/prompt_grid_workflow.md`
- documented prompt-grid config fields, dry-run behavior, output layout, report comparison, and optional CONCH access requirements
- linked the workflow from README and the v0.5.0 plan

Config validation:

- added `pathvlm-litebench validate-config`
- supports retrieval, zero-shot, prompt sensitivity, and zero-shot prompt-grid config validation
- validates committed example configs without loading datasets or model weights
- keeps CLI import lightweight by avoiding heavy model imports for validation-only commands

Offline test coverage:

- added tests for committed example configs
- added zero-shot prompt-grid dry-run validation for the sample MHIST config
- kept CI offline and lightweight

## What Remains Unchanged

The toolkit remains focused on:

- frozen CLIP-style model inference
- patch-level images
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- local CSV, JSON, Markdown, and HTML artifacts
- offline CI tests

Existing workflows remain available:

- CLIP, PLIP, and optional CONCH model wrappers
- retrieval, zero-shot classification, prompt sensitivity, and zero-shot prompt-grid demos
- manifest conversion and balanced sampling
- summary and comparison Markdown reports
- embedding cache and visualization utilities

## What Is Intentionally Not Included

v0.5.0 does not include:

- downloaded or bundled public datasets
- bundled model weights
- generated reports or predictions committed to the repository
- automatic Hugging Face token handling
- CONCH as a required dependency
- model downloads in CI
- clinical diagnosis or clinical decision support
- WSI-scale high-throughput processing
- model training or fine-tuning

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, prompt-grid reports, prediction files, or generated comparison files.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli validate-config configs/zero_shot_prompt_grid_mhist_sample.json
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --dry-run
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
```

Optional local benchmark run:

```bash
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json
```

The optional run may download or load model weights and requires local dataset paths. Optional gated models such as CONCH still require Hugging Face access and the optional CONCH package.

## Release Highlights

PathVLM-LiteBench v0.5.0 improves the path from toy demos to local public-dataset benchmarking:

- users have a documented patch-dataset benchmark recipe
- prompt-grid usage is documented beyond a README snippet
- example configs are validated without model downloads
- release docs keep local model behavior separate from clinical claims
- CI remains offline, lightweight, and suitable for small contributors
