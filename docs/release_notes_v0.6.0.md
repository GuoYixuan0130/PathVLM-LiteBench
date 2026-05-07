# v0.6.0 Release Notes

## Summary

PathVLM-LiteBench v0.6.0 turns the v0.5.0 public patch benchmark workflow into a more reproducible MHIST benchmark audit path. The release adds committed MHIST baseline configs, non-overwriting prompt-grid output overrides, and documented local audit results for CLIP, PLIP, and optional CONCH.

This release does not add model training, bundled datasets, bundled weights, or clinical decision support. It remains a research and educational toolkit for frozen patch-level model evaluation.

## What Is Included

Post-release validation:

- added `docs/v0.5.0_post_release_audit.md`
- verified the v0.5.0 GitHub Release, tag, README links, version command, model registry listing, demo listing, config validation, prompt-grid dry-run, and editable install entry point
- updated `.gitignore` to exclude Python build metadata from editable installs

MHIST sampled baseline workflow:

- added `docs/v0.6.0_mhist_reproducibility_audit.md`
- ran a local sampled MHIST CLIP baseline and PLIP baseline
- documented saved report generation, report summarization, and report comparison
- added committed zero-shot configs:
  - `configs/zero_shot_mhist_clip_sample.json`
  - `configs/zero_shot_mhist_plip_sample.json`
- added tests that validate the committed MHIST baseline configs without loading datasets or models

Prompt-grid workflow:

- added `--output-root` and `--comparison-output` overrides to `pathvlm-litebench run-zero-shot-grid`
- documented non-overwriting prompt-grid runs
- added `docs/v0.6.0_mhist_prompt_grid_audit.md`
- ran a local sampled MHIST CLIP/PLIP/CONCH prompt-grid audit
- confirmed that per-run reports, logs, and comparison summaries are generated under an overridden output root

Documentation polish:

- removed internal follow-up sections from public observation and audit docs
- kept public docs focused on setup, commands, results, reproducibility findings, interpretation, and limitations
- kept generated datasets and outputs out of Git

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

v0.6.0 does not include:

- downloaded or bundled public datasets
- bundled model weights
- generated reports or predictions committed to the repository
- automatic Hugging Face token handling
- CONCH as a required dependency
- model downloads in CI
- clinical diagnosis or clinical decision support
- WSI-scale high-throughput processing
- model training or fine-tuning

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, prompt-grid reports, prediction files, run logs, or generated comparison files.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli validate-config configs/zero_shot_mhist_clip_sample.json
python -m pathvlm_litebench.cli validate-config configs/zero_shot_mhist_plip_sample.json
python -m pathvlm_litebench.cli validate-config configs/zero_shot_prompt_grid_mhist_sample.json
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --output-root outputs/zero_shot_prompt_grid_mhist_sample_release_check --dry-run
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
```

Optional local benchmark checks:

```bash
python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_clip_sample.json
python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_plip_sample.json
pathvlm-litebench run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --output-root outputs/zero_shot_prompt_grid_mhist_sample_v060_audit
```

The optional runs may download or load model weights and require local dataset paths. Optional gated models such as CONCH still require Hugging Face access and the optional CONCH package.

## Release Highlights

PathVLM-LiteBench v0.6.0 improves reproducible local benchmark review:

- users can run sampled MHIST CLIP and PLIP baselines from committed configs
- prompt-grid runs can be directed to a fresh output root without editing JSON
- sampled MHIST baseline and prompt-grid audit docs provide concrete local behavior observations
- generated outputs remain local and ignored
- CI remains offline and lightweight
