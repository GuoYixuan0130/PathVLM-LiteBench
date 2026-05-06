# v0.4.1 Release Notes

## Summary

PathVLM-LiteBench v0.4.1 adds a zero-shot prompt-grid workflow and documents post-v0.4.0 CLIP/PLIP/CONCH model-behavior observations.

This is a patch release after the v0.4.0 optional CONCH integration. It does not change the core model wrapper contract or make CONCH a required dependency.

This release is intended for research and educational use only.

## What Is Included

Prompt-grid workflow:

- added `pathvlm-litebench run-zero-shot-grid`
- added `--dry-run` support to preview model/prompt combinations without loading models
- added JSON config support for model lists, class names, prompt pairs, device, manifest paths, output root, logs, and comparison output
- added `configs/zero_shot_prompt_grid_mhist_sample.json`
- added automatic zero-shot comparison Markdown generation from saved report artifacts
- added offline tests for config loading, run expansion, command construction, CLI dry-run behavior, and comparison generation with fake reports

Model-behavior documentation:

- added a sampled CLIP vs PLIP vs CONCH MHIST zero-shot observation
- added a sampled CLIP/PLIP/CONCH prompt-grid observation
- updated README links and project positioning documentation

## What Remains Unchanged

The toolkit remains focused on:

- frozen model inference
- patch-level images
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- local CSV/JSON/Markdown reporting
- offline CI tests

Existing workflows remain available:

- CLIP, PLIP, and optional CONCH model wrappers
- retrieval, zero-shot classification, and prompt sensitivity demos
- manifest conversion and sampling
- summary and comparison Markdown reports
- embedding cache and visualization utilities

## What Is Intentionally Not Included

v0.4.1 does not include:

- CONCH as a required dependency
- model downloads in CI
- bundled model weights
- committed generated reports or prediction files
- clinical diagnosis or clinical decision support
- full WSI high-throughput processing
- model training or fine-tuning

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, prompt-grid reports, or generated comparison files.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
python -m pathvlm_litebench.cli run-zero-shot-grid --help
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --dry-run
```

Optional local prompt-grid run:

```bash
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json
```

This optional run may download or load model weights and requires local access for optional gated models such as CONCH.

## Release Highlights

PathVLM-LiteBench v0.4.1 improves reproducible prompt-behavior analysis:

- users can run a model-by-prompt zero-shot grid from one config file
- saved report comparison is generated automatically
- prompt-grid results are documented conservatively as local behavior observations
- CI remains lightweight and offline
