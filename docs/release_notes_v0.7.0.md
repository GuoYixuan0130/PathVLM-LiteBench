# v0.7.0 Release Notes

## Summary

PathVLM-LiteBench v0.7.0 adds a patch-coordinate heatmap workflow for visualizing precomputed patch scores over coordinate-aware patch grids. The release adds coordinate-aware manifest loading, score aggregation, heatmap export, an artifact-only render CLI, config validation, and a synthetic demo.

This release does not add whole-slide image processing, slide tiling, model-inference-driven heatmaps, bundled datasets, bundled weights, or clinical decision support. It remains a research and educational toolkit for patch-level evaluation and visualization.

## What Is Included

Patch-coordinate heatmap utilities:

- added coordinate-aware patch records with required `image_path`, `x`, and `y` fields
- added optional coordinate metadata fields such as `width`, `height`, `label`, `split`, `case_id`, and `slide_id`
- added score-grid aggregation over patch `x`/`y` coordinates
- averages repeated patch coordinates
- leaves missing coordinate cells blank in the heatmap grid
- added PNG heatmap export for precomputed patch scores
- added coordinate score CSV export

Artifact-only CLI workflow:

- added `pathvlm-litebench render-coordinate-heatmap`
- supports rendering from a coordinate manifest and an existing score CSV
- supports score alignment by `image_path` or row order
- supports runtime output, column, title, and colormap options
- does not load models, run inference, or download weights

Config-driven workflow:

- added `PatchCoordinateHeatmapConfig`
- added `configs/patch_coordinate_heatmap_demo_config.json`
- added `validate-config` support for `task: patch_coordinate_heatmap`
- added `render-coordinate-heatmap --config`
- supports command-line overrides for config values such as `--output`, `--score-column`, and `--align-by`

Synthetic demo:

- added `examples/05_patch_coordinate_heatmap_demo.py`
- generates synthetic colored patch tiles, a coordinate manifest, a score CSV, and a heatmap
- writes generated artifacts under ignored `outputs/`
- requires no real pathology data, model inference, or model downloads

Documentation and scope cleanup:

- added `docs/patch_coordinate_heatmap_workflow.md`
- added `docs/v0.7.0_plan.md`
- added `docs/release_checklist_v0.7.0.md`
- removed misleading WSI-level heatmap wording from public milestone docs
- kept the feature framed as patch-coordinate visualization only

## What Remains Unchanged

The toolkit remains focused on:

- frozen CLIP-style model inference for retrieval and classification workflows
- patch-level images
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- local CSV, JSON, Markdown, HTML, and PNG artifacts
- offline CI tests

Existing workflows remain available:

- CLIP, PLIP, and optional CONCH model wrappers
- retrieval, zero-shot classification, prompt sensitivity, and zero-shot prompt-grid demos
- manifest conversion and balanced sampling
- summary and comparison Markdown reports
- embedding cache and visualization utilities
- sampled MHIST CLIP/PLIP configs and prompt-grid config

## What Is Intentionally Not Included

v0.7.0 does not include:

- whole-slide image parsing
- slide tiling or tissue detection
- stain normalization or slide pyramid rendering
- model-inference-driven heatmap generation
- downloaded or bundled public datasets
- bundled model weights
- generated heatmaps or score CSV files committed to the repository
- automatic Hugging Face token handling
- CONCH as a required dependency
- model downloads in CI
- clinical diagnosis or clinical decision support
- model training or fine-tuning

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, generated patches, heatmaps, score CSV files, reports, prediction files, or run logs.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_demo_config.json
python -m pathvlm_litebench.cli render-coordinate-heatmap --help
python examples/05_patch_coordinate_heatmap_demo.py
```

Optional artifact-only local check after running the synthetic demo:

```bash
python -m pathvlm_litebench.cli render-coordinate-heatmap \
  --manifest outputs/patch_coordinate_heatmap_demo_synthetic/coordinate_manifest.csv \
  --score-csv outputs/patch_coordinate_heatmap_demo_synthetic/scores.csv \
  --align-by image_path \
  --output outputs/patch_coordinate_heatmap_demo_synthetic/cli_heatmap.png
```

The optional check reads synthetic local artifacts only. These generated files should remain uncommitted.

## Release Highlights

PathVLM-LiteBench v0.7.0 adds a small patch-coordinate visualization path without changing the project boundary:

- users can validate a committed patch-coordinate heatmap config without local artifacts
- users can render a heatmap from existing coordinate and score artifacts
- users can run a synthetic demo without real pathology data
- CI remains offline and lightweight
- documentation avoids presenting the feature as whole-slide image processing
