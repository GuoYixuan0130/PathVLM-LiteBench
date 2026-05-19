# v0.8.0 Release Notes Draft

## Summary

PathVLM-LiteBench v0.8.0 focuses on the prompt-scored patch-coordinate heatmap workflow. It adds a model-backed command that scores pre-extracted coordinate patches against one text prompt, writes a coordinate-aware score CSV, and renders a patch-coordinate heatmap.

This release builds on the v0.7.0 artifact-only heatmap workflow. `render-coordinate-heatmap` remains available for existing score CSV files and does not load models. `score-coordinate-heatmap` is the model-backed path and may download weights if the selected model is not already cached locally.

This release does not add whole-slide image reading, slide tiling, tissue detection, slide pyramid rendering, bundled pathology data, bundled model weights, required CONCH access, Hugging Face token handling, clinical diagnosis, or clinical decision support. It remains a research and educational toolkit for patch-level evaluation and visualization.

## What Is Included

Prompt-scored coordinate heatmap CLI:

- added `pathvlm-litebench score-coordinate-heatmap`
- loads patch images listed in a coordinate-aware manifest
- loads a selected CLIP-style model through the existing model registry
- encodes one text prompt
- computes one image-text similarity score per patch
- writes `scores.csv`
- writes `heatmap.png`
- supports `--device auto`, `--device cpu`, and `--device cuda`
- supports `--max-images` for small local smoke runs

Prompt-scored heatmap config workflow:

- added `PatchCoordinateHeatmapScoringConfig`
- added `configs/patch_coordinate_heatmap_scoring_demo_config.json`
- added config helpers for serialization, loading, saving, and validation
- added `validate-config` support for `task: patch_coordinate_heatmap_scoring`
- added `score-coordinate-heatmap --config`
- supports command-line overrides for config-driven runs

Supported scoring config fields include:

- `manifest`
- `prompt`
- `output_dir`
- `score_csv`
- `heatmap_output`
- `model`
- `device`
- `image_root`
- `path_column`
- `x_column`
- `y_column`
- `max_images`
- `title`
- `cmap`

CLI override support includes:

- `--manifest`
- `--prompt`
- `--output-dir`
- `--score-csv`
- `--heatmap-output`
- `--model`
- `--device`
- `--image-root`
- `--path-column`
- `--x-column`
- `--y-column`
- `--max-images`
- `--title`
- `--cmap`

Documentation:

- added `docs/v0.8.0_plan.md`
- added `docs/prompt_scored_coordinate_heatmap_workflow.md`
- added `docs/release_checklist_v0.8.0.md`
- updated README development links for the v0.8.0 plan, checklist, and prompt-scored workflow
- documented the difference between artifact-only rendering and prompt-scored model inference
- documented that prompt-scored runs may download model weights when they are not already cached
- documented that generated score CSVs, heatmaps, datasets, caches, and weights should remain uncommitted

Test and CI coverage:

- added unit tests for prompt-scored patch scoring logic
- added CLI tests for `score-coordinate-heatmap`
- added config tests for `patch_coordinate_heatmap_scoring`
- added example config tests that load without requiring local artifacts
- uses fake model wrappers in CI for scoring paths
- does not download model weights in CI

## What Remains Unchanged

The toolkit remains focused on:

- frozen CLIP-style model inference for retrieval, classification, and patch scoring workflows
- patch-level images
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
- manifest conversion and balanced sampling
- summary and comparison Markdown reports
- embedding cache and visualization utilities
- v0.7.0 artifact-only `render-coordinate-heatmap`
- synthetic patch-coordinate heatmap demo

CONCH remains optional and gated. v0.8.0 does not make CONCH a required dependency.

## What Is Intentionally Not Included

v0.8.0 does not include:

- whole-slide image file reading
- slide tiling or tissue detection
- stain normalization or slide pyramid rendering
- tumor boundary identification
- clinical diagnosis or clinical decision support
- model training or fine-tuning
- downloaded or bundled public datasets
- bundled model weights
- required model downloads in CI
- required CONCH access
- Hugging Face token handling
- generated heatmaps or score CSV files committed to the repository
- real pathology images or whole-slide image files committed to the repository

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, generated patches, heatmaps, score CSV files, reports, prediction files, or run logs.

## Verification

Recommended verification before tagging on Windows:

```powershell
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m pathvlm_litebench.cli version
.venv\Scripts\python.exe -m pathvlm_litebench.cli models
.venv\Scripts\python.exe -m pathvlm_litebench.cli demos
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_scoring_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli render-coordinate-heatmap --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap --help
```

Optional artifact-only local check after running the synthetic demo:

```powershell
.venv\Scripts\python.exe examples/05_patch_coordinate_heatmap_demo.py

.venv\Scripts\python.exe -m pathvlm_litebench.cli render-coordinate-heatmap `
  --manifest outputs/patch_coordinate_heatmap_demo_synthetic/coordinate_manifest.csv `
  --score-csv outputs/patch_coordinate_heatmap_demo_synthetic/scores.csv `
  --align-by image_path `
  --output outputs/patch_coordinate_heatmap_demo_synthetic/cli_heatmap.png
```

Manual model-backed checks are optional for release verification because `score-coordinate-heatmap` may download model weights if they are not already cached locally. CI covers this path with fake models.

## Example Commands

Validate the prompt-scored heatmap example config:

```bash
pathvlm-litebench validate-config \
  configs/patch_coordinate_heatmap_scoring_demo_config.json
```

Run prompt-scored heatmap generation from config:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --config configs/patch_coordinate_heatmap_scoring_demo_config.json
```

Override config values for a small smoke run:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --config configs/patch_coordinate_heatmap_scoring_demo_config.json \
  --prompt "a histopathology image with lymphocyte-rich tissue" \
  --output-dir outputs/patch_coordinate_heatmap_scored_lymphocyte \
  --max-images 16
```

Use artifact-only rendering when scores already exist:

```bash
pathvlm-litebench render-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --score-csv outputs/patch_coordinate_heatmap_demo/scores.csv \
  --output outputs/patch_coordinate_heatmap_demo/heatmap.png
```

## Release Highlights

PathVLM-LiteBench v0.8.0 makes prompt-scored patch-coordinate heatmaps easier to run and verify:

- users can score coordinate patches from CLI arguments or a JSON config
- users can validate the committed scoring config without local artifacts
- users can run small smoke checks with `--max-images`
- users get both `scores.csv` and `heatmap.png` from the scoring command
- artifact-only rendering remains available without model loading
- CI remains lightweight and avoids model downloads
- documentation keeps the feature framed as patch-coordinate visualization, not whole-slide image processing
