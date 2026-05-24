# v0.9.0 Release Notes

## Summary

PathVLM-LiteBench v0.9.0 focuses on reproducibility and comparison for prompt-scored patch-coordinate heatmaps. It adds metadata sidecars for scoring runs, a model-free dry-run mode for validating scoring inputs, and an artifact-only comparison command for saved score CSV outputs.

This release builds on the v0.7.0 artifact-only rendering workflow and the v0.8.0 prompt-scored heatmap workflow. `render-coordinate-heatmap` remains model-free for existing score artifacts. `score-coordinate-heatmap` remains the model-backed scoring path, and now writes run metadata and supports `--dry-run`. `compare-coordinate-heatmap-scores` compares saved score artifacts without loading models or images.

This release does not add whole-slide image reading, slide tiling, tissue detection, slide pyramid rendering, bundled pathology data, bundled model weights, required CONCH access, Hugging Face token handling, clinical diagnosis, or clinical decision support. It remains a research and educational toolkit for patch-level evaluation and visualization.

## What Is Included

Prompt-scored heatmap metadata:

- writes `metadata.json` by default from `score-coordinate-heatmap`
- records run context including prompt, model, device, manifest path, column names, output paths, patch count, toolkit version, colormap, title, and creation time
- adds `metadata_output` to `PatchCoordinateHeatmapScoringConfig`
- adds `--metadata-output` as a CLI override
- updates the committed prompt-scored heatmap example config

Dry-run validation:

- adds `score-coordinate-heatmap --dry-run`
- validates the config and coordinate manifest
- resolves score CSV, heatmap PNG, and metadata JSON output paths
- applies `--max-images` truncation before reporting patch count
- does not load patch images
- does not create a model
- does not run inference
- does not write score, heatmap, or metadata outputs

Artifact-only score comparison:

- adds `pathvlm-litebench compare-coordinate-heatmap-scores`
- reads multiple saved `scores.csv` artifacts
- loads sibling `metadata.json` files when present
- supports explicit `--metadata-jsons`
- supports user-provided `--run-names`
- writes a compact comparison CSV
- optionally writes a Markdown summary
- reports per-run score row count, mean, standard deviation, minimum, and maximum
- includes metadata fields such as prompt, model, device, manifest path, and heatmap output when available
- rejects mismatched row counts by default for comparable prompt runs
- supports `--allow-row-count-mismatch` for intentional mixed-size summaries

Documentation:

- added `docs/v0.9.0_plan.md`
- added `docs/v0.9.0_pre_release_audit.md`
- added `docs/release_checklist_v0.9.0.md`
- updated `docs/prompt_scored_coordinate_heatmap_workflow.md`
- updated README CLI listings and release links
- documented metadata output, dry-run behavior, and artifact-only score comparison
- documented that generated score CSVs, heatmaps, metadata files, comparison outputs, datasets, caches, and weights should remain uncommitted

Test and CI coverage:

- added CLI tests for scoring metadata output
- added CLI tests that verify dry-run skips model creation and output writes
- added config tests for `metadata_output`
- added example config tests for metadata output
- added helper tests for score comparison summaries
- added CLI tests for `compare-coordinate-heatmap-scores`
- uses fake model wrappers and saved artifacts in CI
- does not download model weights in CI

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
- manifest conversion and balanced sampling
- summary and comparison Markdown reports
- embedding cache and visualization utilities
- v0.7.0 artifact-only `render-coordinate-heatmap`
- v0.8.0 model-backed `score-coordinate-heatmap`
- synthetic patch-coordinate heatmap demo

CONCH remains optional and gated. v0.9.0 does not make CONCH a required dependency.

## What Is Intentionally Not Included

v0.9.0 does not include:

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
- generated heatmaps, score CSV files, metadata files, or comparison files committed to the repository
- real pathology images or whole-slide image files committed to the repository

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, generated patches, heatmaps, score CSV files, metadata JSON files, comparison artifacts, reports, prediction files, or run logs.

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
.venv\Scripts\python.exe -m pathvlm_litebench.cli compare-coordinate-heatmap-scores --help
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

Optional comparison check after creating two saved scoring runs:

```powershell
.venv\Scripts\python.exe -m pathvlm_litebench.cli compare-coordinate-heatmap-scores `
  --score-csvs `
    outputs/patch_coordinate_heatmap_scored_tumor/scores.csv `
    outputs/patch_coordinate_heatmap_scored_lymphocyte/scores.csv `
  --run-names tumor lymphocyte `
  --output-csv outputs/patch_coordinate_heatmap_comparison/score_summary.csv `
  --output-md outputs/patch_coordinate_heatmap_comparison/score_summary.md
```

Manual model-backed checks are optional for release verification because `score-coordinate-heatmap` may download model weights if they are not already cached locally. CI covers this path with fake models.

## Example Commands

Dry-run a prompt-scored heatmap config without loading a model:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --config configs/patch_coordinate_heatmap_scoring_demo_config.json \
  --dry-run
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

Compare saved prompt-scored score artifacts without model loading:

```bash
pathvlm-litebench compare-coordinate-heatmap-scores \
  --score-csvs \
    outputs/patch_coordinate_heatmap_scored_tumor/scores.csv \
    outputs/patch_coordinate_heatmap_scored_lymphocyte/scores.csv \
  --run-names tumor lymphocyte \
  --output-csv outputs/patch_coordinate_heatmap_comparison/score_summary.csv \
  --output-md outputs/patch_coordinate_heatmap_comparison/score_summary.md
```

Use artifact-only rendering when scores already exist:

```bash
pathvlm-litebench render-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --score-csv outputs/patch_coordinate_heatmap_demo/scores.csv \
  --output outputs/patch_coordinate_heatmap_demo/heatmap.png
```

## Release Highlights

PathVLM-LiteBench v0.9.0 makes prompt-scored patch-coordinate heatmap experiments easier to audit and compare:

- scoring runs now write `metadata.json`
- dry-run validation can check manifests and output paths before model loading
- saved score artifacts can be compared without rerunning inference
- comparison summaries include prompt and model context when metadata is available
- artifact-only workflows remain available for rendering and comparison
- CI remains lightweight and avoids model downloads
- documentation keeps the feature framed as patch-coordinate visualization, not whole-slide image processing
