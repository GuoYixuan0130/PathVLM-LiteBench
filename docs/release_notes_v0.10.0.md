# v0.10.0 Release Notes

## Summary

PathVLM-LiteBench v0.10.0 focuses on prompt-set workflows for prompt-scored patch-coordinate heatmaps. It lets users define several prompts over the same coordinate manifest, inspect the expanded run plan without model loading, run the prompt set, and save consistent per-prompt artifacts plus artifact-only comparison summaries.

This release builds on the v0.7.0 artifact-only rendering workflow, the v0.8.0 single-prompt scoring workflow, and the v0.9.0 reproducibility and comparison layer. `render-coordinate-heatmap` remains model-free for existing score artifacts. `score-coordinate-heatmap` remains the model-backed single-prompt path. `score-coordinate-heatmap-prompt-set` is the model-backed multi-prompt path, with a model-free `--dry-run`. `compare-coordinate-heatmap-scores` remains available for artifact-only comparison of saved score outputs.

This release does not add whole-slide image reading, slide tiling, tissue detection, slide pyramid rendering, bundled pathology data, bundled model weights, required CONCH access, Hugging Face token handling, clinical diagnosis, or clinical decision support. It remains a research and educational toolkit for patch-level evaluation and visualization.

## What Is Included

Prompt-set config support:

- adds `PatchCoordinateHeatmapPrompt`
- adds `PatchCoordinateHeatmapPromptSetConfig`
- adds JSON serialization and loading helpers for prompt-set configs
- adds `task: patch_coordinate_heatmap_prompt_set`
- adds `validate-config` support for prompt-set configs
- adds `configs/patch_coordinate_heatmap_prompt_set_demo_config.json`
- supports shared manifest, output root, model, device, image root, coordinate columns, `max_images`, and default colormap settings
- supports per-prompt keys, prompt text, optional heatmap titles, optional output directories, and optional colormap overrides
- rejects duplicate prompt keys, invalid prompt keys, empty prompts, bad devices, unknown top-level fields, and unknown prompt fields

Prompt-set dry-run:

- adds `score-coordinate-heatmap-prompt-set --dry-run`
- loads and validates the prompt-set config
- loads coordinate manifest metadata
- applies `--max-images` truncation before reporting patch count
- prints per-prompt output directories and artifact paths
- prints comparison CSV and Markdown paths
- does not load patch images
- does not create a model
- does not run inference
- does not write score, heatmap, metadata, or comparison outputs

Prompt-set scoring:

- adds model-backed `score-coordinate-heatmap-prompt-set`
- runs several prompt-scored coordinate heatmap runs over the same coordinate manifest
- reuses existing single-prompt scoring and heatmap helpers
- writes per-prompt `scores.csv`
- writes per-prompt `heatmap.png`
- writes per-prompt `metadata.json`
- records `prompt_key` in prompt-set metadata
- supports default output directories under `output_root/<prompt_key>`
- supports per-prompt `output_dir` overrides
- supports `--output-root` and `--max-images` CLI overrides

Prompt-set comparison summaries:

- writes `score_summary.csv` by default under the prompt-set output root
- writes `score_summary.md` by default under the prompt-set output root
- supports `--comparison-output-csv`
- supports `--comparison-output-md`
- reuses artifact-only score comparison helpers
- reads saved per-prompt `scores.csv` and `metadata.json` outputs
- does not load models or images while building comparison summaries

Documentation:

- added `docs/prompt_set_coordinate_heatmap_workflow.md`
- published v0.10.0 release notes
- updated README user workflow links
- documented prompt-set config fields, dry-run behavior, output layout, model-loading behavior, comparison summaries, and artifact hygiene
- documented that generated score CSVs, heatmaps, metadata files, comparison outputs, datasets, caches, and weights should remain uncommitted

Test and CI coverage:

- added prompt-set config roundtrip tests
- added prompt-set config validation rejection tests
- added example config loading coverage
- added CLI validation coverage for prompt-set configs
- added dry-run tests that verify no model creation and no output writes
- added fake-model prompt-set scoring tests
- added comparison summary coverage through prompt-set CLI tests
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
- v0.9.0 `score-coordinate-heatmap --dry-run`
- v0.9.0 artifact-only `compare-coordinate-heatmap-scores`
- synthetic patch-coordinate heatmap demo

CONCH remains optional and gated. v0.10.0 does not make CONCH a required dependency.

## What Is Intentionally Not Included

v0.10.0 does not include:

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
- generated heatmaps, score CSV files, metadata files, comparison files, or reports committed to the repository
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
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_prompt_set_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli render-coordinate-heatmap --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap-prompt-set --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli compare-coordinate-heatmap-scores --help
```

Model-free prompt-set dry-run check:

```powershell
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap-prompt-set `
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json `
  --dry-run `
  --max-images 16
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

Manual model-backed checks are optional for release verification because `score-coordinate-heatmap` and `score-coordinate-heatmap-prompt-set` may download model weights if they are not already cached locally. CI covers these paths with fake models.

## Example Commands

Validate the prompt-set example config:

```bash
pathvlm-litebench validate-config \
  configs/patch_coordinate_heatmap_prompt_set_demo_config.json
```

Dry-run a prompt set without loading images or a model:

```bash
pathvlm-litebench score-coordinate-heatmap-prompt-set \
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json \
  --dry-run
```

Run a prompt set from config:

```bash
pathvlm-litebench score-coordinate-heatmap-prompt-set \
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json
```

Override prompt-set output paths for a smoke run:

```bash
pathvlm-litebench score-coordinate-heatmap-prompt-set \
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json \
  --output-root outputs/patch_coordinate_heatmap_prompt_set_smoke \
  --comparison-output-csv outputs/patch_coordinate_heatmap_prompt_set_smoke/score_summary.csv \
  --comparison-output-md outputs/patch_coordinate_heatmap_prompt_set_smoke/score_summary.md \
  --max-images 16
```

Compare saved prompt-set artifacts without model loading:

```bash
pathvlm-litebench compare-coordinate-heatmap-scores \
  --score-csvs \
    outputs/patch_coordinate_heatmap_prompt_set/tumor/scores.csv \
    outputs/patch_coordinate_heatmap_prompt_set/lymphocyte/scores.csv \
  --metadata-jsons \
    outputs/patch_coordinate_heatmap_prompt_set/tumor/metadata.json \
    outputs/patch_coordinate_heatmap_prompt_set/lymphocyte/metadata.json \
  --run-names tumor lymphocyte \
  --output-csv outputs/patch_coordinate_heatmap_prompt_set/score_summary.csv \
  --output-md outputs/patch_coordinate_heatmap_prompt_set/score_summary.md
```

Use artifact-only rendering when scores already exist:

```bash
pathvlm-litebench render-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --score-csv outputs/patch_coordinate_heatmap_demo/scores.csv \
  --output outputs/patch_coordinate_heatmap_demo/heatmap.png
```

## Release Highlights

PathVLM-LiteBench v0.10.0 makes repeated prompt-scored patch-coordinate heatmap experiments easier to run and compare:

- prompt sets can be defined once in a config file
- dry-run mode expands all prompt runs before model loading
- batch scoring writes consistent per-prompt artifacts
- metadata records prompt keys and run context
- prompt-set comparison summaries are generated from saved artifacts
- artifact-only workflows remain available for rendering and comparison
- CI remains lightweight and avoids model downloads
- documentation keeps the feature framed as patch-coordinate visualization, not whole-slide image processing
