# Prompt-Scored Coordinate Heatmap Workflow

This guide describes the current workflow for scoring pre-extracted patch images against one text prompt and rendering the resulting scores over a patch-coordinate grid.

The workflow is patch-level and local-first. PathVLM-LiteBench does not read whole-slide image files, tile slides, run tissue detection, or render slide pyramids.

## Artifact Rendering vs Prompt Scoring

PathVLM-LiteBench now has two related coordinate heatmap commands:

| Command | Input scores | Loads a model | Typical use |
|---|---|---:|---|
| `render-coordinate-heatmap` | Existing score CSV | No | Re-render a heatmap from saved artifacts |
| `score-coordinate-heatmap` | Computed from patch images and one text prompt | Yes | Produce `scores.csv`, `heatmap.png`, and `metadata.json` from a coordinate manifest |

Use `render-coordinate-heatmap` when scores already exist. It reads only the coordinate manifest and score CSV, then writes a PNG heatmap.

Use `score-coordinate-heatmap` when you want PathVLM-LiteBench to load a CLIP-style model, encode the listed patch images, score them against a prompt, save a score CSV, and render a heatmap.

## Local Data Layout

Prepare pre-extracted patch images and a coordinate-aware manifest:

```text
dataset/
`-- patch_coordinates/
    |-- coordinate_manifest.csv
    `-- patches/
        |-- patch_001.png
        |-- patch_002.png
        `-- ...

outputs/
`-- patch_coordinate_heatmap_scored/
```

The `dataset/` and `outputs/` directories are local and ignored by Git. Do not commit real pathology images, patient-level metadata, generated score CSV files, generated heatmaps, model weights, embedding files, or Hugging Face caches.

## Coordinate Manifest

The manifest must include patch image paths and patch coordinates:

```csv
image_path,x,y,width,height,label,slide_id
patches/patch_001.png,0,0,224,224,tumor,slide_a
patches/patch_002.png,224,0,224,224,normal,slide_a
patches/patch_003.png,0,224,224,224,tumor,slide_a
```

Required columns:

- `image_path`
- `x`
- `y`

Supported optional columns include `width`, `height`, `label`, `split`, `case_id`, and `slide_id`. Additional non-empty columns are preserved as metadata when the manifest is loaded.

## CLI Run

Run prompt-scored patch-coordinate heatmap generation with explicit arguments:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --prompt "a histopathology image of tumor tissue" \
  --output-dir outputs/patch_coordinate_heatmap_scored \
  --model clip \
  --device auto
```

This command requires the patch image files to exist. Relative manifest paths are resolved against the manifest directory unless `--image-root` is provided.

The command writes:

- `outputs/patch_coordinate_heatmap_scored/scores.csv`
- `outputs/patch_coordinate_heatmap_scored/heatmap.png`
- `outputs/patch_coordinate_heatmap_scored/metadata.json`

The score CSV includes one row per scored patch with coordinate metadata, the numeric score, and the prompt used for scoring.

The metadata JSON records the run context needed for lightweight auditing, including the prompt, model, device, manifest path, column names, output paths, patch count, toolkit version, colormap, title, and creation time.

## Dry Run Without Model Loading

Use `--dry-run` to validate the config, coordinate manifest, path resolution, `--max-images` truncation, and resolved output paths before loading a model:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --prompt "a histopathology image of tumor tissue" \
  --output-dir outputs/patch_coordinate_heatmap_scored \
  --model clip \
  --device auto \
  --dry-run
```

Dry-run mode requires the manifest and referenced patch paths to exist, but it does not load patch images, create a model, run inference, or write `scores.csv`, `heatmap.png`, or `metadata.json`.

## Smoke Run With max-images

Use `--max-images` for a quick local smoke run before scoring a larger coordinate manifest:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --prompt "a histopathology image of tumor tissue" \
  --output-dir outputs/patch_coordinate_heatmap_scored_smoke \
  --model clip \
  --device auto \
  --max-images 16
```

`--max-images` truncates the manifest records before loading images and running model inference. It is useful for checking paths, prompt handling, output paths, and local model setup.

For a model-free check, combine `--max-images` with `--dry-run`.

## Config-Driven Run

The committed example config is:

```text
configs/patch_coordinate_heatmap_scoring_demo_config.json
```

Validate it without running model inference:

```bash
pathvlm-litebench validate-config \
  configs/patch_coordinate_heatmap_scoring_demo_config.json
```

Run from the config:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --config configs/patch_coordinate_heatmap_scoring_demo_config.json
```

The config task name is `patch_coordinate_heatmap_scoring`.

Example fields:

```json
{
  "task": "patch_coordinate_heatmap_scoring",
  "manifest": "dataset/patch_coordinates/coordinate_manifest.csv",
  "prompt": "a histopathology image of tumor tissue",
  "output_dir": "outputs/patch_coordinate_heatmap_scored",
  "score_csv": "outputs/patch_coordinate_heatmap_scored/scores.csv",
  "heatmap_output": "outputs/patch_coordinate_heatmap_scored/heatmap.png",
  "metadata_output": "outputs/patch_coordinate_heatmap_scored/metadata.json",
  "model": "clip",
  "device": "auto",
  "image_root": null,
  "path_column": "image_path",
  "x_column": "x",
  "y_column": "y",
  "max_images": 16,
  "title": "Prompt-scored patch coordinate heatmap",
  "cmap": "viridis"
}
```

Config field notes:

- `manifest`: coordinate-aware patch manifest CSV path.
- `prompt`: text prompt encoded once and scored against each patch image.
- `output_dir`: default directory for generated `scores.csv`, `heatmap.png`, and `metadata.json`.
- `score_csv`: optional explicit score CSV path. Defaults to `output_dir/scores.csv`.
- `heatmap_output`: optional explicit PNG path. Defaults to `output_dir/heatmap.png`.
- `metadata_output`: optional explicit metadata JSON path. Defaults to `output_dir/metadata.json`.
- `model`: model registry key or Hugging Face model name.
- `device`: one of `auto`, `cpu`, or `cuda`.
- `image_root`: optional root for resolving relative manifest image paths.
- `path_column`, `x_column`, `y_column`: manifest column names.
- `max_images`: optional positive integer for smoke runs and small local checks.
- `title`: optional heatmap title. If omitted, the prompt is used as the title.
- `cmap`: Matplotlib colormap name.

## CLI Overrides

Command-line arguments override config values at runtime:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --config configs/patch_coordinate_heatmap_scoring_demo_config.json \
  --prompt "a histopathology image with lymphocyte-rich tissue" \
  --output-dir outputs/patch_coordinate_heatmap_scored_lymphocyte \
  --max-images 16
```

Supported overrides include:

- `--manifest`
- `--prompt`
- `--output-dir`
- `--score-csv`
- `--heatmap-output`
- `--metadata-output`
- `--model`
- `--device`
- `--image-root`
- `--path-column`
- `--x-column`
- `--y-column`
- `--max-images`
- `--title`
- `--cmap`

## Model Loading

`score-coordinate-heatmap` loads the selected CLIP-style model and runs model inference unless `--dry-run` is used. It may download model weights if they are not already cached locally.

For offline artifact work, use `render-coordinate-heatmap` with an existing score CSV. That command does not load a model, run inference, or download weights.

CI covers prompt-scored coordinate heatmap logic with fake model wrappers, so CI tests do not download model weights.

CONCH remains optional and gated. This workflow does not make CONCH a required dependency and does not handle Hugging Face tokens.

## Output Interpretation

Use the generated PNG as a patch-coordinate score visualization only. It shows score variation across the coordinate grid represented by the pre-extracted patches in the manifest.

Good wording:

- "This prompt assigned higher scores to these patch coordinates."
- "The heatmap shows patch-level score variation across a coordinate grid."

Avoid:

- "This heatmap diagnoses a slide."
- "This is a whole-slide image pipeline."
- "This identifies tumor boundaries."

The workflow does not perform clinical interpretation. It does not inspect tissue beyond the patch images passed through the selected model.
