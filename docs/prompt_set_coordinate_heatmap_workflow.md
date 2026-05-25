# Prompt-Set Coordinate Heatmap Workflow

This guide describes the prompt-set workflow for running several prompt-scored patch-coordinate heatmap experiments over the same coordinate manifest.

The workflow is patch-level and local-first. PathVLM-LiteBench does not read whole-slide image files, tile slides, run tissue detection, render slide pyramids, or make clinical diagnostic claims.

## When To Use This Workflow

Use `score-coordinate-heatmap-prompt-set` when you want to score the same pre-extracted patch list against several text prompts and save consistent per-prompt artifacts.

Related commands:

| Command | Scope | Loads a model | Typical use |
|---|---|---:|---|
| `render-coordinate-heatmap` | Existing score CSV | No | Re-render one heatmap from saved artifacts |
| `score-coordinate-heatmap` | One prompt | Yes | Score one prompt and write one set of artifacts |
| `score-coordinate-heatmap-prompt-set` | Several prompts over one manifest | Yes, unless `--dry-run` is used | Score a prompt set and write per-prompt artifacts plus comparison summaries |
| `compare-coordinate-heatmap-scores` | Existing score CSV artifacts | No | Compare saved prompt runs without rerunning inference |

The prompt-set command is the batch-style path for repeated prompt experiments. It reuses the same coordinate manifest and output conventions for each prompt.

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
`-- patch_coordinate_heatmap_prompt_set/
```

The `dataset/` and `outputs/` directories are local and ignored by Git. Do not commit real pathology images, patient-level metadata, generated score CSV files, generated heatmaps, metadata JSON files, generated reports, model weights, embedding files, or Hugging Face caches.

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

## Prompt-Set Config

The committed example config is:

```text
configs/patch_coordinate_heatmap_prompt_set_demo_config.json
```

Validate it without loading a model:

```bash
pathvlm-litebench validate-config \
  configs/patch_coordinate_heatmap_prompt_set_demo_config.json
```

Example shape:

```json
{
  "task": "patch_coordinate_heatmap_prompt_set",
  "manifest": "dataset/patch_coordinates/coordinate_manifest.csv",
  "output_root": "outputs/patch_coordinate_heatmap_prompt_set",
  "comparison_output_csv": "outputs/patch_coordinate_heatmap_prompt_set/score_summary.csv",
  "comparison_output_md": "outputs/patch_coordinate_heatmap_prompt_set/score_summary.md",
  "model": "clip",
  "device": "auto",
  "path_column": "image_path",
  "x_column": "x",
  "y_column": "y",
  "max_images": 16,
  "cmap": "viridis",
  "prompts": [
    {
      "key": "tumor",
      "prompt": "a histopathology image of tumor tissue",
      "title": "Tumor prompt score"
    },
    {
      "key": "lymphocyte",
      "prompt": "a histopathology image with lymphocyte-rich tissue",
      "title": "Lymphocyte-rich prompt score"
    }
  ]
}
```

Config field notes:

- `manifest`: coordinate-aware patch manifest CSV path.
- `output_root`: default root for per-prompt output directories.
- `comparison_output_csv`: optional comparison CSV path. Defaults to `output_root/score_summary.csv`.
- `comparison_output_md`: optional comparison Markdown path. Defaults to `output_root/score_summary.md`.
- `model`: model registry key or Hugging Face model name.
- `device`: one of `auto`, `cpu`, or `cuda`.
- `image_root`: optional root for resolving relative manifest image paths.
- `path_column`, `x_column`, `y_column`: manifest column names.
- `max_images`: optional positive integer for smoke runs and small local checks.
- `cmap`: default Matplotlib colormap for prompt heatmaps.
- `prompts`: non-empty list of prompt entries.

Prompt entry fields:

- `key`: stable ASCII run key used for the default output subdirectory.
- `prompt`: text prompt encoded and scored against each patch image.
- `title`: optional heatmap title. If omitted, the prompt text is used.
- `output_dir`: optional explicit output directory for this prompt. Defaults to `output_root/<key>`.
- `cmap`: optional colormap override for this prompt.

Prompt keys must start with an ASCII letter or digit and may contain ASCII letters, digits, `.`, `_`, or `-`.

## Dry Run

Use `--dry-run` before a real run:

```bash
pathvlm-litebench score-coordinate-heatmap-prompt-set \
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json \
  --dry-run
```

Dry-run mode:

- loads and validates the prompt-set config
- loads the coordinate manifest metadata
- applies `--max-images` truncation when provided
- prints each prompt run and resolved artifact path
- prints comparison CSV and Markdown paths
- does not load patch images
- does not create a model
- does not run inference
- does not write `scores.csv`, `heatmap.png`, `metadata.json`, `score_summary.csv`, or `score_summary.md`

Use `--max-images` for a small local smoke check:

```bash
pathvlm-litebench score-coordinate-heatmap-prompt-set \
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json \
  --dry-run \
  --max-images 16
```

## Run A Prompt Set

Run all prompts from the config:

```bash
pathvlm-litebench score-coordinate-heatmap-prompt-set \
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json
```

This command requires the patch image files to exist. Relative manifest paths are resolved against the manifest directory unless `image_root` is provided.

The command loads the selected CLIP-style model and runs model inference. It may download model weights if they are not already cached locally.

For each prompt, the command writes:

```text
outputs/patch_coordinate_heatmap_prompt_set/
|-- tumor/
|   |-- scores.csv
|   |-- heatmap.png
|   `-- metadata.json
|-- lymphocyte/
|   |-- scores.csv
|   |-- heatmap.png
|   `-- metadata.json
|-- score_summary.csv
`-- score_summary.md
```

Each per-prompt `scores.csv` includes one row per scored patch with coordinate metadata, the numeric score, and the prompt used for scoring.

Each per-prompt `metadata.json` records the prompt key, prompt text, model, device, manifest path, column names, output paths, patch count, toolkit version, colormap, title, and creation time.

The prompt-set comparison files are artifact-only summaries generated from the saved per-prompt `scores.csv` and `metadata.json` files:

- `score_summary.csv`: one row per prompt run with score distribution statistics.
- `score_summary.md`: compact Markdown summary for local review.

The comparison summary does not load models or patch images.

## CLI Overrides

The prompt-set command supports runtime overrides:

```bash
pathvlm-litebench score-coordinate-heatmap-prompt-set \
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json \
  --output-root outputs/patch_coordinate_heatmap_prompt_set_smoke \
  --comparison-output-csv outputs/patch_coordinate_heatmap_prompt_set_smoke/score_summary.csv \
  --comparison-output-md outputs/patch_coordinate_heatmap_prompt_set_smoke/score_summary.md \
  --max-images 16
```

Supported overrides:

- `--output-root`
- `--comparison-output-csv`
- `--comparison-output-md`
- `--max-images`

Use these overrides for smoke runs, alternate output locations, and local experiments without changing the committed example config.

## Comparing Existing Artifacts

If prompt-set artifacts already exist, compare them directly with the artifact-only command:

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

This path does not load patch images, create a model, run inference, or download weights.

## Model Loading And CI

`score-coordinate-heatmap-prompt-set` loads the selected CLIP-style model and runs model inference unless `--dry-run` is used. It may download model weights if they are not already cached locally.

CI covers prompt-set scoring with fake model wrappers, so CI tests do not download model weights.

CONCH remains optional and gated. This workflow does not make CONCH a required dependency and does not handle Hugging Face tokens.

## Output Interpretation

Use generated heatmaps as patch-coordinate score visualizations only. They show score variation across the coordinate grid represented by pre-extracted patches in the manifest.

Good wording:

- "This prompt assigned higher scores to these patch coordinates."
- "The comparison summary shows aggregate score differences across saved prompt runs."
- "The heatmap shows patch-level score variation across a coordinate grid."

Avoid:

- "This heatmap diagnoses a slide."
- "This is a whole-slide image model."
- "This identifies tumor boundaries."
