# Patch Coordinate Heatmap Workflow

This guide describes patch-coordinate heatmap utilities introduced for the v0.7.0 milestone.

The workflow uses pre-extracted patches with coordinate metadata. PathVLM-LiteBench does not read whole-slide image files, tile slides, run tissue detection, or render slide pyramids.

## Local Data Layout

Keep patch images and generated outputs local:

```text
dataset/
`-- patch_coordinates/
    |-- coordinate_manifest.csv
    `-- patches/
        |-- patch_001.png
        |-- patch_002.png
        `-- ...

outputs/
`-- patch_coordinate_heatmap_demo/
```

Do not commit real pathology images, patient-level metadata, generated heatmaps, score CSV files, model weights, or embedding caches.

## Synthetic Demo

To verify the artifact workflow without real pathology data or model inference:

```bash
python examples/05_patch_coordinate_heatmap_demo.py
```

This writes synthetic colored patches, a coordinate manifest, a score CSV, and a heatmap under:

```text
outputs/patch_coordinate_heatmap_demo_synthetic/
```

The generated patches are simple colored tiles for smoke testing only.

## Coordinate Manifest

The coordinate-aware manifest requires `image_path`, `x`, and `y` columns:

```csv
image_path,x,y,width,height,label,slide_id
patches/patch_001.png,0,0,224,224,tumor,slide_a
patches/patch_002.png,224,0,224,224,normal,slide_a
patches/patch_003.png,0,224,224,224,tumor,slide_a
```

Supported optional columns include:

- `width`
- `height`
- `label`
- `split`
- `case_id`
- `slide_id`

Additional non-empty columns are preserved as metadata.

## Load Coordinate Records

```python
from pathvlm_litebench.data import load_coordinate_patch_manifest

records = load_coordinate_patch_manifest(
    "dataset/patch_coordinates/coordinate_manifest.csv",
    image_root="dataset/patch_coordinates",
)
```

This only resolves patch image paths and validates coordinate metadata. It does not load or process a whole-slide image.

## Save a Heatmap from Precomputed Scores

If you already have one score per patch, aggregate scores by patch coordinates and save a heatmap:

```python
from pathvlm_litebench.visualization import (
    aggregate_patch_scores_to_grid,
    save_patch_scores_csv,
    save_score_heatmap,
)

scores = [0.12, 0.84, 0.43]  # one score per coordinate record

grid = aggregate_patch_scores_to_grid(records, scores)

save_score_heatmap(
    grid,
    "outputs/patch_coordinate_heatmap_demo/tumor_prompt_heatmap.png",
    title="Tumor prompt score",
)

save_patch_scores_csv(
    records,
    scores,
    "outputs/patch_coordinate_heatmap_demo/tumor_prompt_scores.csv",
    prompt="a histopathology image of tumor tissue",
)
```

Repeated coordinates are averaged. Missing coordinate cells are left blank in the heatmap grid.

## Render a Heatmap from Existing Artifacts

The CLI can render a patch-coordinate heatmap from a coordinate manifest and an existing score CSV:

```bash
pathvlm-litebench render-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --score-csv outputs/patch_coordinate_heatmap_demo/tumor_prompt_scores.csv \
  --output outputs/patch_coordinate_heatmap_demo/tumor_prompt_heatmap.png
```

By default, scores are aligned to manifest rows by `image_path`. The score CSV must include:

```csv
image_path,score
patches/patch_001.png,0.12
patches/patch_002.png,0.84
patches/patch_003.png,0.43
```

If the score CSV has exactly the same row order as the manifest, use order-based alignment:

```bash
pathvlm-litebench render-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --score-csv outputs/patch_coordinate_heatmap_demo/tumor_prompt_scores.csv \
  --align-by order \
  --output outputs/patch_coordinate_heatmap_demo/tumor_prompt_heatmap.png
```

This command reads saved artifacts only. It does not load a model, run inference, or download weights.

You can also validate and use the example config:

```bash
pathvlm-litebench validate-config \
  configs/patch_coordinate_heatmap_demo_config.json

pathvlm-litebench render-coordinate-heatmap \
  --config configs/patch_coordinate_heatmap_demo_config.json
```

Command-line arguments such as `--output`, `--score-column`, and `--align-by` can override config values at runtime.

## Score Patches from a Text Prompt

If you want PathVLM-LiteBench to compute the patch scores, use:

```bash
pathvlm-litebench score-coordinate-heatmap \
  --manifest dataset/patch_coordinates/coordinate_manifest.csv \
  --prompt "a histopathology image of tumor tissue" \
  --output-dir outputs/patch_coordinate_heatmap_scored \
  --model clip \
  --device auto
```

You can also validate and use the prompt-scored heatmap example config:

```bash
pathvlm-litebench validate-config \
  configs/patch_coordinate_heatmap_scoring_demo_config.json

pathvlm-litebench score-coordinate-heatmap \
  --config configs/patch_coordinate_heatmap_scoring_demo_config.json
```

Command-line arguments such as `--prompt`, `--output-dir`, `--score-csv`, `--heatmap-output`, `--model`, `--device`, and `--max-images` can override config values at runtime.

This command:

- loads patch images listed in the coordinate manifest
- loads the selected CLIP-style model
- encodes one text prompt
- computes one image-text similarity score per patch
- writes `scores.csv`
- writes `heatmap.png`

Unlike `render-coordinate-heatmap`, this command runs model inference and may download model weights if they are not already cached locally. CI covers this path with fake models and does not download weights.

## Interpretation

Use these heatmaps as patch-coordinate score visualizations only.

Good wording:

- "This text prompt assigned higher scores to these patch coordinates."
- "The heatmap shows patch-level score variation across a coordinate grid."

Avoid:

- "This heatmap diagnoses a slide."
- "This is a whole-slide image model."
- "This identifies tumor boundaries."

## Current Scope

The current utilities provide:

- coordinate-aware manifest loading
- score aggregation over patch coordinates
- PNG heatmap export
- coordinate score CSV export
- artifact-only heatmap rendering from the CLI
- offline tests without model inference
- prompt-scored patch-coordinate heatmap rendering from the CLI
- config validation for artifact-only and prompt-scored patch-coordinate heatmap workflows
