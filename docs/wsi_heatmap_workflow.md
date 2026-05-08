# WSI-Oriented Patch Heatmap Workflow

This guide describes the lightweight WSI-oriented heatmap utilities introduced for the v0.7.0 milestone.

The workflow uses pre-extracted slide patches with coordinate metadata. PathVLM-LiteBench does not read WSI files, tile slides, run tissue detection, or render WSI pyramids.

## Local Data Layout

Keep slide-derived patches and generated outputs local:

```text
dataset/
`-- slide_patches/
    |-- coordinate_manifest.csv
    `-- patches/
        |-- patch_001.png
        |-- patch_002.png
        `-- ...

outputs/
`-- heatmap_demo/
```

Do not commit real pathology images, patient-level metadata, generated heatmaps, score CSV files, model weights, or embedding caches.

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
    "dataset/slide_patches/coordinate_manifest.csv",
    image_root="dataset/slide_patches",
)
```

This only resolves patch image paths and validates coordinate metadata. It does not load a WSI file.

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
    "outputs/heatmap_demo/tumor_prompt_heatmap.png",
    title="Tumor prompt score",
)

save_patch_scores_csv(
    records,
    scores,
    "outputs/heatmap_demo/tumor_prompt_scores.csv",
    prompt="a histopathology image of tumor tissue",
)
```

Repeated coordinates are averaged. Missing coordinate cells are left blank in the heatmap grid.

## Interpretation

Use these heatmaps as patch-coordinate score visualizations only.

Good wording:

- "This text prompt assigned higher scores to this region of pre-extracted patches."
- "The heatmap shows patch-level score variation across slide-derived coordinates."

Avoid:

- "This heatmap diagnoses a slide."
- "This is a clinical WSI model."
- "This identifies tumor boundaries."

## Current Scope

The current utilities provide:

- coordinate-aware manifest loading
- score aggregation over patch coordinates
- PNG heatmap export
- coordinate score CSV export
- offline tests without model inference

Future v0.7.0 work may add a config-driven demo, optional cached embeddings, and multi-prompt comparison utilities while keeping WSI readers optional and out of CI.
