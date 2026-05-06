# Public Patch Benchmark Workflow

This guide shows a reproducible local workflow for running PathVLM-LiteBench on a public patch-level pathology dataset.

It uses MHIST as the concrete example because the repository already includes a converter preset, sampled-manifest commands, CLIP/PLIP/CONCH observations, and a zero-shot prompt-grid config. The same pattern applies to other patch datasets after converting annotations into the standard manifest format.

This workflow is for research and engineering review only. It is not a clinical benchmark or diagnostic workflow.

## Workflow Overview

Recommended path:

```text
download public patch dataset locally
-> convert annotations to standard manifest
-> sample a small balanced manifest
-> run a CLIP zero-shot baseline
-> run a model/prompt zero-shot grid
-> summarize and compare saved reports
-> record conservative model-behavior observations
```

Keep data and outputs local:

```text
dataset/   # local datasets, ignored by Git
outputs/   # generated reports, ignored by Git
```

Do not commit real patch images, patient-level metadata, model weights, generated reports, prompt-grid outputs, or comparison files.

## 1. Prepare Local Data

Expected MHIST-style local layout:

```text
dataset/
`-- MHIST/
    |-- annotations.csv
    |-- manifest.csv
    |-- manifest_test_50_per_class.csv
    `-- images/
        |-- MHIST_aaa.png
        |-- MHIST_aab.png
        `-- ...
```

The repository does not include MHIST or any other pathology dataset. Download datasets according to their own licenses and terms.

For other datasets, place files under a local ignored folder such as:

```text
dataset/<DATASET_NAME>/
```

## 2. Convert Annotations to a Standard Manifest

PathVLM-LiteBench uses a standard patch manifest format:

```csv
image_path,label,split,case_id,slide_id
patch_001.png,tumor,test,case_001,slide_001
patch_002.png,normal,test,case_001,slide_001
```

MHIST preset:

```bash
pathvlm-litebench convert-manifest \
  --preset mhist \
  --input dataset/MHIST/annotations.csv \
  --output dataset/MHIST/manifest.csv \
  --image_root dataset/MHIST/images \
  --require_exists
```

Generic conversion:

```bash
pathvlm-litebench convert-manifest \
  --input dataset/<DATASET_NAME>/annotations.csv \
  --output dataset/<DATASET_NAME>/manifest.csv \
  --path_column "Image Name" \
  --label_column "Majority Vote Label" \
  --split_column "Partition"
```

Use `--require_exists` when you want conversion to fail if an image path cannot be resolved.

## 3. Create a Small Balanced Sample

Start with a balanced sampled manifest before running full-split experiments:

```bash
pathvlm-litebench sample-manifest \
  --input dataset/MHIST/manifest.csv \
  --output dataset/MHIST/manifest_test_50_per_class.csv \
  --split test \
  --samples_per_label 50 \
  --seed 42
```

Why sample first:

- faster iteration on laptops
- lower GPU memory pressure
- easier error inspection
- less misleading raw accuracy when the original split is class-imbalanced

For a first pass, 20-100 patches per class is usually enough to validate the workflow.

## 4. Run a Single-Model Zero-Shot Baseline

Run CLIP on the sampled manifest:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest_test_50_per_class.csv \
  --image_root dataset/MHIST/images \
  --model clip \
  --device auto \
  --split test \
  --class_names HP SSA \
  --class_prompts \
    "a histopathology image of hyperplastic polyp" \
    "a histopathology image of sessile serrated adenoma" \
  --top_k 2 \
  --save_report \
  --report_dir outputs/zero_shot_clip_mhist_sample
```

Expected outputs:

```text
outputs/zero_shot_clip_mhist_sample/
|-- predictions.csv
|-- errors.csv
`-- metrics.json
```

Inspect `metrics.json` for:

- accuracy
- balanced accuracy
- macro-F1
- per-class precision, recall, and F1
- confusion matrix
- true and predicted label distributions
- prediction-collapse warning, if present

## 5. Run a Zero-Shot Prompt Grid

Preview the configured grid:

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json \
  --dry-run
```

Run the grid:

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json
```

The example config runs:

```text
models: clip, plip, conch
prompt pairs: default, diagnosis, patch
```

CONCH is optional and requires the official CONCH package plus gated Hugging Face access. If CONCH is unavailable in your environment, edit the config and remove `conch` from `models`.

Expected output layout:

```text
outputs/zero_shot_prompt_grid_mhist_sample/
|-- clip/
|   |-- default/
|   |-- diagnosis/
|   `-- patch/
|-- plip/
|   |-- default/
|   |-- diagnosis/
|   `-- patch/
|-- conch/
|   |-- default/
|   |-- diagnosis/
|   `-- patch/
`-- comparison.md
```

Each model/prompt directory contains the standard zero-shot report files plus `run.log` when logging is enabled.

For details on config fields and output interpretation, see [prompt_grid_workflow.md](prompt_grid_workflow.md).

## 6. Summarize or Compare Saved Reports

Summarize one zero-shot report:

```bash
pathvlm-litebench summarize-report \
  --task zero-shot \
  --report_dir outputs/zero_shot_clip_mhist_sample
```

Compare several saved reports:

```bash
pathvlm-litebench compare-reports \
  --task zero-shot \
  --report_dirs \
    outputs/zero_shot_prompt_grid_mhist_sample/clip/default \
    outputs/zero_shot_prompt_grid_mhist_sample/plip/default \
    outputs/zero_shot_prompt_grid_mhist_sample/conch/default \
  --run_names clip_default plip_default conch_default \
  --output outputs/zero_shot_default_model_comparison.md
```

These commands read saved CSV/JSON artifacts only. They do not rerun inference or download model weights.

## 7. Interpret Results Conservatively

For fine-grained pathology tasks, do not rely on raw accuracy alone.

Check:

- balanced accuracy
- macro-F1
- per-class recall
- confusion matrix
- predicted-label distribution
- prediction-collapse warning
- whether prompt wording changes the bias direction

Good wording:

- "This sampled run showed a strong predicted-label bias."
- "Changing prompt wording changed the model's predicted-label distribution."
- "The prompt grid remained close to chance-level balanced accuracy."

Avoid:

- "This model diagnoses HP or SSA."
- "This prompt proves the model is clinically reliable."
- "This local report is a certified benchmark result."

## 8. Scale Up Carefully

After the sampled workflow is stable:

1. Increase `samples_per_label`.
2. Run additional prompt pairs.
3. Compare CLIP, PLIP, and optional CONCH only when local model access is working.
4. Run a full test split if compute and dataset terms allow it.
5. Record observations in docs without committing generated outputs.

Scaling up should preserve the same output hygiene:

```text
dataset/   ignored
outputs/   ignored
```

## Related Docs

- [data_preparation.md](data_preparation.md)
- [small_dataset_quickstart.md](small_dataset_quickstart.md)
- [real_patch_workflow.md](real_patch_workflow.md)
- [prompt_grid_workflow.md](prompt_grid_workflow.md)
- [clip_plip_conch_mhist_sampled_observation.md](clip_plip_conch_mhist_sampled_observation.md)
- [clip_plip_conch_mhist_prompt_grid_observation.md](clip_plip_conch_mhist_prompt_grid_observation.md)
