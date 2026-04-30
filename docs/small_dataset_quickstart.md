# Small Dataset Quickstart

This guide shows the shortest recommended path for running PathVLM-LiteBench on a small local pathology patch dataset.

It uses MHIST as the concrete example, but the same workflow applies to other patch-level datasets after converting their annotations into the standard manifest format.

## Goal

Run three lightweight CLIP baseline workflows:

- patch-text retrieval
- zero-shot classification
- prompt sensitivity analysis

The recommended first pass uses a balanced sampled manifest so that experiments remain laptop-friendly.

## Local Folder Layout

Keep real data under `dataset/`, which is ignored by Git:

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

Do not commit real pathology images, generated reports, or sampled manifests.

## 1. Convert Dataset Annotations

MHIST provides `annotations.csv`. Convert it into the standard manifest used by the demos:

```bash
pathvlm-litebench convert-manifest \
  --preset mhist \
  --input dataset/MHIST/annotations.csv \
  --output dataset/MHIST/manifest.csv \
  --image_root dataset/MHIST/images \
  --require_exists
```

The output manifest uses standard columns:

```text
image_path,label,split,case_id
MHIST_aaa.png,SSA,train,MHIST_aaa
MHIST_aab.png,HP,train,MHIST_aab
```

## 2. Sample a Small Balanced Manifest

Start with a small balanced test subset before running the full split:

```bash
pathvlm-litebench sample-manifest \
  --input dataset/MHIST/manifest.csv \
  --output dataset/MHIST/manifest_test_50_per_class.csv \
  --split test \
  --samples_per_label 50 \
  --seed 42
```

For laptop development, 20-100 patches per class is usually enough to validate the workflow quickly.

## 3. Run Retrieval

Use label prompts so the demo can compute Recall@K:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --manifest dataset/MHIST/manifest_test_50_per_class.csv \
  --image_root dataset/MHIST/images \
  --model clip \
  --device auto \
  --split test \
  --prompts \
    "a histopathology image of hyperplastic polyp" \
    "a histopathology image of sessile serrated adenoma" \
  --label_prompts HP SSA \
  --recall_k 1 5 10 \
  --top_k 5 \
  --use_cache \
  --save_visualization \
  --save_html_report \
  --save_report \
  --report_dir outputs/retrieval_demo
```

Expected outputs include:

- `outputs/retrieval_demo/retrieval_results.csv`
- `outputs/retrieval_demo/retrieval_metrics.json`
- `outputs/retrieval_demo/retrieval_report.html`
- `outputs/retrieval_demo/retrieval_report_assets/`

The HTML report copies images into `retrieval_report_assets/` so the report does not depend on absolute local image paths.

## 4. Run Zero-Shot Classification

Run the same sampled manifest through the zero-shot classification demo:

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
  --report_dir outputs/zero_shot_demo
```

Expected outputs include:

- `outputs/zero_shot_demo/predictions.csv`
- `outputs/zero_shot_demo/errors.csv`
- `outputs/zero_shot_demo/metrics.json`

For fine-grained pathology tasks, do not rely on accuracy alone. Check balanced accuracy, macro-F1, per-class recall, the confusion matrix, and prediction distribution.

## 5. Run Prompt Sensitivity Analysis

Prompt sensitivity currently works from an image folder rather than a manifest:

```bash
python examples/03_prompt_sensitivity_demo.py \
  --model clip \
  --device auto \
  --image_dir dataset/MHIST/images \
  --use_pathology_prompts \
  --concepts tumor normal necrosis \
  --top_k 5 \
  --save_report \
  --report_dir outputs/prompt_sensitivity_demo
```

Expected outputs include:

- `outputs/prompt_sensitivity_demo/prompt_sensitivity_summary.csv`
- `outputs/prompt_sensitivity_demo/prompt_sensitivity_details.csv`
- `outputs/prompt_sensitivity_demo/prompt_sensitivity_metrics.json`

## 6. Scale Up Carefully

After the sampled run works, you can replace `manifest_test_50_per_class.csv` with `manifest.csv` for a full test split run.

Recommended order:

1. Run CPU or CUDA smoke tests on generated demo images.
2. Convert the real dataset manifest.
3. Run a balanced sampled manifest.
4. Inspect reports and metrics.
5. Run the full test split only after the workflow is validated.

## Notes

- Use `--device cuda` when a CUDA GPU is available and you want faster patch embedding.
- Use `--device auto` for portable commands that run on both CPU-only and CUDA machines.
- `dataset/`, `outputs/`, and `examples/demo_patches/` are ignored by Git.
- General CLIP can perform poorly on fine-grained pathology labels. This is expected for a baseline and motivates comparison with pathology-specific VLMs such as PLIP.
- PathVLM-LiteBench is for research and educational use only, not clinical diagnosis.
