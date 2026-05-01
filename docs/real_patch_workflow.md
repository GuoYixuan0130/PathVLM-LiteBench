# Real Patch Workflow

This document describes how to use PathVLM-LiteBench with a folder of real pathology patch images.

It assumes that you already have patch-level images prepared locally. The repository does not include pathology data.

## 1. Prepare a Patch Folder

Organize your patch images in a single folder:

```text
your_patch_folder/
|-- patch_001.png
|-- patch_002.png
|-- patch_003.jpg
|-- patch_004.tif
`-- patch_005.png
```

Supported extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`

Do not commit real pathology data to GitHub.

### Optional: Prepare a CSV Manifest

If you only run retrieval on a folder, an image directory is sufficient.

If you need dataset metadata (`label`, `split`, `case_id`, `slide_id`), prepare a CSV manifest:

```text
image_path,label,split,case_id
patches/patch_001.png,tumor,train,case_001
patches/patch_002.png,normal,train,case_001
patches/patch_003.png,necrosis,test,case_002
```

You can load it with:

```python
from pathvlm_litebench.data import load_patch_manifest, records_to_image_paths

records = load_patch_manifest("manifest.csv", image_root="path/to/dataset")
image_paths = records_to_image_paths(records)
```

If your dataset provides dataset-specific `annotations.csv`, convert it first:

```bash
pathvlm-litebench convert-manifest \
  --preset mhist \
  --input dataset/MHIST/annotations.csv \
  --output dataset/MHIST/manifest.csv \
  --image_root dataset/MHIST/images \
  --require_exists
```

Then run retrieval and zero-shot demos with `dataset/MHIST/manifest.csv`.

`dataset/` is a local dataset directory and is ignored by Git. Do not commit real pathology images.

### Optional: Create a Small Balanced Manifest

For quick laptop-friendly testing, you can sample a smaller balanced subset first:

```bash
pathvlm-litebench sample-manifest \
  --input dataset/MHIST/manifest.csv \
  --output dataset/MHIST/manifest_test_50_per_class.csv \
  --split test \
  --samples_per_label 50 \
  --seed 42
```

It is usually faster to validate your workflow on a sampled manifest first, then run the full test split.

## 2. Choose Text Prompts

You can start with pathology-style prompts such as:

- `a histopathology image of tumor tissue`
- `a histopathology image of normal tissue`
- `a histopathology image showing necrosis`
- `a pathology patch with inflammatory cells`
- `a pathology patch showing lymphocyte infiltration`

You can also use the built-in prompt template library:

```python
from pathvlm_litebench.prompts import get_prompt_variants

tumor_prompts = get_prompt_variants("tumor")
```

## 3. Run Patch-Text Retrieval

Example command:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --model clip \
  --device auto \
  --image_dir path/to/your_patch_folder \
  --prompts \
    "a histopathology image of tumor tissue" \
    "a histopathology image of normal tissue" \
    "a histopathology image showing necrosis" \
  --top_k 5 \
  --use_cache \
  --save_visualization \
  --save_html_report
```

`--device auto` uses CUDA if available and falls back to CPU otherwise.

`--use_cache` saves image embeddings so that repeated runs do not need to re-encode all patches.

Manifest-based retrieval evaluation with Recall@K:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --manifest path/to/manifest.csv \
  --image_root path/to/dataset_root \
  --model clip \
  --device auto \
  --split test \
  --prompts \
    "a histopathology image of tumor tissue" \
    "a histopathology image of normal tissue" \
    "a histopathology image showing necrosis" \
  --label_prompts tumor normal necrosis \
  --recall_k 1 5 10 \
  --top_k 5 \
  --save_html_report
```

`label_prompts` maps each text prompt to a manifest label. When labels are available, the retrieval demo computes text-to-image Recall@K automatically.

When both manifest labels and `label_prompts` are available, the HTML retrieval report also shows each retrieved patch's label, target label, and match status (`yes`/`no`). This makes it easier to inspect not only ranking quality but also class-consistent retrieval behavior.

Use `--save_report` to persist structured retrieval outputs:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --manifest path/to/manifest.csv \
  --image_root path/to/dataset_root \
  --model clip \
  --device auto \
  --split test \
  --prompts \
    "a histopathology image of tumor tissue" \
    "a histopathology image of normal tissue" \
  --label_prompts tumor normal \
  --top_k 5 \
  --save_report \
  --report_dir outputs/retrieval_demo
```

`retrieval_results.csv` helps inspect top-k matches for each prompt, and `retrieval_metrics.json` records Recall@K and experiment parameters for reproducibility. Do not commit `outputs/`.

## 4. Inspect Outputs

By default, generated outputs are saved under:

```text
outputs/retrieval_demo/
```

Typical outputs:

```text
outputs/
|-- cache/
|   |-- image_embeddings.pt
|   `-- image_paths.json
`-- retrieval_demo/
    |-- retrieval_report.html
    `-- topk_prompt_*.png
```

Open `retrieval_report.html` in a browser to inspect the retrieved patches for each prompt.

## 5. Use a JSON Config for Reproducibility

You can copy the example config:

```text
configs/retrieval_demo_config.json
```

Edit fields such as:

- `image_dir`
- `prompts`
- `top_k`
- `model`
- `device`
- `use_cache`
- `save_visualization`
- `save_html_report`

Then run:

```bash
python examples/01_patch_text_retrieval_demo.py --config configs/retrieval_demo_config.json
```

Command-line arguments can override config values.

For example:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --config configs/retrieval_demo_config.json \
  --top_k 3 \
  --device cuda
```

## 6. Run Prompt Sensitivity Analysis

To evaluate how different prompt formulations affect retrieval results:

```bash
python examples/03_prompt_sensitivity_demo.py \
  --model clip \
  --device auto \
  --image_dir path/to/your_patch_folder \
  --use_pathology_prompts \
  --concepts tumor normal necrosis \
  --top_k 5
```

This uses built-in prompt variants for each concept.

For example, the concept `tumor` includes multiple prompts such as:

- `a histopathology image of tumor tissue`
- `a pathology patch showing malignant tissue`
- `a microscopic image of cancerous tissue`
- `a H&E stained tissue patch with tumor region`

The prompt sensitivity module compares whether these variants retrieve similar top-k patches.

Use `--save_report` to save structured prompt sensitivity outputs:

```bash
python examples/03_prompt_sensitivity_demo.py \
  --model clip \
  --device auto \
  --image_dir path/to/your_patch_folder \
  --use_pathology_prompts \
  --concepts tumor normal necrosis \
  --top_k 5 \
  --save_report \
  --report_dir outputs/prompt_sensitivity_demo
```

`prompt_sensitivity_summary.csv` is for concept-level `mean_topk_overlap` and `mean_similarity_std`. `prompt_sensitivity_details.csv` stores top-k retrieval rows for each prompt variant. `prompt_sensitivity_metrics.json` records full results and experiment metadata. Do not commit `outputs/`.

Prompt sensitivity also supports config-driven runs:

```bash
python examples/03_prompt_sensitivity_demo.py --config configs/prompt_sensitivity_demo_config.json
```

Update `image_dir` and `concepts` in the config based on your dataset. Config-driven runs help keep experiments reproducible across machines and reruns.

## 7. Run Zero-Shot Classification

You can also run zero-shot patch classification:

```bash
python examples/02_zero_shot_classification_demo.py \
  --model clip \
  --device auto \
  --image_dir path/to/your_patch_folder \
  --class_names tumor normal necrosis \
  --top_k 3
```

If no custom `--class_prompts` are provided, the demo builds simple pathology-style prompts automatically.

Manifest-based zero-shot evaluation:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest path/to/manifest.csv \
  --image_root path/to/dataset_root \
  --model clip \
  --device auto \
  --split test \
  --top_k 3
```

If labels are available in the manifest, the demo prints a fuller classification report including accuracy, balanced accuracy, macro-F1, and confusion matrix. If labels are missing, you can still inspect predictions but classification metrics are skipped.

These metrics are for lightweight model behavior analysis and benchmarking only, not clinical validation.

Use `--save_report` to save zero-shot outputs for later analysis:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest path/to/manifest.csv \
  --image_root path/to/dataset_root \
  --model clip \
  --device auto \
  --split test \
  --class_names tumor normal necrosis \
  --top_k 3 \
  --save_report \
  --report_dir outputs/zero_shot_demo
```

This writes `predictions.csv`, `errors.csv`, and `metrics.json`, which are useful for experiment tracking and downstream analysis. Do not commit `outputs/`.

If the model predicts almost everything as one class, `metrics.json` includes a warning message about possible prediction collapse or class bias. `errors.csv` helps you quickly inspect misclassified samples. For CLIP baselines on fine-grained pathology tasks, this kind of class bias can happen and does not imply a toolkit bug.

## 8. Interpreting Results

When inspecting results, consider:

- Are retrieved patches visually coherent for each prompt?
- Do prompt variants for the same concept retrieve similar images?
- Are there prompts that produce unstable or unexpected results?
- Does the general CLIP baseline struggle with pathology-specific concepts?
- Would a pathology-specific VLM such as PLIP or CONCH be useful for comparison?

Do not interpret these outputs as clinical diagnostic results.

## 9. Compute Notes

For small folders, CPU may be sufficient.

For larger patch folders, CUDA is recommended:

```text
--device cuda
```

The toolkit is designed for consumer-grade laptop GPUs and does not require large-scale training hardware.

## 10. Data Safety

Do not commit:

- real pathology images
- patient-level metadata
- model weights
- embedding cache files
- generated outputs

The following folders are ignored by Git:

```text
dataset/
outputs/
examples/demo_patches/
```

## Summary

This workflow demonstrates how to use PathVLM-LiteBench for real patch-level CPath VLM evaluation:

```text
patch folder
-> text prompts
-> frozen model embeddings
-> retrieval / zero-shot / prompt sensitivity
-> visualization and reports
```

This is intended for research and educational use only.
