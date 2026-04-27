# Real Patch Workflow

This document describes how to use PathVLM-LiteBench with a folder of real pathology patch images.

It assumes that you already have patch-level images prepared locally. The repository does not include pathology data.

## 1. Prepare a Patch Folder

Organize your patch images in a single folder:

```text
your_patch_folder/
├── patch_001.png
├── patch_002.png
├── patch_003.jpg
├── patch_004.tif
└── patch_005.png
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

records = load_patch_manifest("manifest.csv", image_root="path/to/data")
image_paths = records_to_image_paths(records)
```

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

## 4. Inspect Outputs

By default, generated outputs are saved under:

```text
outputs/retrieval_demo/
```

Typical outputs:

```text
outputs/
├── cache/
│   ├── image_embeddings.pt
│   └── image_paths.json
└── retrieval_demo/
    ├── retrieval_report.html
    └── topk_prompt_*.png
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

If labels are available in the manifest, the demo can compute accuracy. If labels are missing, you can still inspect predictions but accuracy is skipped.

## 8. Suggested Interpretation

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
outputs/
examples/demo_patches/
data/
```

## Summary

This workflow demonstrates how to use PathVLM-LiteBench for real patch-level CPath VLM evaluation:

```text
patch folder
→ text prompts
→ frozen model embeddings
→ retrieval / zero-shot / prompt sensitivity
→ visualization and reports
```

This is intended for research and educational use only.
