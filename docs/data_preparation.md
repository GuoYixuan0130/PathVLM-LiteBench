# Data Preparation

PathVLM-LiteBench is designed to start from patch-level pathology images.

The current version does not include public pathology datasets or whole-slide images. Users should prepare their own patch image folder before running retrieval or zero-shot demos on real pathology data.

The toolkit is CPU-compatible and laptop-GPU friendly. Small smoke tests can run on CPU-only machines. If CUDA is available, patch embedding can be accelerated on consumer-grade laptop GPUs.

## Expected Input Format

The simplest input format is a folder containing patch images:

```text
your_patch_folder/
├── patch_001.png
├── patch_002.png
├── patch_003.png
├── patch_004.jpg
└── patch_005.tif
```

Supported image extensions include:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`
- `.tif`
- `.tiff`

All images are loaded as RGB PIL images.

## Manifest Format (Optional)

You can run demos directly from an image folder, or prepare a CSV manifest for structured datasets.

A recommended manifest includes columns such as:

- `image_path`
- `label`
- `split`
- `case_id`
- `slide_id`

Example:

```text
image_path,label,split,case_id
patches/patch_001.png,tumor,train,case_001
patches/patch_002.png,normal,train,case_001
patches/patch_003.png,necrosis,test,case_002
```

Manifest files are useful for downstream zero-shot evaluation, retrieval metrics, and split-aware benchmarking workflows.

Do not commit real pathology data or patient-level metadata to GitHub.

The manifest format can also be used by the zero-shot classification demo:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest path/to/manifest.csv \
  --image_root path/to/dataset_root \
  --model clip \
  --device auto \
  --split test
```

In this workflow:

- `label` can be used to infer class names and compute accuracy.
- `split` can be used to evaluate a target subset such as `test`.
- `case_id` and `slide_id` are useful metadata fields for future grouped analysis.

The same manifest labels can be used in the retrieval demo to compute text-to-image Recall@K:

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
  --top_k 5
```

`label_prompts` should align one-to-one with `prompts`, mapping each text prompt to its target manifest label.

## Patch-Level Workflow

The current toolkit works at patch level:

```text
patch image folder
→ load images
→ encode images with a frozen vision-language model
→ encode text prompts
→ compute similarity
→ retrieve top-k patches
→ optionally save visualization and HTML report
```

This design avoids full WSI processing in the early stage and keeps the workflow laptop-friendly.

## Compute and Device Usage

The toolkit is designed to be CPU-compatible and laptop-GPU friendly.

For small smoke tests, CPU execution is sufficient. For larger patch folders, CUDA acceleration is recommended when available:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --model clip \
  --device cuda \
  --image_dir path/to/your_patch_folder \
  --prompts "tumor region" "normal tissue" \
  --top_k 5 \
  --use_cache
```

If you are unsure, use:

```text
--device auto
```

This will use CUDA if available and fall back to CPU otherwise.

## Running Retrieval on Your Own Patches

Example:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --model clip \
  --device auto \
  --image_dir path/to/your_patch_folder \
  --prompts "tumor region" "normal tissue" "necrosis" \
  --top_k 5 \
  --use_cache \
  --save_visualization \
  --save_html_report
```

The results will be printed in the terminal. If visualization and HTML report options are enabled, outputs will be saved under:

```text
outputs/retrieval_demo/
```

For reproducible retrieval experiments, you can copy `configs/retrieval_demo_config.json`, edit `image_dir` and `prompts`, and run:

```bash
python examples/01_patch_text_retrieval_demo.py --config configs/retrieval_demo_config.json
```

## Running Zero-Shot Classification on Your Own Patches

Example:

```bash
python examples/02_zero_shot_classification_demo.py \
  --model clip \
  --device auto \
  --image_dir path/to/your_patch_folder \
  --class_names tumor normal necrosis \
  --class_prompts \
    "a histopathology image of tumor tissue" \
    "a histopathology image of normal tissue" \
    "a histopathology image showing necrosis" \
  --top_k 3
```

This compares each patch image embedding against class text prompt embeddings.

## Running Prompt Sensitivity Analysis

The current prompt sensitivity demo uses built-in prompt groups. It can still run on your own patch folder:

```bash
python examples/03_prompt_sensitivity_demo.py \
  --model clip \
  --device auto \
  --image_dir path/to/your_patch_folder \
  --top_k 5
```

This evaluates whether multiple prompt variants for the same concept retrieve similar top-k patches.

## Notes on Pathology Data

When using real pathology data:

- Make sure the data can be legally used for research.
- Do not commit private or patient-level data to GitHub.
- Do not commit large datasets, WSI files, model weights, or cached embeddings.
- Keep local datasets outside the repository when possible.
- Use `outputs/` for generated reports and cache files, which are ignored by Git.

## Recommended Patch Size

For CLIP-style models, common patch sizes include:

- `224 × 224`
- `256 × 256`
- `336 × 336`

The current CLIP baseline will internally preprocess images using its Hugging Face processor.

## About WSI Processing

Whole-slide image processing is intentionally not included in the current core workflow.

Future versions may include optional WSI-level features, such as:

- tissue mask generation
- patch sampling from WSI
- text-guided WSI heatmap generation

These features should remain optional to preserve the low-compute design philosophy.

## Demo Images

If no `--image_dir` is provided, the demos automatically generate simple RGB images under:

```text
examples/demo_patches/
```

These images are not pathology images. They are only smoke tests to verify that the pipeline runs correctly.
