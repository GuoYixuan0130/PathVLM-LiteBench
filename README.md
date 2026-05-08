# PathVLM-LiteBench

[![CI](https://github.com/GuoYixuan0130/PathVLM-LiteBench/actions/workflows/ci.yml/badge.svg)](https://github.com/GuoYixuan0130/PathVLM-LiteBench/actions/workflows/ci.yml)

PathVLM-LiteBench is a lightweight, CPU-compatible and laptop-GPU accelerated toolkit for benchmarking and visualizing vision-language models in computational pathology under limited computing resources.

This project focuses on patch-level pathology image-text retrieval, zero-shot classification, prompt sensitivity analysis, embedding caching, and visualization reports using frozen CLIP/PLIP-style vision-language models, with CUDA acceleration when available.

## Motivation

Recent computational pathology foundation models and vision-language models have shown strong potential in histopathology image understanding. However, reproducing large-scale pretraining is computationally expensive and often infeasible for students or small research groups.

PathVLM-LiteBench aims to provide a low-compute and reproducible toolkit for evaluating and comparing pathology vision-language models using frozen models and lightweight utilities.

Instead of training large models from scratch, this project focuses on:

- Frozen vision-language model inference
- Patch-level image-text retrieval
- Zero-shot classification
- Prompt sensitivity analysis
- Embedding caching
- Visualization and HTML reporting
- CPU-compatible smoke tests and laptop-friendly experimentation
- CUDA-accelerated inference on consumer-grade laptop GPUs when available

The project does not require A100 GPUs, multi-GPU training, or large-scale VLM pretraining.

## Current Stage

The current version supports a minimal but complete patch-level workflow:

- Loading patch images from a folder
- Encoding images with CLIP-style models
- Encoding text prompts with CLIP-style models
- Computing image-text similarity
- Retrieving top-k images for each prompt
- Saving top-k visualization grids
- Generating HTML retrieval reports
- Caching image embeddings
- Running zero-shot classification
- Running prompt sensitivity analysis

The current demos use simple RGB images for smoke testing. These demo images are not pathology images.

## Example Observation on MHIST

A local full-test MHIST zero-shot comparison showed opposite class-bias patterns: CLIP favored `HP`, while PLIP favored `SSA`. CLIP had higher raw accuracy because it predicted the majority `HP` class for most samples, while PLIP improved balanced accuracy and `SSA` recall but still showed strong class bias.

This is a useful baseline observation rather than a clinical result: it shows why accuracy alone can be misleading for fine-grained pathology tasks and why balanced accuracy, macro-F1, confusion matrices, predicted-label distributions, error reports, and prompt-sensitivity checks are included.

See [docs/mhist_baseline_observation.md](docs/mhist_baseline_observation.md) for the setup, observed metrics, and interpretation.
For the full-test CLIP vs PLIP comparison with prompt-sensitivity checks, see [docs/clip_vs_plip_mhist_observation.md](docs/clip_vs_plip_mhist_observation.md).
For a post-v0.4.0 sampled CLIP vs PLIP vs CONCH comparison, see [docs/clip_plip_conch_mhist_sampled_observation.md](docs/clip_plip_conch_mhist_sampled_observation.md).
For a sampled CLIP/PLIP/CONCH prompt-grid comparison, see [docs/clip_plip_conch_mhist_prompt_grid_observation.md](docs/clip_plip_conch_mhist_prompt_grid_observation.md).

## Model Registry

PathVLM-LiteBench uses a lightweight model registry so that demos can accept either a short model key or a Hugging Face model name.

Currently supported model keys:

| Model key | Resolved model name | Status | Notes |
|---|---|---|---|
| `clip` | `openai/clip-vit-base-patch32` | Implemented | Default CLIP baseline |
| `clip-vit-base-patch32` | `openai/clip-vit-base-patch32` | Implemented | Alias for CLIP ViT-B/32 |
| `plip` | `vinid/plip` | Implemented | Pathology-specific CLIP-compatible PLIP wrapper |
| `conch` | `MahmoodLab/CONCH` | Optional | CONCH wrapper; requires gated Hugging Face access and the optional CONCH package |

You can run demos with a registered model key:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip
```

or with a Hugging Face model name:

```bash
python examples/01_patch_text_retrieval_demo.py --model openai/clip-vit-base-patch32
```

PLIP is supported through a CLIP-compatible Hugging Face wrapper. CONCH is supported through the official `mahmoodlab/CONCH` package and requires gated Hugging Face access. If the optional package or authentication is missing, `--model conch` raises a clear setup error.

To use CONCH locally, first install its optional package and authenticate with Hugging Face:

```bash
pip install git+https://github.com/Mahmoodlab/CONCH.git
hf auth login
```

## Device Support

PathVLM-LiteBench supports explicit device selection for model-based demos:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto
python examples/01_patch_text_retrieval_demo.py --model clip --device cuda
python examples/01_patch_text_retrieval_demo.py --model clip --device cpu
```

`auto` is the default mode. It uses CUDA if available and falls back to CPU otherwise.

This design keeps the toolkit compatible with CPU-only environments while allowing faster patch embedding on consumer-grade laptop GPUs.

The retrieval-metrics demo (`examples/04_retrieval_metrics_demo.py`) does not use a model and therefore does not need a `--device` argument.

## Prompt Templates

PathVLM-LiteBench includes a small built-in prompt template library for common pathology concepts such as `tumor`, `normal`, `necrosis`, `inflammation`, `stroma`, `lymphocyte`, and `gland`.

These templates are intended for lightweight experimentation with zero-shot classification and prompt sensitivity analysis.

Example:

```python
from pathvlm_litebench.prompts import get_prompt_variants, build_class_prompts

tumor_prompts = get_prompt_variants("tumor")
class_prompts = build_class_prompts(["tumor", "normal", "necrosis"])
```

The prompt library is not a clinical ontology. It is a lightweight research utility for reproducible prompt experiments.

Zero-shot with built-in class prompt generation:

```bash
python examples/02_zero_shot_classification_demo.py --model clip --device auto --class_names tumor normal necrosis --top_k 2
```

Prompt sensitivity with pathology prompt templates:

```bash
python examples/03_prompt_sensitivity_demo.py --model clip --device auto --use_pathology_prompts --concepts tumor normal necrosis --top_k 2
```

If no real pathology patch folder is passed, the demos still use generated RGB images as smoke tests. For meaningful CPath experiments, provide `--image_dir path/to/your_patch_folder`.

## Benchmark Configuration

PathVLM-LiteBench supports lightweight JSON configuration files for reproducible experiments.

Example retrieval config:

```text
configs/retrieval_demo_config.json
```

Run retrieval demo with config:

```bash
python examples/01_patch_text_retrieval_demo.py --config configs/retrieval_demo_config.json
```

JSON config can store model, device, image_dir, prompts, top_k, cache, visualization, and report settings. Command-line arguments can override config values.

Zero-shot classification also supports config-driven runs:

```bash
python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_demo_config.json
```

Prompt sensitivity analysis also supports config-driven runs:

```bash
python examples/03_prompt_sensitivity_demo.py --config configs/prompt_sensitivity_demo_config.json
```

Example config:

```json
{
  "task": "prompt_sensitivity",
  "model": "clip",
  "device": "auto",
  "image_dir": "dataset/MHIST/images",
  "top_k": 5,
  "use_pathology_prompts": true,
  "concepts": ["tumor", "normal", "necrosis"],
  "save_report": true,
  "report_dir": "outputs/prompt_sensitivity_demo"
}
```

The `dataset/` folder is local and ignored by Git. The config file is a template and does not include data.

Manifest-specific retrieval evaluation options such as `--manifest`, `--image_root`, `--split`, `--label_prompts`, and `--recall_k` are currently passed via CLI.

## Patch Manifest Support

PathVLM-LiteBench can also read CSV manifests for real patch datasets.

Example manifest:

```text
image_path,label,split,case_id
patches/patch_001.png,tumor,train,case_001
patches/patch_002.png,normal,train,case_001
patches/patch_003.png,necrosis,test,case_002
```

Example usage:

```python
from pathvlm_litebench.data import load_patch_manifest, records_to_image_paths, get_unique_labels

records = load_patch_manifest("manifest.csv", image_root="path/to/dataset")
image_paths = records_to_image_paths(records)
labels = get_unique_labels(records)
```

The manifest loader does not load images or run models. It only prepares structured metadata for downstream retrieval, classification, and evaluation workflows.

## Manifest Conversion

PathVLM-LiteBench uses a standard patch manifest format:

```csv
image_path,label,split,case_id,slide_id
patch_001.png,tumor,test,case_001,slide_001
patch_002.png,normal,test,case_001,slide_001
```

For command-line demos, users are encouraged to convert dataset-specific annotation files into this standard format.

Example: convert MHIST annotations:

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
  --input annotations.csv \
  --output manifest.csv \
  --path_column "Image Name" \
  --label_column "Majority Vote Label" \
  --split_column "Partition"
```

The local `dataset/` folder is intended for private local datasets and is ignored by Git.

## Manifest Sampling

For quick low-compute experiments, users can create a smaller balanced manifest:

```bash
pathvlm-litebench sample-manifest \
  --input dataset/MHIST/manifest.csv \
  --output dataset/MHIST/manifest_test_50_per_class.csv \
  --split test \
  --samples_per_label 50 \
  --seed 42
```

This does not copy images. It only creates a smaller CSV manifest.

Then run retrieval or zero-shot evaluation with:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --manifest dataset/MHIST/manifest_test_50_per_class.csv \
  --image_root dataset/MHIST/images \
  --model clip \
  --device auto
```

This is useful for laptop-friendly testing and balanced class evaluation. `dataset/` is local and ignored by Git.

## Repository Structure

```text
PathVLM-LiteBench/
|-- pathvlm_litebench/
|   |-- data/
|   |-- models/
|   |-- retrieval/
|   |-- evaluation/
|   |-- prompts/
|   `-- visualization/
|-- examples/
|   |-- 01_quick_start.ipynb
|   |-- 01_patch_text_retrieval_demo.py
|   |-- 02_zero_shot_classification_demo.py
|   |-- 03_prompt_sensitivity_demo.py
|   `-- 04_retrieval_metrics_demo.py
|-- configs/
|-- docs/
|-- tests/
|-- README.md
`-- pyproject.toml
```

## Installation

Clone the repository:

```bash
git clone https://github.com/GuoYixuan0130/PathVLM-LiteBench.git
cd PathVLM-LiteBench
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Command-Line Interface

After editable installation:

```bash
pip install -e .
```

You can inspect the toolkit with:

```bash
pathvlm-litebench version
pathvlm-litebench models
pathvlm-litebench demos
pathvlm-litebench convert-manifest --help
pathvlm-litebench validate-config --help
pathvlm-litebench run-zero-shot-grid --help
```

The CLI does not download models by default. It only lists registry information and available demo commands.

## Data Preparation

To run PathVLM-LiteBench on real pathology patches, prepare a folder of patch-level images and pass it with `--image_dir`.

For the shortest MHIST-style workflow, see [docs/small_dataset_quickstart.md](docs/small_dataset_quickstart.md).
See [docs/data_preparation.md](docs/data_preparation.md) for details.
For a step-by-step guide on running the toolkit with real pathology patch folders, see [docs/real_patch_workflow.md](docs/real_patch_workflow.md).
For an end-to-end public patch benchmark recipe using conversion, sampling, zero-shot, prompt grids, and report comparison, see [docs/public_patch_benchmark_workflow.md](docs/public_patch_benchmark_workflow.md).

## Quick Start

### Demo 1: Patch-Text Retrieval

Run the minimal patch-text retrieval demo:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto
```

If no image folder is provided, the script automatically creates a small demo folder with simple RGB images. These demo images are only used to verify that the pipeline works end-to-end.

### Demo 2: Zero-Shot Classification

Run the zero-shot classification demo:

```bash
python examples/02_zero_shot_classification_demo.py --model clip --device auto
```

Run zero-shot evaluation from a CSV manifest:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest path/to/manifest.csv \
  --image_root path/to/dataset_root \
  --model clip \
  --device auto \
  --split test \
  --top_k 3
```

If labels are available in the manifest, the demo reports:
- accuracy
- balanced accuracy
- macro precision / recall / F1
- per-class precision / recall / F1
- confusion matrix

If `class_names` are not provided, they can be inferred from manifest labels. This is intended for lightweight patch-level evaluation, not clinical diagnosis.

Example MHIST-style manifest evaluation command:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest.csv \
  --image_root dataset/MHIST/images \
  --model clip \
  --device auto \
  --split test \
  --class_names HP SSA \
  --class_prompts \
    "a histopathology image of hyperplastic polyp" \
    "a histopathology image of sessile serrated adenoma" \
  --top_k 2
```

The current CLIP baseline is not pathology-specific, so low performance is expected and should be interpreted as a baseline.

This demo classifies each patch by comparing its image embedding with class text prompt embeddings.

To save predictions and metrics:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest.csv \
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

This generates:

- `predictions.csv`
- `errors.csv`
- `metrics.json`

`predictions.csv` stores per-image predictions. `metrics.json` stores aggregate classification metrics when labels are available.

When labels are available, `--save_report` also generates:

- `errors.csv`: misclassified samples only
- prediction distribution summary in `metrics.json`
- true label distribution
- predicted label distribution
- error rate
- optional warning when predictions collapse into one class

This is useful for detecting class bias in general CLIP baselines.

To generate a compact Markdown summary from a saved zero-shot report:

```bash
pathvlm-litebench summarize-report \
  --task zero-shot \
  --report_dir outputs/zero_shot_demo
```

This writes `experiment_summary.md` in the report directory. The summary is generated from saved artifacts only; it does not run model inference or download model weights.

### Demo 3: Prompt Sensitivity Analysis

Run the prompt sensitivity demo:

```bash
python examples/03_prompt_sensitivity_demo.py --model clip --device auto
```

This demo evaluates whether different prompt variants for the same concept retrieve similar top-k image results.

To save prompt sensitivity results:

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

This generates:

- `prompt_sensitivity_summary.csv`
- `prompt_sensitivity_details.csv`
- `prompt_sensitivity_metrics.json`

The summary CSV stores concept-level stability metrics. The details CSV stores prompt-level top-k retrieved indices and scores. The JSON stores full results and experiment metadata.

To generate a compact Markdown summary from a saved prompt sensitivity report:

```bash
pathvlm-litebench summarize-report \
  --task prompt-sensitivity \
  --report_dir outputs/prompt_sensitivity_demo
```

This writes `experiment_summary.md` in the report directory without rerunning model inference.

Prompt sensitivity analysis can also be configured with JSON:

```bash
python examples/03_prompt_sensitivity_demo.py --config configs/prompt_sensitivity_demo_config.json
```

Command-line arguments can override config values. For meaningful analysis, use real pathology patch images via `--image_dir` or `config.image_dir`.

## Compare Saved Reports

After generating multiple reports for the same task, you can create a compact Markdown comparison without rerunning inference:

```bash
pathvlm-litebench compare-reports \
  --task zero-shot \
  --report_dirs outputs/zero_shot_clip outputs/zero_shot_plip \
  --run_names clip plip \
  --output outputs/zero_shot_comparison.md
```

The same command supports `--task retrieval` and `--task prompt-sensitivity`. The comparison reads saved CSV/JSON artifacts only and writes a local Markdown file. Do not commit generated comparisons if they contain dataset-specific paths or metrics.

## Zero-Shot Prompt Grids

To run a small zero-shot prompt grid from JSON config:

```bash
pathvlm-litebench validate-config \
  configs/zero_shot_prompt_grid_mhist_sample.json
```

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json
```

Preview the expanded model/prompt runs without loading models:

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json \
  --dry-run
```

Each run writes a normal zero-shot report directory under the configured `output_root`. When `save_comparison` is enabled, the command also writes a zero-shot comparison Markdown file from saved artifacts. Generated reports stay under `outputs/` and are ignored by Git.

For config fields, output layout, and interpretation guidance, see [docs/prompt_grid_workflow.md](docs/prompt_grid_workflow.md).

## Example Commands

Patch-text retrieval with custom prompts:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto --prompts "a red image" "a blue image" "a black image" --top_k 2
```

Patch-text retrieval on your own patch image folder:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto --image_dir path/to/your/patches --prompts "tumor region" "normal tissue" --top_k 5
```

Manifest-based patch-text retrieval with Recall@K:

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

`--label_prompts` maps each text prompt to a manifest label. If labels are available, the demo prints text-to-image Recall@K for lightweight patch-level retrieval benchmarking.

When `--label_prompts` is used with a labeled manifest, retrieval results in the terminal and HTML report also include:
- the retrieved patch label
- the target label for each text prompt
- whether each retrieved patch is a positive match
- Recall@K metrics in the terminal

To save structured retrieval outputs:

```bash
python examples/01_patch_text_retrieval_demo.py \
  --manifest dataset/MHIST/manifest.csv \
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
  --save_html_report \
  --save_report \
  --report_dir outputs/retrieval_demo
```

This generates:

- `retrieval_results.csv`
- `retrieval_metrics.json`
- optionally `retrieval_report.html`

`retrieval_results.csv` stores prompt-level top-k retrieval results. `retrieval_metrics.json` stores Recall@K metrics and experiment metadata when labels are available. Outputs are ignored by Git.

To generate a compact Markdown summary from a saved retrieval report:

```bash
pathvlm-litebench summarize-report \
  --task retrieval \
  --report_dir outputs/retrieval_demo
```

This writes `experiment_summary.md` in the report directory without rerunning model inference.

Save top-k visualization grids:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto --save_visualization
```

Generate an HTML retrieval report:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto --save_html_report
```

Use image embedding cache:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto --use_cache
```

Combine cache, visualization, and HTML report:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --device auto --use_cache --save_visualization --save_html_report --top_k 3
```

Zero-shot classification with custom pathology-style class prompts:

```bash
python examples/02_zero_shot_classification_demo.py --model clip --device auto --class_names tumor normal necrosis --class_prompts "a histopathology image of tumor tissue" "a histopathology image of normal tissue" "a histopathology image showing necrosis" --top_k 2
```

Zero-shot classification with built-in class prompt generation:

```bash
python examples/02_zero_shot_classification_demo.py --model clip --device auto --class_names tumor normal necrosis --top_k 2
```

Prompt sensitivity with a different top-k value:

```bash
python examples/03_prompt_sensitivity_demo.py --model clip --device auto --top_k 2
```

Prompt sensitivity with pathology prompt templates:

```bash
python examples/03_prompt_sensitivity_demo.py --model clip --device auto --use_pathology_prompts --concepts tumor normal necrosis --top_k 2
```

## Example Output

The retrieval demo prints top-k results for each text prompt:

```text
========== Retrieval Results ==========

Prompt: a red image
  Top 1: index=0, score=0.2841, path=examples/demo_patches/patch_red.png
  Top 2: index=3, score=0.2317, path=examples/demo_patches/patch_green.png
```

The zero-shot demo prints predicted labels:

```text
========== Zero-Shot Classification Results ==========

Image: examples/demo_patches/patch_red.png
Predicted: red (confidence=0.3012)
Top predictions:
  Top 1: class=red, probability=0.3012, logit=0.2841
```

The prompt sensitivity demo prints concept-level stability metrics:

```text
========== Prompt Sensitivity Results ==========

Concept: red_like
Number of prompts: 3
Mean top-k overlap: 0.6667
Mean similarity std: 0.0312
```

Exact scores may vary depending on model version and runtime environment.

## Outputs

Generated outputs are saved under `outputs/` by default and are ignored by Git.

Typical outputs include:

```text
outputs/
|-- cache/
|   |-- image_embeddings.pt
|   `-- image_paths.json
`-- retrieval_demo/
    |-- retrieval_report.html
    `-- topk_prompt_*.png
```

The automatically generated demo images are saved under:

```text
examples/demo_patches/
```

This folder is also ignored by Git.

## Low-Compute Design

This project is designed for laptop-friendly experimentation.

Current design choices:

- Use frozen CLIP-style models instead of training large models
- Start from patch-level images instead of full whole-slide image processing
- Avoid heavy dependencies such as OpenSlide and TIAToolbox in the early stage
- Cache image embeddings to reduce repeated computation
- Keep demos simple, reproducible, and easy to inspect
- Provide terminal outputs, image grids, and HTML reports
- Run smoke tests on CPU-only environments
- Accelerate model inference with CUDA on consumer-grade laptop GPUs when available
- Use default `--device auto` mode to choose CUDA when available and otherwise fall back to CPU

## Current Limitations

- CLIP remains the default baseline model. PLIP is available as a pathology-specific CLIP-compatible model key.
- CONCH requires gated Hugging Face access and the optional `mahmoodlab/CONCH` package.
- The built-in demo images are not pathology images.
- Whole-slide image processing is not supported in the current version.
- No large-scale benchmark dataset is included.
- Prompt sensitivity analysis currently focuses on retrieval stability rather than clinical validity.

Future milestones may add more packaged patch-level benchmark examples, but the current release remains patch-level and local-first.

## Roadmap

- [x] Build basic project structure
- [x] Implement a CLIP model wrapper
- [x] Support patch image loading
- [x] Implement image-text retrieval
- [x] Add minimal command-line retrieval demo
- [x] Add top-k image visualization
- [x] Add embedding caching
- [x] Add HTML retrieval report
- [x] Add zero-shot classification utility
- [x] Add zero-shot classification demo
- [x] Add prompt sensitivity analysis utility
- [x] Add prompt sensitivity demo
- [x] Add lightweight model registry
- [x] Add manifest conversion and balanced sampling utilities
- [x] Add retrieval metrics such as Recall@K
- [x] Add classification metrics beyond accuracy
- [x] Add config-driven retrieval, zero-shot, and prompt sensitivity demos
- [x] Document MHIST CLIP baseline observation
- [x] Add pathology-specific PLIP wrapper
- [x] Document preliminary CLIP vs PLIP MHIST observation
- [x] Add zero-shot, retrieval, and prompt sensitivity experiment summary Markdown utilities
- [x] Add multi-run comparison summaries for saved report artifacts
- [x] Add v0.4.0 CONCH feasibility-first milestone plan
- [x] Add CONCH local feasibility check protocol
- [x] Implement optional CONCH wrapper
- [x] Add zero-shot prompt-grid runner
- [x] Document public patch benchmark workflow
- [x] Add benchmark config validation CLI
- [x] Add optional patch-coordinate text-guided heatmap demo

Related docs:

- Current release: [v0.6.0 release notes](docs/release_notes_v0.6.0.md), [v0.6.0 checklist](docs/release_checklist_v0.6.0.md)
- Post-release audit: [v0.6.0 audit](docs/v0.6.0_post_release_audit.md), [v0.5.0 audit](docs/v0.5.0_post_release_audit.md)
- Next milestone: [v0.7.0 plan](docs/v0.7.0_plan.md), [v0.7.0 release notes](docs/release_notes_v0.7.0.md), [v0.7.0 checklist](docs/release_checklist_v0.7.0.md), [patch coordinate heatmap workflow](docs/patch_coordinate_heatmap_workflow.md)
- v0.6.0 milestone: [v0.6.0 plan](docs/v0.6.0_plan.md), [v0.6.0 MHIST reproducibility audit](docs/v0.6.0_mhist_reproducibility_audit.md), [v0.6.0 MHIST prompt-grid audit](docs/v0.6.0_mhist_prompt_grid_audit.md)
- Previous release: [v0.5.0 release notes](docs/release_notes_v0.5.0.md), [v0.5.0 checklist](docs/release_checklist_v0.5.0.md)
- Earlier releases: [v0.4.1 release notes](docs/release_notes_v0.4.1.md), [v0.4.1 checklist](docs/release_checklist_v0.4.1.md), [v0.4.0 release notes](docs/release_notes_v0.4.0.md), [v0.4.0 checklist](docs/release_checklist_v0.4.0.md)
- CONCH milestone: [v0.4.0 plan](docs/v0.4.0_plan.md), [CONCH feasibility checks](docs/conch_feasibility_check.md)
- Earlier milestones: [v0.3.1 notes](docs/release_notes_v0.3.1.md), [v0.3.0 notes](docs/release_notes_v0.3.0.md), [v0.2.0 notes](docs/release_notes_v0.2.0.md)
- PLIP and MHIST references: [PLIP feasibility checks](docs/plip_feasibility_check.md), [CLIP vs PLIP protocol](docs/clip_vs_plip_mhist_protocol.md), [CLIP vs PLIP observations](docs/clip_vs_plip_mhist_observation.md)

## Research Positioning

PathVLM-LiteBench is intended as a lightweight research engineering project for computational pathology and medical vision-language model evaluation.

The project does not aim to reproduce large-scale foundation model pretraining. Instead, it focuses on reproducible low-compute evaluation workflows that can help students and small research groups analyze pathology vision-language models.

For a more detailed discussion of the research motivation and scope, see [docs/project_positioning.md](docs/project_positioning.md).

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development principles, testing instructions, and contribution guidelines.

## License

This project is released for academic and research purposes.
