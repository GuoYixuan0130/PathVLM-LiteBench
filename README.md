# PathVLM-LiteBench

[![CI](https://github.com/GuoYixuan0130/PathVLM-LiteBench/actions/workflows/ci.yml/badge.svg)](https://github.com/GuoYixuan0130/PathVLM-LiteBench/actions/workflows/ci.yml)

PathVLM-LiteBench is a lightweight toolkit for benchmarking and visualizing vision-language models in computational pathology under limited computing resources.

This project focuses on patch-level pathology image-text retrieval, zero-shot classification, prompt sensitivity analysis, embedding caching, and visualization reports using frozen CLIP/PLIP-style vision-language models.

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
- Laptop-friendly experimentation

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

## Model Registry

PathVLM-LiteBench uses a lightweight model registry so that demos can accept either a short model key or a Hugging Face model name.

Currently supported model keys:

| Model key | Resolved model name | Status | Notes |
|---|---|---|---|
| `clip` | `openai/clip-vit-base-patch32` | Implemented | Default CLIP baseline |
| `clip-vit-base-patch32` | `openai/clip-vit-base-patch32` | Implemented | Alias for CLIP ViT-B/32 |
| `plip` | `vinid/plip` | Placeholder | Registered for future pathology-specific VLM support |
| `conch` | `MahmoodLab/CONCH` | Placeholder | Registered for future pathology-specific VLM support |

You can run demos with a registered model key:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip
```

or with a Hugging Face model name:

```bash
python examples/01_patch_text_retrieval_demo.py --model openai/clip-vit-base-patch32
```

Pathology-specific wrappers such as PLIP and CONCH are planned but not implemented yet. Passing `--model plip` or `--model conch` will raise a clear `NotImplementedError` in the current version.

## Repository Structure

```text
PathVLM-LiteBench/
├── pathvlm_litebench/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── patch_loader.py
│   │   └── embedding_cache.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── clip_wrapper.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── image_text_search.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── zero_shot.py
│   │   └── prompt_sensitivity.py
│   └── visualization/
│       ├── __init__.py
│       ├── topk_viewer.py
│       └── html_report.py
├── examples/
│   ├── 01_quick_start.ipynb
│   ├── 01_patch_text_retrieval_demo.py
│   ├── 02_zero_shot_classification_demo.py
│   └── 03_prompt_sensitivity_demo.py
├── README.md
├── requirements.txt
└── .gitignore
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

## Data Preparation

To run PathVLM-LiteBench on real pathology patches, prepare a folder of patch-level images and pass it with `--image_dir`.

See [docs/data_preparation.md](docs/data_preparation.md) for details.

## Quick Start

### Demo 1: Patch-Text Retrieval

Run the minimal patch-text retrieval demo:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip
```

If no image folder is provided, the script automatically creates a small demo folder with simple RGB images. These demo images are only used to verify that the pipeline works end-to-end.

### Demo 2: Zero-Shot Classification

Run the zero-shot classification demo:

```bash
python examples/02_zero_shot_classification_demo.py --model clip
```

This demo classifies each patch by comparing its image embedding with class text prompt embeddings.

### Demo 3: Prompt Sensitivity Analysis

Run the prompt sensitivity demo:

```bash
python examples/03_prompt_sensitivity_demo.py --model clip
```

This demo evaluates whether different prompt variants for the same concept retrieve similar top-k image results.

## Example Commands

Patch-text retrieval with custom prompts:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --prompts "a red image" "a blue image" "a black image" --top_k 2
```

Patch-text retrieval on your own patch image folder:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --image_dir path/to/your/patches --prompts "tumor region" "normal tissue" --top_k 5
```

Save top-k visualization grids:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --save_visualization
```

Generate an HTML retrieval report:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --save_html_report
```

Use image embedding cache:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --use_cache
```

Combine cache, visualization, and HTML report:

```bash
python examples/01_patch_text_retrieval_demo.py --model clip --use_cache --save_visualization --save_html_report --top_k 3
```

Zero-shot classification with custom pathology-style class prompts:

```bash
python examples/02_zero_shot_classification_demo.py --model clip --class_names tumor normal necrosis --class_prompts "a histopathology image of tumor tissue" "a histopathology image of normal tissue" "a histopathology image showing necrosis" --top_k 2
```

Prompt sensitivity with a different top-k value:

```bash
python examples/03_prompt_sensitivity_demo.py --model clip --top_k 2
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
├── cache/
│   ├── image_embeddings.pt
│   └── image_paths.json
└── retrieval_demo/
    ├── retrieval_report.html
    └── topk_prompt_*.png
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
- Start from patch-level images instead of full WSI processing
- Avoid heavy dependencies such as OpenSlide and TIAToolbox in the early stage
- Cache image embeddings to reduce repeated computation
- Keep demos simple, reproducible, and easy to inspect
- Provide terminal outputs, image grids, and HTML reports

## Current Limitations

- The current implementation uses CLIP by default rather than a pathology-specific VLM.
- The current implemented model wrapper uses CLIP-style Hugging Face models.
- PLIP and CONCH are registered as placeholders but are not implemented in the current version.
- The built-in demo images are not pathology images.
- WSI-level processing is not supported in the current version.
- No large-scale benchmark dataset is included.
- No PLIP/CONCH-specific wrapper is included yet.
- Prompt sensitivity analysis currently focuses on retrieval stability rather than clinical validity.

These features may be added in later milestones.

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
- [ ] Add pathology-specific PLIP wrapper
- [ ] Implement PLIP wrapper
- [ ] Implement CONCH wrapper
- [ ] Add retrieval metrics such as Recall@K
- [ ] Add classification metrics beyond accuracy
- [ ] Add example with real public pathology patch data
- [ ] Add optional WSI-level text-guided heatmap demo

## Academic Positioning

PathVLM-LiteBench is intended as a lightweight research engineering project for computational pathology and medical vision-language model evaluation.

The project does not aim to reproduce large-scale foundation model pretraining. Instead, it focuses on reproducible low-compute evaluation workflows that can help students and small research groups analyze pathology vision-language models.

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development principles, testing instructions, and contribution guidelines.

## License

This project is released for academic and research purposes.
