# PathVLM-LiteBench

PathVLM-LiteBench is a lightweight toolkit for benchmarking and visualizing vision-language models in computational pathology under limited computing resources.

This project focuses on patch-level pathology image-text retrieval, zero-shot classification, prompt sensitivity analysis, and visualization reports using frozen pathology vision-language models.

## Motivation

Recent computational pathology foundation models and vision-language models have shown strong potential in histopathology image understanding. However, reproducing large-scale pretraining is computationally expensive and often infeasible for students or small research groups.

PathVLM-LiteBench aims to provide a low-compute and reproducible toolkit for evaluating and comparing pathology vision-language models using precomputed or frozen embeddings.

Instead of training large models from scratch, this project focuses on:

- Frozen vision-language model inference
- Patch-level image-text retrieval
- Lightweight evaluation utilities
- Reproducible demos
- Laptop-friendly experimentation

## Planned Features

- Patch-level image-text retrieval
- Zero-shot pathology patch classification
- Prompt sensitivity analysis
- Embedding caching
- HTML visualization report
- Optional WSI-level text-guided heatmap demo

## Current Stage

The current version is in early development. The first milestone is to build a minimal patch-level image-text retrieval demo using frozen CLIP-style models.

Currently supported:

- Loading patch images from a folder
- Encoding images with CLIP
- Encoding text prompts with CLIP
- Computing image-text similarity
- Retrieving top-k images for each prompt
- Running a minimal command-line demo

## Repository Structure

```text
PathVLM-LiteBench/
├── pathvlm_litebench/
│   ├── data/
│   │   ├── __init__.py
│   │   └── patch_loader.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── clip_wrapper.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── image_text_search.py
│   ├── evaluation/
│   └── visualization/
├── examples/
│   ├── 01_quick_start.ipynb
│   └── 01_patch_text_retrieval_demo.py
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

## Quick Start

Run the minimal patch-text retrieval demo:

```bash
python examples/01_patch_text_retrieval_demo.py
```

If no image folder is provided, the script automatically creates a small demo folder with simple RGB images. These demo images are not pathology images. They are only used to verify that the pipeline works end-to-end.

You can also provide custom text prompts:

```bash
python examples/01_patch_text_retrieval_demo.py --prompts "a red image" "a blue image" "a black image" --top_k 2
```

To run the demo on your own patch image folder:

```bash
python examples/01_patch_text_retrieval_demo.py --image_dir path/to/your/patches --prompts "tumor region" "normal tissue" --top_k 5
```

## Command-Line Demo Usage

```bash
python examples/01_patch_text_retrieval_demo.py --help
```

Key arguments:

- `--image_dir`: folder containing patch images
- `--prompts`: one or multiple text prompts
- `--top_k`: number of top matched images per prompt
- `--model_name`: Hugging Face CLIP-style model name

## Example Output Format

The demo prints top-k retrieval results for each text prompt:

```text
========== Retrieval Results ==========

Prompt: a red image
  Top 1: index=0, score=0.2841, path=examples/demo_patches/patch_red.png
  Top 2: index=3, score=0.2317, path=examples/demo_patches/patch_green.png

Prompt: a blue image
  Top 1: index=1, score=0.3015, path=examples/demo_patches/patch_blue.png
  Top 2: index=4, score=0.1882, path=examples/demo_patches/patch_black.png
```

The exact scores may vary depending on the model version and runtime environment.

## Low-Compute Design Philosophy

This project is designed for laptop-friendly experimentation.

Current design choices:

- Use frozen CLIP-style models instead of training large models
- Start from patch-level images instead of full WSI processing
- Avoid heavy dependencies such as OpenSlide and TIAToolbox in the early stage
- Keep the first demo simple and reproducible
- Support future embedding caching to reduce repeated computation

## Current Limitations

- The current demo uses CLIP rather than a pathology-specific VLM.
- The automatically generated demo images are not pathology images.
- WSI-level processing is not supported in the current version.
- No zero-shot classification benchmark is included yet.
- No prompt sensitivity analysis is included yet.
- No HTML visualization report is included yet.

These features are planned for later milestones.

## Roadmap

- [x] Build basic project structure
- [x] Implement a simple CLIP model wrapper
- [x] Support patch image loading
- [x] Implement image-text retrieval
- [x] Add minimal command-line retrieval demo
- [ ] Add top-k image visualization
- [ ] Add zero-shot classification utility
- [ ] Add prompt sensitivity analysis
- [ ] Add embedding caching
- [ ] Add HTML report generation
- [ ] Add optional WSI-level text-guided heatmap demo

## Academic Positioning

PathVLM-LiteBench is intended as a lightweight research engineering project for computational pathology and medical vision-language model evaluation.

The project does not aim to reproduce large-scale foundation model pretraining. Instead, it focuses on reproducible low-compute evaluation workflows that can help students and small research groups analyze pathology vision-language models.

## License

This project is released for academic and research purposes.
