# Contributing to PathVLM-LiteBench

Thank you for your interest in contributing to PathVLM-LiteBench.

PathVLM-LiteBench is a lightweight research engineering toolkit for computational pathology vision-language model evaluation. The project is designed to be laptop-friendly, reproducible, and easy to inspect.

## Development Principles

This project follows several design principles:

1. **Low-compute first**
   Avoid large-scale training, heavy GPU requirements, or mandatory WSI-level processing in the core workflow.

2. **Frozen model evaluation**
   The default workflow should use frozen CLIP/PLIP-style models rather than training large vision-language models from scratch.

3. **Patch-level before WSI-level**
   Core modules should work on patch-level images first. WSI-level features should remain optional and lightweight.

4. **Modular design**
   Data loading, model wrappers, retrieval, evaluation, visualization, and demos should remain separated.

5. **Reproducibility**
   Demos and tests should be easy to run on a standard laptop.

6. **No clinical claims**
   This toolkit is intended for research and educational use. It should not make clinical diagnostic claims.

## Local Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/PathVLM-LiteBench.git
cd PathVLM-LiteBench
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For editable development installation, you can also run:

```bash
pip install -e .
```

## Running Tests

Run all unit tests:

```bash
pytest
```

or:

```bash
pytest tests
```

The unit tests should not download CLIP models or require GPU access.

## Running Demos

Patch-text retrieval:

```bash
python examples/01_patch_text_retrieval_demo.py
```

Zero-shot classification:

```bash
python examples/02_zero_shot_classification_demo.py
```

Prompt sensitivity analysis:

```bash
python examples/03_prompt_sensitivity_demo.py
```

Retrieval metrics:

```bash
python examples/04_retrieval_metrics_demo.py
```

## Code Style

Please keep code:

- simple
- readable
- typed where practical
- modular
- friendly to CPU-only environments

Avoid adding heavy dependencies unless they are clearly optional.

## Adding New Model Wrappers

New model wrappers should be placed under:

```text
pathvlm_litebench/models/
```

A wrapper should ideally provide:

- encode_images
- encode_text
- normalized embeddings
- clear device handling
- helpful error messages

If a model requires restricted access or large downloads, document it clearly and keep it optional.

## Adding New Evaluation Metrics

New metrics should be placed under:

```text
pathvlm_litebench/evaluation/
```

Please include:

- clear input shape assumptions
- validation checks
- small toy examples
- unit tests

## Adding New Visualizations

New visualizations should be placed under:

```text
pathvlm_litebench/visualization/
```

Keep visual outputs lightweight and avoid requiring external web servers or large UI frameworks for core functionality.

## Git Workflow

Before committing, run:

```bash
pytest
```

Check changed files:

```bash
git status
```

Then commit:

```bash
git add .
git commit -m "Your concise commit message"
git push
```

## Files That Should Not Be Committed

Do not commit:

- outputs/
- examples/demo_patches/
- large datasets
- model weights
- embedding cache files
- private pathology data
- patient-level information

These should remain ignored by .gitignore.

## Research Use Disclaimer

PathVLM-LiteBench is intended for research, education, and engineering experimentation. It is not a clinical diagnostic tool.
