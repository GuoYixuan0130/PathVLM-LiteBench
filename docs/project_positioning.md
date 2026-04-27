# Project Positioning

PathVLM-LiteBench is a lightweight, CPU-compatible and CUDA-accelerated toolkit for evaluating vision-language models in computational pathology under limited computing resources.

The project is designed around a practical constraint: large-scale pathology vision-language model pretraining is often infeasible for students, small research groups, and laptop-based development environments. Instead of reproducing expensive foundation model training, PathVLM-LiteBench focuses on reproducible low-compute evaluation workflows.

## Core Positioning

PathVLM-LiteBench is not intended to be a large-scale foundation model training framework.

Instead, it focuses on:

- frozen vision-language model inference
- patch-level pathology image-text retrieval
- zero-shot classification
- prompt sensitivity analysis
- retrieval metrics
- embedding caching
- lightweight visualization
- HTML-based reporting

This makes it suitable for research prototyping, model comparison, and educational use in computational pathology.

## Compute Philosophy

The toolkit follows a low-compute but not CPU-only design.

Minimum mode:

- CPU-compatible smoke tests
- toy embedding metric demos
- small patch-folder retrieval examples
- unit tests that do not download models

Recommended mode:

- consumer-grade laptop GPU
- CUDA acceleration for image embedding extraction
- cached image embeddings for repeated experiments
- patch-level rather than full WSI-level workflows

Not required:

- A100 GPUs
- multi-GPU training
- large-scale VLM pretraining
- full-resolution WSI inference
- high-throughput deployment infrastructure

This design makes the project accessible while still allowing practical acceleration on devices such as RTX laptop GPUs.

## Why Patch-Level First?

Whole-slide images are extremely large and often require specialized storage, preprocessing, and memory management. Full WSI-level analysis can quickly become computationally expensive.

PathVLM-LiteBench starts from patch-level images because patch-level workflows are:

- easier to reproduce
- easier to debug
- compatible with limited hardware
- suitable for embedding-based retrieval
- useful for studying prompt and model behavior
- extensible to sampled WSI-level analysis later

WSI-level text-guided heatmap generation may be added as an optional future module, but it should not become a mandatory dependency of the core toolkit.

## Why Vision-Language Evaluation?

Computational pathology is increasingly moving toward multimodal foundation models that align histology images with natural language, diagnostic concepts, pathology reports, or medical prompts.

PathVLM-LiteBench focuses on evaluation rather than pretraining because many important research questions can be studied through frozen models:

- Can a text prompt retrieve relevant pathology patches?
- How stable are retrieval results under different prompt formulations?
- Can class-level text prompts support zero-shot patch classification?
- How does model behavior change across prompt styles?
- Can lightweight metrics expose failure cases or prompt sensitivity?

These questions are directly relevant to medical vision-language model reliability and practical deployment.

## Current Technical Scope

The current implementation supports:

- CLIP-style model wrapping
- model registry
- explicit CPU/CUDA device selection
- patch image loading
- image-text retrieval
- zero-shot classification
- prompt sensitivity analysis
- Recall@K retrieval metrics
- embedding caching
- top-k image grid visualization
- HTML retrieval reports
- command-line demos
- lightweight unit tests and CI

Pathology-specific models such as PLIP and CONCH are registered as placeholders for future support but are not fully implemented in the current version.

## Research Value

The project demonstrates several research engineering capabilities:

1. **Problem-aware design**
   It acknowledges the compute barrier in CPath foundation model research and proposes a practical low-compute evaluation workflow.

2. **Modular implementation**
   Data loading, model wrapping, retrieval, evaluation, visualization, and reporting are separated into reusable modules.

3. **Evaluation orientation**
   The toolkit goes beyond running a model by including retrieval metrics, zero-shot classification, and prompt sensitivity analysis.

4. **Reproducibility**
   The project includes tests, CI, documentation, and deterministic toy examples.

5. **Extensibility**
   The model registry and modular structure allow future support for pathology-specific VLMs such as PLIP and CONCH.

## Suggested PhD Application Description

A concise description for a CV or research statement:

> Developed PathVLM-LiteBench, a low-compute and laptop-friendly toolkit for evaluating pathology vision-language models. The toolkit supports CPU-compatible workflows and optional CUDA acceleration on consumer-grade GPUs, enabling patch-level image-text retrieval, zero-shot classification, prompt sensitivity analysis, retrieval metrics, embedding caching, and HTML-based visualization without large-scale model training.

A shorter version:

> Built PathVLM-LiteBench, a lightweight open-source toolkit for low-compute evaluation of computational pathology vision-language models, including patch-level retrieval, zero-shot classification, prompt sensitivity analysis, and visualization.

## Limitations

The current version has several limitations:

- It uses CLIP as the implemented baseline model.
- PLIP and CONCH wrappers are not yet fully implemented.
- Built-in demo images are smoke tests, not pathology data.
- The toolkit does not include full WSI processing.
- Evaluation results depend heavily on prompt design and dataset quality.
- It is not intended for clinical diagnosis.

## Future Directions

Possible future extensions include:

- PLIP wrapper implementation
- CONCH wrapper implementation
- public pathology patch dataset demo
- prompt template library for pathology concepts
- richer classification metrics
- retrieval benchmark configuration files
- optional sampled WSI-level text-guided heatmap
- comparison across different pathology VLMs

## Summary

PathVLM-LiteBench is positioned as a practical research engineering toolkit for low-compute computational pathology vision-language model evaluation.

Its value lies not in training a large foundation model, but in making frozen-model evaluation, prompt analysis, retrieval metrics, and visualization easier to reproduce under realistic hardware constraints.
