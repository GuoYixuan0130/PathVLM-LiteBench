# PathVLM-LiteBench

PathVLM-LiteBench is a lightweight toolkit for benchmarking and visualizing vision-language models in computational pathology under limited computing resources.

This project focuses on patch-level pathology image-text retrieval, zero-shot classification, prompt sensitivity analysis, and visualization reports using frozen pathology vision-language models.

## Motivation

Recent computational pathology foundation models and vision-language models have shown strong potential in histopathology image understanding. However, reproducing large-scale pretraining is computationally expensive and often infeasible for students or small research groups.

PathVLM-LiteBench aims to provide a low-compute and reproducible toolkit for evaluating and comparing pathology vision-language models using precomputed or frozen embeddings.

## Planned Features

- Patch-level image-text retrieval
- Zero-shot pathology patch classification
- Prompt sensitivity analysis
- Embedding caching
- HTML visualization report
- Optional WSI-level text-guided heatmap demo

## Current Stage

The current version is in early development. The first milestone is to build a minimal patch-level image-text retrieval demo using frozen CLIP/PLIP-style models.

## Repository Structure

```text
PathVLM-LiteBench/
├── pathvlm_litebench/
│   ├── models/
│   ├── retrieval/
│   ├── evaluation/
│   └── visualization/
├── examples/
├── README.md
├── requirements.txt
└── .gitignore