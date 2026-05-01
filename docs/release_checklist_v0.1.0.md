# v0.1.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a stable v0.1.0 research engineering release.

The goal of v0.1.0 is not to support every pathology VLM or whole-slide workflow. The goal is to provide a reproducible, laptop-friendly CLIP baseline pipeline for patch-level computational pathology vision-language evaluation.

## Release Positioning

v0.1.0 positioning includes:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level evaluation workflow
- a frozen vision-language model inference toolkit
- a CLIP baseline implementation with extension points for pathology-specific VLMs
- a research and educational tool

v0.1.0 positioning excludes:

- a clinical diagnostic system
- a CPU-only project
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- an implementation of PLIP or CONCH

## Core Feature Checklist

- [x] CLIP wrapper for frozen image and text embedding
- [x] Model registry with implemented CLIP and placeholder PLIP / CONCH keys
- [x] Explicit `--device` support: `auto`, `cpu`, and `cuda`
- [x] Patch image loading from folders
- [x] Patch image loading from manifest paths
- [x] Standard manifest loader
- [x] Manifest conversion utility
- [x] MHIST manifest conversion preset
- [x] Balanced manifest sampling utility
- [x] Image embedding cache
- [x] Patch-text retrieval
- [x] Manifest-based retrieval evaluation with Recall@K
- [x] Label-aware retrieval reports
- [x] Retrieval CSV / JSON report saving
- [x] Zero-shot classification
- [x] Manifest-based zero-shot evaluation
- [x] Classification metrics beyond accuracy
- [x] Zero-shot predictions and metrics report saving
- [x] Zero-shot error analysis report
- [x] Prediction distribution and collapse warning
- [x] Prompt sensitivity analysis
- [x] Prompt sensitivity CSV / JSON report saving
- [x] Top-k visualization
- [x] HTML retrieval report
- [x] HTML report asset copying to avoid broken image paths
- [x] Prompt template library
- [x] Benchmark config utilities
- [x] Config-driven retrieval demo
- [x] Config-driven zero-shot demo
- [x] Config-driven prompt sensitivity demo
- [x] CLI entry point

## Documentation Checklist

- [x] README explains project motivation and scope
- [x] README states CPU-compatible plus laptop-GPU accelerated positioning
- [x] README lists model registry status
- [x] README documents `--device`
- [x] README links to real-data preparation docs
- [x] README links to small dataset quickstart
- [x] README links to MHIST baseline observation
- [x] `docs/data_preparation.md` uses `dataset/` for local data examples
- [x] `docs/real_patch_workflow.md` documents patch-level workflows
- [x] `docs/small_dataset_quickstart.md` gives a short MHIST-style path
- [x] `docs/mhist_baseline_observation.md` records the CLIP baseline failure mode
- [x] `docs/project_positioning.md` explains the research engineering narrative
- [x] Documentation warns that demo images are smoke tests, not pathology images
- [x] Documentation warns that real pathology data should not be committed
- [x] Documentation avoids claiming clinical use

## Testing and CI Checklist

- [x] Unit tests run without downloading models
- [x] CLI smoke tests cover `version`, `models`, and `demos`
- [x] Manifest conversion and sampling are tested
- [x] Manifest loader helpers are tested
- [x] Retrieval metrics are tested
- [x] Classification metrics are tested
- [x] Zero-shot report saving is tested
- [x] Prompt sensitivity config merging is tested
- [x] HTML retrieval report asset handling is tested
- [x] GitHub Actions CI runs lightweight tests
- [x] CLI import path avoids loading model dependencies for lightweight commands

Recommended local verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
```

Optional CUDA smoke check on a laptop GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## Data and Artifact Hygiene

Before release, verify that these are not committed:

- `dataset/`
- `outputs/`
- `examples/demo_patches/`
- model weights
- embedding cache files
- real pathology images
- generated reports

Expected `.gitignore` coverage:

```text
/dataset/
/outputs/
/examples/demo_patches/
*.pt
*.pth
*.ckpt
*.npy
*.npz
```

Do not add a root-level `data/` ignore rule because `pathvlm_litebench/data/` is a code module.

## Example v0.1.0 Smoke Workflow

1. Run CLI smoke commands.
2. Run unit tests.
3. Convert MHIST annotations into a standard manifest.
4. Sample a balanced test manifest.
5. Run retrieval with Recall@K.
6. Run zero-shot classification with report saving.
7. Run prompt sensitivity with report saving.
8. Confirm generated outputs are under `outputs/`.

## Known Limitations for v0.1.0

- CLIP is the only implemented model wrapper.
- PLIP and CONCH are registered placeholders only.
- The toolkit starts from patch-level images, not raw WSI files.
- Prompt sensitivity measures retrieval stability, not clinical validity.
- General CLIP may perform poorly on fine-grained pathology tasks.
- No datasets, real pathology images, model weights, or generated reports are included.

## Tag Criteria

Tag v0.1.0 when:

- [ ] `python -m pytest tests` passes locally
- [ ] CI passes on GitHub
- [ ] README and docs render correctly on GitHub
- [ ] `.gitignore` excludes local data and generated outputs
- [ ] The latest commit is pushed to `main`
- [ ] Optional MHIST sampled workflow has been tested locally
- [ ] Release notes mention that PLIP / CONCH are not implemented yet

## Release Notes

See `docs/release_notes_v0.1.0.md` for the release notes.
