# v0.1.0 Release Notes

## Summary

PathVLM-LiteBench v0.1.0 provides a low-compute, patch-level computational pathology vision-language evaluation toolkit centered on a frozen CLIP baseline.

The release focuses on reproducible laptop-friendly workflows rather than large-scale model training. It supports patch-text retrieval, zero-shot classification, prompt sensitivity analysis, manifest conversion and sampling, embedding caching, CSV / JSON reports, HTML retrieval reports, config-driven demos, tests, and CI.

This release is intended for research and educational use only.

## What Is Included

Core model workflow:

- CLIP wrapper for frozen image and text embedding
- model registry with `clip`, `clip-vit-base-patch32`, `plip`, and `conch` keys
- implemented CLIP baseline through `openai/clip-vit-base-patch32`
- explicit device selection with `--device auto`, `--device cpu`, and `--device cuda`
- CUDA acceleration for model inference when available

Data workflow:

- patch image loading from folders
- patch image loading from manifest paths
- standard manifest loader
- MHIST manifest conversion preset
- generic manifest conversion utility
- balanced manifest sampling utility for low-compute experiments

Evaluation workflow:

- image-text retrieval
- manifest-based retrieval evaluation with Recall@K
- zero-shot patch classification
- classification metrics including accuracy, balanced accuracy, macro precision / recall / F1, per-class metrics, and confusion matrix
- zero-shot error analysis with `errors.csv`
- true and predicted label distribution reporting
- prediction-collapse warning for strongly skewed predictions
- prompt sensitivity analysis with CSV / JSON reports

Reporting and reproducibility:

- embedding cache
- top-k visualization grids
- HTML retrieval reports
- copied HTML report assets to avoid broken image paths
- retrieval CSV / JSON reports
- zero-shot CSV / JSON reports
- prompt sensitivity CSV / JSON reports
- benchmark config utilities
- config-driven retrieval, zero-shot, and prompt sensitivity demos
- lightweight CLI commands
- pytest test suite
- GitHub Actions CI

## Example Workflows

Small dataset workflow:

1. Convert dataset annotations into a standard manifest.
2. Sample a balanced subset for laptop-friendly testing.
3. Run retrieval with Recall@K.
4. Run zero-shot classification with metrics and error reports.
5. Run prompt sensitivity analysis.
6. Inspect generated outputs under `outputs/`.

For a concrete MHIST-style walkthrough, see:

- `docs/small_dataset_quickstart.md`
- `docs/data_preparation.md`
- `docs/real_patch_workflow.md`

## MHIST Baseline Observation

A local zero-shot run on MHIST with the default CLIP baseline showed strong class bias toward `HP` and very low `SSA` recall.

This is documented as a baseline observation, not a clinical claim. It motivates the toolkit's emphasis on balanced accuracy, macro-F1, per-class recall, confusion matrices, saved predictions, and error analysis.

See:

- `docs/mhist_baseline_observation.md`

## What Is Intentionally Not Included

v0.1.0 does not include:

- PLIP implementation
- CONCH implementation
- full WSI high-throughput processing
- large-scale VLM pretraining
- clinical diagnosis or clinical decision support
- bundled public datasets
- real pathology images
- model weights
- generated reports

PLIP and CONCH are registered as future extension points only. Passing `--model plip` or `--model conch` should raise a clear `NotImplementedError` in this version.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
```

The lightweight CLI commands do not download models by default.

Optional local CUDA check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## Suggested GitHub Release Text

PathVLM-LiteBench v0.1.0 is the first stable low-compute release of the project. It provides a patch-level computational pathology vision-language evaluation toolkit centered on a frozen CLIP baseline, with CPU-compatible smoke tests and CUDA acceleration when available.

Highlights:

- CLIP baseline wrapper and model registry
- patch-text retrieval with Recall@K
- zero-shot classification with balanced accuracy, macro-F1, per-class metrics, confusion matrix, predictions, and error reports
- prompt sensitivity analysis
- manifest conversion and balanced sampling utilities, including an MHIST preset
- embedding cache, top-k visualization, HTML reports, and CSV / JSON report outputs
- config-driven demo workflows
- lightweight CLI commands, tests, and GitHub Actions CI

This release is intended for research and educational use only. It is not a clinical diagnostic tool. PLIP and CONCH are registered as future extension points but are not implemented in v0.1.0.
