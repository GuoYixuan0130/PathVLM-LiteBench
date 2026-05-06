# v0.4.0 Release Notes

## Summary

PathVLM-LiteBench v0.4.0 adds optional CONCH model support.

This release extends the model registry beyond CLIP and PLIP by adding a `CONCHWrapper` for the gated `MahmoodLab/CONCH` pathology vision-language model. CONCH support is optional: it requires the official CONCH package, approved Hugging Face access, and local Hugging Face authentication. The core package dependencies and CI tests remain lightweight and offline.

This release is intended for research and educational use only.

## What Is Included

CONCH workflow:

- added `pathvlm_litebench.models.CONCHWrapper`
- registered `conch` as an implemented optional model key
- routed both `--model conch` and `--model MahmoodLab/CONCH` through `CONCHWrapper`
- added clear setup errors for missing optional CONCH dependencies or gated-model access
- added local feasibility documentation for CONCH access, install, CPU, and CUDA checks
- added offline unit tests using fake CONCH objects so CI does not download model weights

Local feasibility checks passed on the development laptop:

- Hugging Face gated access and token authentication worked
- CONCH package import worked
- CPU model loading worked
- CPU text and image embeddings were `torch.Size([*, 512])`
- CUDA image smoke test worked on `cuda:0`
- batch size 1 CUDA image smoke peak memory was about `1549 MB`

## Optional CONCH Setup

CONCH is not a core dependency. To use it locally:

```bash
pip install git+https://github.com/Mahmoodlab/CONCH.git
hf auth login
```

The Hugging Face account used for `hf auth login` must have access to the gated `MahmoodLab/CONCH` model.

Then run demos with:

```bash
python examples/01_patch_text_retrieval_demo.py --model conch --device cpu
python examples/02_zero_shot_classification_demo.py --model conch --device cpu
python examples/03_prompt_sensitivity_demo.py --model conch --device cpu
```

Use `--device cuda` only when local GPU memory is sufficient.

## What Remains Unchanged

The toolkit remains focused on:

- frozen model inference
- patch-level images
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- lightweight tests and CI
- local CSV/JSON/Markdown reporting

Existing workflows remain available:

- CLIP and PLIP model wrappers
- retrieval, zero-shot classification, and prompt sensitivity demos
- manifest conversion and sampling
- summary and comparison Markdown reports
- embedding cache and visualization utilities

## What Is Intentionally Not Included

v0.4.0 does not include:

- CONCH as a required dependency
- CONCH model downloads in CI
- bundled CONCH weights
- automatic Hugging Face token management
- clinical diagnosis or clinical decision support
- full WSI high-throughput processing
- model training or fine-tuning
- dashboards or hosted tracking integrations
- generated output reports

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, or generated reports.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
```

Optional local CONCH smoke check:

```bash
python -c "from PIL import Image; from pathvlm_litebench.models import create_model; model = create_model('conch', device='cpu'); text = model.encode_text(['a histopathology image']); image = model.encode_images([Image.new('RGB', (224, 224), 'white')]); print(text.shape, image.shape)"
```

This optional check may download CONCH weights and requires gated Hugging Face access.

## Release Highlights

PathVLM-LiteBench v0.4.0 improves pathology VLM coverage:

- CLIP, PLIP, and optional CONCH are available through one model registry
- CONCH can be used in existing retrieval, zero-shot, and prompt sensitivity workflows
- CI stays offline and lightweight
- setup constraints are documented rather than hidden
