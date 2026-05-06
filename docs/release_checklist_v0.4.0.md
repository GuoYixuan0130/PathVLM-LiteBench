# v0.4.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.4.0 research engineering release.

The goal of v0.4.0 is to add optional CONCH support while keeping the core toolkit low-compute, patch-level, and CI-friendly.

## Release Positioning

v0.4.0 positioning includes:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level evaluation workflow
- a frozen vision-language model inference toolkit
- CLIP and PLIP evaluation workflows
- optional CONCH evaluation workflows
- structured report saving for retrieval, zero-shot classification, and prompt sensitivity
- Markdown summaries and multi-run comparisons for saved report artifacts
- a research and educational tool

v0.4.0 positioning excludes:

- a clinical diagnostic system
- a CPU-only project
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- dashboard or experiment-tracking infrastructure
- automatic dataset or output upload
- required CONCH downloads or bundled CONCH weights

## Release Scope

- [x] Add `CONCHWrapper`
- [x] Register `conch` as an implemented optional model key
- [x] Route `MahmoodLab/CONCH` through the CONCH wrapper
- [x] Keep CONCH dependencies optional
- [x] Add clear setup errors for missing package or gated-model access
- [x] Add offline tests with fake CONCH objects
- [x] Avoid CONCH downloads in CI
- [x] Document local CONCH feasibility checks
- [x] Update README and project positioning documentation

## Current Release Blockers

- [x] Bump package version from `0.3.1` to `0.4.0` in `pyproject.toml`
- [x] Bump runtime version from `0.3.1` to `0.4.0` in `pathvlm_litebench/__init__.py`
- [x] Add `docs/release_notes_v0.4.0.md`
- [x] Add this release checklist
- [x] Verify GitHub Actions CI passes on the release commit
- [ ] Decide whether to tag `v0.4.0` after the release notes are reviewed

## Validation Checklist

Run locally before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
```

Optional local CONCH checks:

```bash
python -c "from PIL import Image; from pathvlm_litebench.models import create_model; model = create_model('conch', device='cpu'); text = model.encode_text(['a histopathology image']); image = model.encode_images([Image.new('RGB', (224, 224), 'white')]); print(text.shape, image.shape)"
```

These optional checks depend on local Hugging Face access, local authentication, and cached or downloadable model weights. They should not be added to CI.

## Data and Artifact Hygiene

Before release, verify that these are not committed:

- `dataset/`
- `outputs/`
- `examples/demo_patches/`
- Hugging Face model caches
- CONCH model weights
- embedding cache files
- real pathology images
- generated reports
- generated comparison Markdown files

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

## Documentation Checklist

- [x] README documents CONCH model status
- [x] README documents optional CONCH install and HF auth
- [x] `docs/conch_feasibility_check.md` records local smoke results
- [x] `docs/project_positioning.md` describes CONCH as optional
- [x] Documentation avoids clinical diagnosis claims
- [x] Documentation avoids claiming full WSI processing support
- [x] Documentation states that generated outputs and model weights should not be committed

## Release Notes Coverage

The v0.4.0 release notes cover:

- `CONCHWrapper`
- optional CONCH dependency and gated access requirements
- local feasibility results
- offline CI tests
- no bundled CONCH weights
- research and educational use only
- continued low-compute, patch-level positioning

## Claim Boundaries

The release documentation avoids claiming:

- CONCH results are benchmark certification
- outputs are clinical reports
- PathVLM-LiteBench supports raw WSI high-throughput processing
- PathVLM-LiteBench trains pathology foundation models
- generated reports should be committed as benchmark artifacts

Preferred wording:

- "optional CONCH wrapper"
- "gated Hugging Face access"
- "local smoke check"
- "frozen-model evaluation"
- "not clinical interpretation"

## Tag Criteria

Tag `v0.4.0` when:

- [x] Release blockers are resolved locally
- [x] `python -m pytest tests` passes locally
- [x] CLI smoke commands pass locally
- [x] CI passes on GitHub
- [x] README and docs are ready for GitHub-facing release documentation
- [x] `.gitignore` excludes local data and generated outputs
- [x] The latest release commit is pushed to `main`
- [x] `docs/release_notes_v0.4.0.md` is complete
