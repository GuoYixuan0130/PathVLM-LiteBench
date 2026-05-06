# v0.4.1 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.4.1 research engineering patch release.

The goal of v0.4.1 is to add a reproducible zero-shot prompt-grid workflow and document CLIP/PLIP/CONCH sampled MHIST behavior after the v0.4.0 CONCH release.

## Release Positioning

v0.4.1 positioning includes:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level evaluation workflow
- frozen vision-language model inference
- CLIP, PLIP, and optional CONCH evaluation workflows
- zero-shot prompt-grid batch runs
- structured report saving and Markdown comparison summaries
- local model-behavior observations, not clinical claims

v0.4.1 positioning excludes:

- a clinical diagnostic system
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- required CONCH downloads or bundled CONCH weights
- committed generated reports or prediction files

## Release Scope

- [x] Add `run-zero-shot-grid` CLI command
- [x] Add zero-shot prompt-grid JSON config loading
- [x] Add dry-run support
- [x] Add example MHIST sampled prompt-grid config
- [x] Add comparison Markdown generation for prompt-grid runs
- [x] Add offline tests for prompt-grid expansion and fake-run comparison output
- [x] Document sampled CLIP vs PLIP vs CONCH zero-shot behavior
- [x] Document sampled CLIP/PLIP/CONCH prompt-grid behavior

## Current Release Blockers

- [x] Bump package version from `0.4.0` to `0.4.1` in `pyproject.toml`
- [x] Bump runtime version from `0.4.0` to `0.4.1` in `pathvlm_litebench/__init__.py`
- [x] Add `docs/release_notes_v0.4.1.md`
- [x] Add this release checklist
- [x] Verify GitHub Actions CI passes on the release commit
- [ ] Decide whether to tag `v0.4.1` after the release notes are reviewed

## Validation Checklist

Run locally before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
python -m pathvlm_litebench.cli run-zero-shot-grid --help
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --dry-run
```

Optional local prompt-grid checks:

```bash
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json
```

These optional checks depend on local datasets, model caches, Hugging Face access, and hardware. They should not be added to CI.

## Data and Artifact Hygiene

Before release, verify that these are not committed:

- `dataset/`
- `outputs/`
- `examples/demo_patches/`
- Hugging Face model caches
- model weights
- embedding cache files
- real pathology images
- generated reports
- generated prompt-grid comparison Markdown files

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

## Documentation Checklist

- [x] README documents `run-zero-shot-grid`
- [x] README links v0.4.1 release notes and checklist
- [x] Prompt-grid observation document avoids clinical claims
- [x] Documentation states generated outputs should not be committed
- [x] Documentation states optional CONCH still requires gated access

## Release Notes Coverage

The v0.4.1 release notes cover:

- `run-zero-shot-grid`
- dry-run behavior
- example config
- offline tests
- sampled CLIP/PLIP/CONCH observations
- no model downloads in CI
- research and educational use only

## Tag Criteria

Tag `v0.4.1` when:

- [x] Release blockers are resolved locally
- [x] `python -m pytest tests` passes locally
- [x] CLI smoke commands pass locally
- [x] CI passes on GitHub
- [x] README and docs are ready for GitHub-facing release documentation
- [x] `.gitignore` excludes local data and generated outputs
- [x] The latest release commit is pushed to `main`
- [x] `docs/release_notes_v0.4.1.md` is complete
