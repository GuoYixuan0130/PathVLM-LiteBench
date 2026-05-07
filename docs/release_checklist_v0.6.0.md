# v0.6.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.6.0 reproducible MHIST benchmark audit release.

The goal of v0.6.0 is to make the v0.5.0 public patch benchmark workflow easier to rerun, audit, and interpret locally while keeping the project low-compute, patch-level, and local-first.

## Release Positioning

v0.6.0 positioning includes:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level benchmark workflow
- frozen vision-language model inference
- CLIP, PLIP, and optional CONCH evaluation workflows
- committed sampled MHIST zero-shot baseline configs
- non-overwriting zero-shot prompt-grid output overrides
- local MHIST baseline and prompt-grid audit documentation
- local model-behavior observations, not clinical claims

v0.6.0 positioning excludes:

- a clinical diagnostic system
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- required CONCH downloads or bundled CONCH weights
- committed datasets, generated reports, run logs, or prediction files
- automatic Hugging Face token handling

## Release Scope

- [x] Add v0.5.0 post-release audit documentation
- [x] Add sampled MHIST CLIP and PLIP baseline audit documentation
- [x] Add sampled MHIST CLIP and PLIP zero-shot baseline configs
- [x] Validate committed MHIST baseline configs in offline tests
- [x] Add `run-zero-shot-grid --output-root`
- [x] Add `run-zero-shot-grid --comparison-output`
- [x] Document non-overwriting prompt-grid output overrides
- [x] Add sampled MHIST CLIP/PLIP/CONCH prompt-grid audit documentation
- [x] Polish public audit docs to remove internal follow-up sections
- [x] Keep CI offline and lightweight

## Current Release Blockers

- [x] Bump package version from `0.5.0` to `0.6.0` in `pyproject.toml`
- [x] Bump runtime version from `0.5.0` to `0.6.0` in `pathvlm_litebench/__init__.py`
- [x] Add `docs/release_notes_v0.6.0.md`
- [x] Add this release checklist
- [ ] Verify GitHub Actions CI passes on the release commit
- [ ] Decide whether to tag `v0.6.0` after the release notes are reviewed

## Validation Checklist

Run locally before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli validate-config configs/zero_shot_mhist_clip_sample.json
python -m pathvlm_litebench.cli validate-config configs/zero_shot_mhist_plip_sample.json
python -m pathvlm_litebench.cli validate-config configs/zero_shot_prompt_grid_mhist_sample.json
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --output-root outputs/zero_shot_prompt_grid_mhist_sample_release_check --dry-run
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
```

Optional local benchmark checks:

```bash
python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_clip_sample.json
python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_plip_sample.json
pathvlm-litebench run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --output-root outputs/zero_shot_prompt_grid_mhist_sample_v060_audit
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
- Python build metadata
- real pathology images
- generated reports
- generated run logs
- generated prompt-grid comparison Markdown files

Expected `.gitignore` coverage:

```text
/dataset/
/outputs/
/examples/demo_patches/
*.egg-info/
build/
dist/
*.pt
*.pth
*.ckpt
*.npy
*.npz
```

## Documentation Checklist

- [x] README links v0.6.0 release notes and checklist when the version is bumped
- [x] README links v0.6.0 plan and audit docs
- [x] Public audit docs avoid internal follow-up sections
- [x] Documentation states generated outputs should not be committed
- [x] Documentation states optional CONCH still requires gated Hugging Face access
- [x] Documentation avoids clinical claims

## Release Notes Coverage

The v0.6.0 release notes cover:

- v0.5.0 post-release audit
- sampled MHIST baseline audit
- committed CLIP and PLIP sampled MHIST configs
- prompt-grid output overrides
- sampled MHIST CLIP/PLIP/CONCH prompt-grid audit
- no model downloads in CI
- optional CONCH remains optional and gated
- research and educational use only

## Tag Criteria

Tag `v0.6.0` when:

- [ ] Release blockers are resolved locally
- [x] `python -m pytest tests` passes locally
- [x] CLI smoke commands pass locally
- [ ] CI passes on GitHub
- [ ] README and docs are ready for GitHub-facing release documentation
- [x] `.gitignore` excludes local data, generated outputs, and Python build metadata
- [ ] The latest release commit is pushed to `main`
- [x] `docs/release_notes_v0.6.0.md` is complete
