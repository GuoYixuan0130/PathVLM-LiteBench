# v0.5.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.5.0 public patch benchmark workflow release.

The goal of v0.5.0 is to make the existing local tools easier to run on a reproducible public patch-level pathology dataset while keeping the project low-compute, patch-level, and local-first.

## Release Positioning

v0.5.0 positioning includes:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level benchmark workflow
- frozen vision-language model inference
- CLIP, PLIP, and optional CONCH evaluation workflows
- manifest conversion, balanced sampling, zero-shot runs, prompt grids, and report comparison
- config validation that does not load datasets or model weights
- local model-behavior observations, not clinical claims

v0.5.0 positioning excludes:

- a clinical diagnostic system
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- required CONCH downloads or bundled CONCH weights
- committed datasets, generated reports, or prediction files
- automatic Hugging Face token handling

## Release Scope

- [x] Add public patch benchmark workflow documentation
- [x] Add zero-shot prompt-grid workflow documentation
- [x] Add sample MHIST prompt-grid config
- [x] Add tests for committed example configs
- [x] Add `validate-config` CLI command
- [x] Validate zero-shot prompt-grid configs in dry-run mode without loading models
- [x] Keep CI offline and lightweight

## Current Release Blockers

- [x] Bump package version from `0.4.1` to `0.5.0` in `pyproject.toml`
- [x] Bump runtime version from `0.4.1` to `0.5.0` in `pathvlm_litebench/__init__.py`
- [x] Add `docs/release_notes_v0.5.0.md`
- [x] Add this release checklist
- [ ] Verify GitHub Actions CI passes on the release commit
- [ ] Decide whether to tag `v0.5.0` after the release notes are reviewed

## Validation Checklist

Run locally before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli validate-config configs/zero_shot_prompt_grid_mhist_sample.json
python -m pathvlm_litebench.cli run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --dry-run
python -m pathvlm_litebench.cli summarize-report --help
python -m pathvlm_litebench.cli compare-reports --help
```

Optional local benchmark checks:

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

- [x] README links v0.5.0 release notes and checklist
- [x] README links the public patch benchmark workflow
- [x] README links the prompt-grid workflow
- [x] Documentation states generated outputs should not be committed
- [x] Documentation states optional CONCH still requires gated Hugging Face access
- [x] Documentation avoids clinical claims

## Release Notes Coverage

The v0.5.0 release notes cover:

- public patch benchmark workflow documentation
- prompt-grid workflow documentation
- `validate-config`
- example config validation tests
- no model downloads in CI
- optional CONCH remains optional and gated
- research and educational use only

## Tag Criteria

Tag `v0.5.0` when:

- [x] Release blockers are resolved locally
- [x] `python -m pytest tests` passes locally
- [x] CLI smoke commands pass locally
- [ ] CI passes on GitHub
- [x] README and docs are ready for GitHub-facing release documentation
- [x] `.gitignore` excludes local data and generated outputs
- [ ] The latest release commit is pushed to `main`
- [x] `docs/release_notes_v0.5.0.md` is complete
