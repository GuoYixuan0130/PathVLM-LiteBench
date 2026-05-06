# v0.3.1 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.3.1 research engineering patch release.

The goal of v0.3.1 is to add local Markdown comparisons for multiple saved report directories while preserving the v0.3.0 low-compute, patch-level reporting workflow.

## Release Positioning

v0.3.1 positioning includes:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level evaluation workflow
- a frozen vision-language model inference toolkit
- CLIP and PLIP evaluation workflows
- structured report saving for retrieval, zero-shot classification, and prompt sensitivity
- Markdown summaries for individual saved report directories
- Markdown comparisons for multiple saved report directories
- a research and educational tool

v0.3.1 positioning excludes:

- a clinical diagnostic system
- a CPU-only project
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- dashboard or experiment-tracking infrastructure
- automatic dataset or output upload
- an implementation of CONCH

## Release Scope

- [x] Add `compare-reports` CLI command
- [x] Add zero-shot report comparison support
- [x] Add retrieval report comparison support
- [x] Add prompt sensitivity report comparison support
- [x] Read existing CSV/JSON report artifacts only
- [x] Write local Markdown comparison files
- [x] Avoid model inference, image loading, and model downloads during comparison generation
- [x] Add tests with synthetic report artifacts
- [x] Update README and real patch workflow documentation

## Current Release Blockers

- [x] Bump package version from `0.3.0` to `0.3.1` in `pyproject.toml`
- [x] Bump runtime version from `0.3.0` to `0.3.1` in `pathvlm_litebench/__init__.py`
- [x] Add `docs/release_notes_v0.3.1.md`
- [x] Add this release checklist
- [ ] Verify GitHub Actions CI passes on the release commit
- [ ] Decide whether to tag `v0.3.1` after the release notes are reviewed

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

Optional local comparison checks:

```bash
pathvlm-litebench compare-reports \
  --task zero-shot \
  --report_dirs outputs/zero_shot_clip outputs/zero_shot_plip \
  --run_names clip plip \
  --output outputs/zero_shot_comparison.md
```

These optional checks depend on local generated reports and should not be added to CI.

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
- generated `experiment_summary.md` files
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

- [x] README documents `compare-reports` usage
- [x] `docs/real_patch_workflow.md` documents `compare-reports`
- [x] Documentation states that comparisons read saved artifacts only
- [x] Documentation avoids clinical diagnosis claims
- [x] Documentation avoids claiming full WSI processing support
- [x] Documentation states that generated outputs and real pathology data should not be committed

## Release Notes Coverage

The v0.3.1 release notes cover:

- `compare-reports` CLI command
- zero-shot comparison summary support
- retrieval comparison summary support
- prompt sensitivity comparison summary support
- local artifact-only behavior
- no model inference or model downloads during comparison generation
- generated comparisons are local outputs
- research and educational use only
- tests and CI remain lightweight

## Claim Boundaries

The release documentation avoids claiming:

- comparisons are benchmark certification
- comparisons are clinical reports
- PathVLM-LiteBench supports raw WSI high-throughput processing
- PathVLM-LiteBench trains pathology foundation models
- generated comparisons should be committed as benchmark artifacts

Preferred wording:

- "local Markdown comparison"
- "saved report artifacts"
- "post-run review"
- "model-behavior comparison"
- "not clinical interpretation"

## Tag Criteria

Tag `v0.3.1` when:

- [x] Release blockers are resolved locally
- [x] `python -m pytest tests` passes locally
- [x] CLI smoke commands pass locally
- [ ] CI passes on GitHub
- [x] README and docs are ready for GitHub-facing release documentation
- [x] `.gitignore` excludes local data and generated outputs
- [ ] The latest release commit is pushed to `main`
- [x] `docs/release_notes_v0.3.1.md` is complete
