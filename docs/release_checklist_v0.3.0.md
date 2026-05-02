# v0.3.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.3.0 research engineering release.

The goal of v0.3.0 is to add local Markdown experiment summaries for saved evaluation reports while keeping the project low-compute, patch-level, and laptop-friendly.

## Release Positioning

v0.3.0 positioning includes:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level evaluation workflow
- a frozen vision-language model inference toolkit
- CLIP and PLIP evaluation workflows
- structured report saving for retrieval, zero-shot classification, and prompt sensitivity
- Markdown summaries for saved report artifacts
- a research and educational tool

v0.3.0 positioning excludes:

- a clinical diagnostic system
- a CPU-only project
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- dashboard or experiment-tracking infrastructure
- automatic dataset or output upload
- an implementation of CONCH

## Release Scope

- [x] Add `summarize-report` CLI command
- [x] Add zero-shot report summary support
- [x] Add retrieval report summary support
- [x] Add prompt sensitivity report summary support
- [x] Read existing CSV/JSON report artifacts only
- [x] Write local `experiment_summary.md`
- [x] Avoid model inference, image loading, and model downloads during summary generation
- [x] Add tests with synthetic report artifacts
- [x] Validate summary generation on local real-output report directories
- [x] Update README and real patch workflow documentation

## Current Release Blockers

- [x] Bump package version from `0.2.0` to `0.3.0` in `pyproject.toml`
- [x] Bump runtime version from `0.2.0` to `0.3.0` in `pathvlm_litebench/__init__.py`
- [x] Add `docs/release_notes_v0.3.0.md`
- [x] Verify GitHub Actions CI passes on the release commit
- [x] Decide whether to tag `v0.3.0` after the release notes are reviewed

## Validation Checklist

Run locally before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
python -m pathvlm_litebench.cli summarize-report --help
```

Optional local summary checks:

```bash
pathvlm-litebench summarize-report --task zero-shot --report_dir outputs/zero_shot_demo
pathvlm-litebench summarize-report --task retrieval --report_dir outputs/retrieval_demo
pathvlm-litebench summarize-report --task prompt-sensitivity --report_dir outputs/prompt_sensitivity_demo
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

- [x] README documents zero-shot summary usage
- [x] README documents retrieval summary usage
- [x] README documents prompt sensitivity summary usage
- [x] `docs/real_patch_workflow.md` documents all three summary commands
- [x] `docs/v0.3.0_plan.md` reflects implemented summary support
- [x] Documentation states that summaries read saved artifacts only
- [x] Documentation avoids clinical diagnosis claims
- [x] Documentation avoids claiming full WSI processing support
- [x] Documentation states that generated outputs and real pathology data should not be committed

## Release Notes Coverage

The v0.3.0 release notes cover:

- `summarize-report` CLI command
- zero-shot report summary support
- retrieval report summary support
- prompt sensitivity report summary support
- local artifact-only behavior
- no model inference or model downloads during summary generation
- generated summaries are local outputs
- research and educational use only
- tests and CI remain lightweight

## Claim Boundaries

The release documentation avoids claiming:

- summaries are benchmark certification
- summaries are clinical reports
- PathVLM-LiteBench supports raw WSI high-throughput processing
- PathVLM-LiteBench trains pathology foundation models
- generated summaries should be committed as benchmark artifacts

Preferred wording:

- "local Markdown experiment summary"
- "saved report artifacts"
- "post-run review"
- "model-behavior summary"
- "not clinical interpretation"

## Tag Criteria

Tag `v0.3.0` when:

- [x] Release blockers are resolved
- [x] `python -m pytest tests` passes locally
- [x] CLI smoke commands pass locally
- [x] CI passes on GitHub
- [x] README and docs are ready for GitHub-facing release documentation
- [x] `.gitignore` excludes local data and generated outputs
- [x] The latest release commit is pushed to `main`
- [x] `docs/release_notes_v0.3.0.md` is complete
