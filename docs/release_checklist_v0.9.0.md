# v0.9.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.9.0 reproducible prompt-scored patch-coordinate heatmap workflow release.

The goal of v0.9.0 is to make prompt-scored patch-coordinate heatmap runs easier to audit, validate before model loading, and compare from saved artifacts while keeping the project patch-level, local-first, and lightweight.

## Release Positioning

v0.9.0 positioning includes:

- run metadata output for prompt-scored patch-coordinate heatmap scoring
- `score-coordinate-heatmap --dry-run` validation without model loading
- `metadata_output` support in `task: patch_coordinate_heatmap_scoring` configs
- artifact-only score comparison through `compare-coordinate-heatmap-scores`
- comparison CSV output for multiple saved `scores.csv` artifacts
- optional Markdown comparison summaries for saved score artifacts
- automatic sibling `metadata.json` loading for score comparison when present
- command-line run names for comparison outputs
- row-count consistency checks for comparable prompt runs
- updated workflow documentation for metadata, dry-run, and artifact-only comparison
- continued support for v0.7.0 artifact-only heatmap rendering
- continued support for v0.8.0 model-backed prompt-scored heatmap generation

v0.9.0 positioning excludes:

- whole-slide image file reading
- slide tiling or tissue detection
- slide pyramid rendering
- clinical diagnostic claims
- tumor boundary identification claims
- bundled real pathology images or whole-slide image files
- required model downloads in CI
- required CONCH access
- Hugging Face token handling
- committed generated heatmaps, score CSV files, metadata JSON files, comparison artifacts, patches, reports, caches, model weights, or local datasets

## Release Scope

- [x] Add metadata sidecar output for `score-coordinate-heatmap`
- [x] Add `metadata_output` config support for `patch_coordinate_heatmap_scoring`
- [x] Add `--metadata-output` CLI override for scoring runs
- [x] Add `score-coordinate-heatmap --dry-run`
- [x] Ensure dry-run does not create a model, load images, run inference, or write outputs
- [x] Add artifact-only `compare-coordinate-heatmap-scores`
- [x] Add score summary CSV output for comparison runs
- [x] Add optional Markdown summary output for comparison runs
- [x] Support user-provided run names for comparison outputs
- [x] Support metadata-assisted comparison summaries
- [x] Reject missing score columns and mismatched row counts by default
- [x] Document metadata, dry-run, and artifact-only comparison workflows
- [x] Keep CI model-free with fake model tests and saved artifacts
- [x] Keep CONCH optional and gated
- [x] Keep generated outputs and local datasets ignored

## Current Release Blockers

- [x] Bump package version from `0.9.0.dev0` to `0.9.0` in `pyproject.toml`
- [x] Bump runtime version from `0.9.0.dev0` to `0.9.0` in `pathvlm_litebench/__init__.py`
- [x] Update version assertions in tests
- [x] Add `docs/release_notes_v0.9.0.md`
- [x] Add `docs/release_checklist_v0.9.0.md`
- [x] Add `docs/v0.9.0_pre_release_audit.md`
- [x] Update README current release links during release preparation
- [x] Verify local tests pass on the release preparation commit
- [x] Verify GitHub Actions CI passes on the release preparation commit
- [ ] Tag `v0.9.0` after release notes are reviewed
- [ ] Create the GitHub Release only after the tag and release notes are final

No v0.9.0 release should be cut while the package is still on a `.dev0` version.

## Validation Checklist

Run locally before tagging:

```powershell
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m pathvlm_litebench.cli version
.venv\Scripts\python.exe -m pathvlm_litebench.cli models
.venv\Scripts\python.exe -m pathvlm_litebench.cli demos
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_scoring_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli render-coordinate-heatmap --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli compare-coordinate-heatmap-scores --help
```

Optional artifact-only local check:

```powershell
.venv\Scripts\python.exe examples/05_patch_coordinate_heatmap_demo.py

.venv\Scripts\python.exe -m pathvlm_litebench.cli render-coordinate-heatmap `
  --manifest outputs/patch_coordinate_heatmap_demo_synthetic/coordinate_manifest.csv `
  --score-csv outputs/patch_coordinate_heatmap_demo_synthetic/scores.csv `
  --align-by image_path `
  --output outputs/patch_coordinate_heatmap_demo_synthetic/cli_heatmap.png
```

Optional comparison check after creating two saved scoring runs:

```powershell
.venv\Scripts\python.exe -m pathvlm_litebench.cli compare-coordinate-heatmap-scores `
  --score-csvs `
    outputs/patch_coordinate_heatmap_scored_tumor/scores.csv `
    outputs/patch_coordinate_heatmap_scored_lymphocyte/scores.csv `
  --run-names tumor lymphocyte `
  --output-csv outputs/patch_coordinate_heatmap_comparison/score_summary.csv `
  --output-md outputs/patch_coordinate_heatmap_comparison/score_summary.md
```

Generated artifacts from optional checks should remain uncommitted.

Manual model-backed checks are optional because `score-coordinate-heatmap` may download model weights when they are not already cached locally. Release CI should continue to cover this path with fake model wrappers.

## Data and Artifact Hygiene

Before release, verify that these are not committed:

- `dataset/`
- `outputs/`
- `examples/demo_patches/`
- synthetic demo outputs
- real pathology images
- whole-slide image files
- generated heatmaps
- generated score CSV files
- generated metadata JSON files
- generated comparison CSV or Markdown files
- generated reports
- model weights
- embedding cache files
- Hugging Face model caches
- Python build metadata

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

- [x] `docs/v0.9.0_plan.md` defines the milestone scope and non-goals
- [x] `docs/v0.9.0_pre_release_audit.md` records release-readiness checks
- [x] `docs/prompt_scored_coordinate_heatmap_workflow.md` documents metadata, dry-run, and score comparison
- [x] README links the v0.9.0 release notes, checklist, plan, audit, and workflow docs
- [x] Documentation explains the difference between artifact-only rendering, model-backed scoring, and artifact-only comparison
- [x] Documentation states `score-coordinate-heatmap` loads a model and may download weights unless `--dry-run` is used
- [x] Documentation states comparison reads saved artifacts and does not load models or images
- [x] Documentation states CI uses fake models and does not download weights
- [x] Documentation states generated outputs should not be committed
- [x] Documentation avoids clinical claims
- [x] Documentation avoids presenting the feature as whole-slide image processing
- [x] `docs/release_notes_v0.9.0.md` exists
- [x] `docs/release_notes_v0.9.0.md` is final before tagging

## Release Notes Coverage

The v0.9.0 release notes should cover:

- metadata sidecar output for scoring runs
- `--dry-run` validation for scoring runs
- `metadata_output` config support and CLI override
- artifact-only score comparison command
- comparison CSV and optional Markdown outputs
- metadata-assisted comparison summaries
- row-count consistency checks
- local-first artifact hygiene
- fake-model CI coverage with no model downloads
- no required CONCH access or Hugging Face token handling
- patch-coordinate visualization only, not whole-slide image processing
- research and educational use only

## Tag Criteria

Tag `v0.9.0` when:

- [x] Release blockers are resolved locally
- [x] Package and runtime versions are `0.9.0`
- [x] `docs/release_notes_v0.9.0.md` is complete
- [x] README current release links point to v0.9.0 docs
- [x] `.venv\Scripts\python.exe -m pytest tests` passes locally
- [x] CLI smoke commands pass locally
- [x] CI passes on GitHub for the release preparation commit
- [x] `.gitignore` excludes local data, generated outputs, and Python build metadata
- [x] The latest release preparation commit is pushed to `main`
