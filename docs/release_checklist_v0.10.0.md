# v0.10.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.10.0 prompt-set patch-coordinate heatmap workflow release.

The goal of v0.10.0 is to make repeated prompt-scored patch-coordinate heatmap runs easier to configure, dry-run, execute, and compare while keeping the project patch-level, local-first, and lightweight.

## Release Positioning

v0.10.0 positioning includes:

- prompt-set config support for repeated prompt-scored coordinate heatmap runs
- `validate-config` support for `task: patch_coordinate_heatmap_prompt_set`
- committed prompt-set example config
- `score-coordinate-heatmap-prompt-set --dry-run`
- model-free and image-loading-free prompt-set dry-run behavior
- model-backed prompt-set scoring for several prompts over one coordinate manifest
- per-prompt `scores.csv`, `heatmap.png`, and `metadata.json` outputs
- per-prompt metadata with prompt key, prompt text, model, device, manifest, artifact paths, and patch count
- artifact-only prompt-set comparison CSV and Markdown summaries
- CLI overrides for prompt-set output root, comparison paths, and max-images smoke runs
- prompt-set workflow documentation
- continued support for v0.7.0 artifact-only heatmap rendering
- continued support for v0.8.0 single-prompt model-backed scoring
- continued support for v0.9.0 scoring metadata, dry-run, and artifact-only score comparison

v0.10.0 positioning excludes:

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

- [x] Add prompt-set heatmap config dataclasses and JSON helpers
- [x] Add prompt-set config serialization and validation tests
- [x] Add committed prompt-set example config
- [x] Add `validate-config` support for `patch_coordinate_heatmap_prompt_set`
- [x] Add `score-coordinate-heatmap-prompt-set --dry-run`
- [x] Ensure prompt-set dry-run does not create a model, load images, run inference, or write outputs
- [x] Add prompt-set scoring through existing single-prompt scoring helpers
- [x] Save per-prompt `scores.csv`, `heatmap.png`, and `metadata.json`
- [x] Record prompt key in prompt-set metadata output
- [x] Add artifact-only prompt-set comparison CSV and Markdown summaries
- [x] Add CLI overrides for output root, comparison output paths, and max-images
- [x] Keep CI model-free with fake model tests
- [x] Document prompt-set dry-run, scoring, output layout, model-loading behavior, and artifact hygiene
- [x] Keep CONCH optional and gated
- [x] Keep generated outputs and local datasets ignored

## Current Release Blockers

- [ ] Bump package version from `0.10.0.dev0` to `0.10.0` in `pyproject.toml`
- [ ] Bump runtime version from `0.10.0.dev0` to `0.10.0` in `pathvlm_litebench/__init__.py`
- [ ] Update version assertions in tests
- [x] Add `docs/release_notes_v0.10.0.md`
- [x] Add `docs/release_checklist_v0.10.0.md`
- [ ] Add `docs/v0.10.0_pre_release_audit.md`
- [ ] Update README current release links during release preparation
- [ ] Verify local tests pass on the release preparation commit
- [ ] Verify GitHub Actions CI passes on the release preparation commit
- [ ] Tag `v0.10.0` after release notes are reviewed
- [ ] Create the GitHub Release only after the tag and release notes are final

No v0.10.0 release should be cut while the package is still on a `.dev0` version.

## Validation Checklist

Run locally before tagging:

```powershell
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m pathvlm_litebench.cli version
.venv\Scripts\python.exe -m pathvlm_litebench.cli models
.venv\Scripts\python.exe -m pathvlm_litebench.cli demos
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_scoring_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli validate-config configs/patch_coordinate_heatmap_prompt_set_demo_config.json
.venv\Scripts\python.exe -m pathvlm_litebench.cli render-coordinate-heatmap --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap --help
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap-prompt-set --help
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

Optional prompt-set smoke run:

```powershell
.venv\Scripts\python.exe -m pathvlm_litebench.cli score-coordinate-heatmap-prompt-set `
  --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json `
  --dry-run `
  --max-images 16
```

Manual model-backed checks are optional because `score-coordinate-heatmap` and `score-coordinate-heatmap-prompt-set` may download model weights when they are not already cached locally. Release CI should continue to cover model-backed paths with fake model wrappers.

Generated artifacts from optional checks should remain uncommitted.

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

- [x] `docs/v0.10.0_plan.md` defines the milestone scope and non-goals
- [x] `docs/prompt_set_coordinate_heatmap_workflow.md` documents prompt-set config, dry-run, scoring, output layout, and comparison summaries
- [x] README links the v0.10.0 plan and prompt-set workflow docs
- [x] Documentation explains the difference between artifact-only rendering, single-prompt scoring, prompt-set scoring, and artifact-only comparison
- [x] Documentation states prompt-set dry-run does not load models or images
- [x] Documentation states prompt-set scoring loads a model and may download weights unless `--dry-run` is used
- [x] Documentation states comparison reads saved artifacts and does not load models or images
- [x] Documentation states CI uses fake models and does not download weights
- [x] Documentation states generated outputs should not be committed
- [x] Documentation avoids clinical claims
- [x] Documentation avoids presenting the feature as whole-slide image processing
- [x] `docs/release_notes_v0.10.0.md` exists
- [ ] README current release links point to v0.10.0 docs during release preparation
- [ ] `docs/v0.10.0_pre_release_audit.md` records release-readiness checks
- [ ] `docs/release_notes_v0.10.0.md` is final before tagging

## Release Notes Coverage

The v0.10.0 release notes should cover:

- prompt-set config support
- prompt-set config validation
- prompt-set dry-run behavior
- prompt-set batch scoring
- per-prompt score, heatmap, and metadata artifacts
- prompt key metadata
- artifact-only prompt-set comparison summaries
- CLI overrides for smoke runs and output paths
- local-first artifact hygiene
- fake-model CI coverage with no model downloads
- no required CONCH access or Hugging Face token handling
- patch-coordinate visualization only, not whole-slide image processing
- research and educational use only

## Tag Criteria

Tag `v0.10.0` when:

- [ ] Release blockers are resolved locally
- [ ] Package and runtime versions are `0.10.0`
- [ ] `docs/release_notes_v0.10.0.md` is complete
- [ ] README current release links point to v0.10.0 docs
- [ ] `.venv\Scripts\python.exe -m pytest tests` passes locally
- [ ] CLI smoke commands pass locally
- [ ] CI passes on GitHub for the release preparation commit
- [ ] `.gitignore` excludes local data, generated outputs, and Python build metadata
- [ ] The latest release preparation commit is pushed to `main`
