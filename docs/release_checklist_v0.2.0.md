# v0.2.0 Release Readiness Checklist

This checklist defines what should be true before presenting PathVLM-LiteBench as a v0.2.0 research engineering release.

The goal of v0.2.0 is to extend the v0.1.0 CLIP baseline into a same-pipeline CLIP vs PLIP comparison workflow while keeping the project low-compute, patch-level, and laptop-friendly.

## Release Positioning

v0.2.0 should be described as:

- a CPU-compatible and CUDA-accelerated toolkit
- a low-compute patch-level evaluation workflow
- a frozen vision-language model inference toolkit
- a CLIP baseline plus PLIP pathology-specific comparison workflow
- a toolkit for evaluating model behavior, prompt sensitivity, and class-bias failure modes
- a research and educational tool

v0.2.0 should not be described as:

- a clinical diagnostic system
- a CPU-only project
- a full WSI high-throughput processing framework
- a large-scale VLM pretraining framework
- proof that PLIP solves MHIST
- an implementation of CONCH

## Scope Since v0.1.0

- [x] Add `PLIPWrapper` using the CLIP-compatible Hugging Face PLIP checkpoint
- [x] Mark `plip` as implemented in the model registry
- [x] Keep `conch` registered as a placeholder with a clear `NotImplementedError`
- [x] Verify PLIP image and text embeddings against Hugging Face `text_embeds` / `image_embeds`
- [x] Run local PLIP CPU / CUDA smoke checks
- [x] Document PLIP feasibility checks
- [x] Add a CLIP vs PLIP MHIST comparison protocol
- [x] Run full-test MHIST zero-shot comparison for CLIP and PLIP
- [x] Run full-test PLIP class-order sanity check
- [x] Run full-test PLIP prompt-sensitivity sanity check
- [x] Update README with the full-test MHIST observation
- [x] Document preliminary CLIP vs PLIP MHIST observations

## Current Release Blockers

- [ ] Bump package version from `0.1.0` to `0.2.0` in `pyproject.toml`
- [ ] Bump runtime version from `0.1.0` to `0.2.0` in `pathvlm_litebench/__init__.py`
- [x] Add `docs/release_notes_v0.2.0.md`
- [ ] Verify GitHub Actions CI passes on the release commit
- [ ] Decide whether to tag `v0.2.0` after the release notes are reviewed

Do not tag v0.2.0 until the blockers above are resolved.

## Validation Checklist

Run locally before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
```

Optional local CUDA check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

Optional local PLIP smoke check:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest.csv \
  --image_root dataset/MHIST/images \
  --model plip \
  --device cuda \
  --split test \
  --class_names HP SSA \
  --class_prompts \
    "a histopathology image of hyperplastic polyp" \
    "a histopathology image of sessile serrated adenoma" \
  --top_k 2 \
  --save_report \
  --report_dir outputs/zero_shot_plip_mhist_full
```

Do not add PLIP model downloads or MHIST runs to CI.

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

- [x] README states CPU-compatible plus laptop-GPU accelerated positioning
- [x] README lists PLIP as implemented and CONCH as placeholder
- [x] README links to the CLIP vs PLIP MHIST observation
- [x] `docs/plip_feasibility_check.md` records local PLIP feasibility checks
- [x] `docs/clip_vs_plip_mhist_protocol.md` explains how to reproduce the comparison
- [x] `docs/clip_vs_plip_mhist_observation.md` records full-test results and sanity checks
- [x] Documentation avoids clinical diagnosis claims
- [x] Documentation avoids claiming full WSI processing support
- [x] Documentation states that generated outputs and real pathology data should not be committed

## Release Notes Draft Points

The v0.2.0 release notes should mention:

- PLIP support through `vinid/plip`
- CLIP-compatible PLIP wrapper implementation
- unchanged CONCH placeholder status
- full-test MHIST CLIP vs PLIP observation
- class-bias and prediction-collapse findings
- prompt-sensitivity sanity checks
- research and educational use only
- PLIP weights are downloaded locally through Hugging Face when used
- tests and CI remain lightweight and do not download models

## Claims to Avoid

Do not claim:

- PLIP is clinically reliable
- PLIP diagnoses MHIST correctly
- PLIP is uniformly better than CLIP
- PathVLM-LiteBench supports raw WSI high-throughput processing
- PathVLM-LiteBench trains pathology foundation models
- the local MHIST observation is a definitive benchmark result

Use cautious language:

- "local full-test MHIST zero-shot observation"
- "PLIP improved balanced accuracy and SSA recall in this run, while still showing SSA prediction bias"
- "CLIP and PLIP showed opposite class-bias patterns under simple prompts"
- "prompt wording affected PLIP prediction distribution"

## Suggested Tag Criteria

Tag v0.2.0 when:

- [ ] Release blockers are resolved
- [ ] `python -m pytest tests` passes locally
- [ ] CLI smoke commands pass locally
- [ ] CI passes on GitHub
- [ ] README and docs render correctly on GitHub
- [ ] `.gitignore` excludes local data and generated outputs
- [ ] The latest release commit is pushed to `main`
- [x] `docs/release_notes_v0.2.0.md` is complete
