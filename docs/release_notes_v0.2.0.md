# v0.2.0 Release Notes

## Summary

PathVLM-LiteBench v0.2.0 extends the v0.1.0 CLIP baseline into a same-pipeline CLIP vs PLIP comparison workflow for low-compute computational pathology vision-language evaluation.

The release adds a CLIP-compatible PLIP wrapper, updates the model registry, documents local PLIP feasibility checks, and records a full-test MHIST zero-shot observation comparing CLIP and PLIP under the same patch-level evaluation pipeline.

This release is intended for research and educational use only.

## What Is New Since v0.1.0

Model workflow:

- added `PLIPWrapper`
- added implemented `plip` model key resolving to `vinid/plip`
- kept `clip` and `clip-vit-base-patch32` as CLIP baseline keys
- kept `conch` as a placeholder with a clear `NotImplementedError`
- reused the CLIP-compatible Hugging Face processor/model interface for PLIP
- verified PLIP wrapper embeddings against Hugging Face `text_embeds` and `image_embeds`

Local feasibility and documentation:

- documented PLIP access and compatibility checks
- documented CPU and CUDA smoke checks for PLIP
- documented local PLIP memory behavior on a laptop GPU smoke test
- added a CLIP vs PLIP MHIST comparison protocol
- added a full-test CLIP vs PLIP MHIST observation document
- updated README with the full-test MHIST observation
- added v0.2.0 release readiness notes

## MHIST CLIP vs PLIP Observation

A local full-test MHIST zero-shot comparison showed opposite class-bias patterns:

- CLIP favored `HP`.
- PLIP favored `SSA`.
- CLIP had higher raw accuracy because it predicted the majority `HP` class for most samples.
- PLIP improved balanced accuracy and `SSA` recall in this run, while still showing strong `SSA` prediction bias.
- PLIP class-order swapping did not change the metrics.
- PLIP synonym prompts reduced the `SSA` prediction skew and removed the prediction-collapse warning in this run.

This observation is not a clinical result and should not be treated as proof that PLIP solves MHIST. It motivates the toolkit's emphasis on balanced accuracy, macro-F1, per-class recall, confusion matrices, predicted-label distributions, saved errors, prediction-collapse warnings, and prompt-sensitivity checks.

See:

- `docs/clip_vs_plip_mhist_protocol.md`
- `docs/clip_vs_plip_mhist_observation.md`
- `docs/plip_feasibility_check.md`

## What Remains Unchanged

The toolkit remains focused on:

- frozen model inference
- patch-level images
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- lightweight tests and CI

The release still includes the v0.1.0 workflows:

- patch-text retrieval
- manifest-based retrieval evaluation with Recall@K
- zero-shot patch classification
- classification metrics beyond accuracy
- zero-shot predictions, metrics, and error reports
- prediction distribution and collapse warnings
- prompt sensitivity analysis
- manifest conversion and sampling
- embedding cache
- visualization and HTML retrieval reports
- config-driven retrieval, zero-shot, and prompt sensitivity demos
- CLI entry point
- pytest tests
- GitHub Actions CI

## What Is Intentionally Not Included

v0.2.0 does not include:

- CONCH implementation
- full WSI high-throughput processing
- large-scale VLM pretraining
- clinical diagnosis or clinical decision support
- bundled public datasets
- real pathology images
- model weights
- generated reports

PLIP model weights are not included in the repository. They are downloaded locally through Hugging Face when users run PLIP-based workflows.

## Verification

Recommended verification before tagging:

```bash
python -m pytest tests
python -m pathvlm_litebench.cli version
python -m pathvlm_litebench.cli models
python -m pathvlm_litebench.cli demos
```

The lightweight CLI commands and CI tests do not download models by default.

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

## Suggested GitHub Release Text

PathVLM-LiteBench v0.2.0 adds PLIP support to the existing low-compute CLIP baseline toolkit for patch-level computational pathology vision-language evaluation.

Highlights:

- PLIP wrapper through `vinid/plip`
- same-pipeline CLIP vs PLIP evaluation
- full-test MHIST zero-shot observation
- class-bias and prediction-collapse analysis
- prompt-sensitivity sanity checks
- continued support for retrieval, zero-shot classification, prompt sensitivity, manifest utilities, reports, configs, CLI, tests, and CI

This release is intended for research and educational use only. It is not a clinical diagnostic tool. PLIP is included as a pathology-specific comparison model, while CONCH remains a planned placeholder.
