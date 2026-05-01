# CLIP vs PLIP MHIST Preliminary Observation

This note records a local, low-compute zero-shot comparison between CLIP and PLIP on a sampled MHIST test manifest.

It is an engineering and research observation, not a clinical result. The goal is to document model behavior, prompt sensitivity, and prediction-collapse signals under the same PathVLM-LiteBench evaluation pipeline.

## Setup

Dataset:

- MHIST / Dartmouth Colorectal Polyps Histopathology Binary Classification Dataset
- Standard manifest format: `image_path,label,split,case_id`
- Sampled test manifest: `dataset/MHIST/manifest_test_50_per_class.csv`
- 100 total samples: 50 `HP`, 50 `SSA`

Task:

- Zero-shot patch classification
- Frozen model inference only
- No fine-tuning
- CUDA laptop GPU execution
- Reports saved locally under `outputs/`

Models:

| Model key | Model | Notes |
|---|---|---|
| `clip` | `openai/clip-vit-base-patch32` | General CLIP baseline |
| `plip` | `vinid/plip` | Pathology-specific CLIP-compatible wrapper |

Default class prompts:

| Class | Prompt |
|---|---|
| `HP` | `a histopathology image of hyperplastic polyp` |
| `SSA` | `a histopathology image of sessile serrated adenoma` |

## Local Results

| Run | Accuracy | Balanced accuracy | Macro-F1 | HP recall | SSA recall | Predicted distribution |
|---|---:|---:|---:|---:|---:|---|
| CLIP, default prompts | 0.5000 | 0.5000 | 0.4048 | 0.9000 | 0.1000 | `HP=90, SSA=10` |
| PLIP, default prompts | 0.5500 | 0.5500 | 0.4357 | 0.1000 | 1.0000 | `SSA=95, HP=5` |
| PLIP, class order swapped | 0.5500 | 0.5500 | 0.4357 | 0.1000 | 1.0000 | `SSA=95, HP=5` |
| PLIP, synonym prompts | 0.5200 | 0.5200 | 0.4792 | 0.2400 | 0.8000 | `SSA=78, HP=22` |

Synonym prompts:

| Class | Prompt |
|---|---|
| `HP` | `an H&E histopathology patch showing a hyperplastic colorectal polyp` |
| `SSA` | `an H&E histopathology patch showing a sessile serrated lesion` |

## Confusion Matrices

Rows are true labels and columns are predicted labels.

CLIP, default prompts:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 45 | 5 |
| SSA | 45 | 5 |

PLIP, default prompts:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 5 | 45 |
| SSA | 0 | 50 |

PLIP, synonym prompts:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 12 | 38 |
| SSA | 10 | 40 |

## Sanity Checks

PLIP embedding extraction was checked against the Hugging Face `CLIPModel` forward output:

- `PLIPWrapper.encode_text(...)` matched `forward.text_embeds` after normalization.
- `PLIPWrapper.encode_images(...)` matched `forward.image_embeds` after normalization.
- Maximum absolute differences were approximately `1e-8`.

The PLIP class-order swap produced the same metrics and prediction distribution as the default PLIP run. This suggests the observed `SSA` bias is not caused by class-name ordering or label mapping.

Changing PLIP prompts reduced, but did not eliminate, the `SSA` prediction bias. This suggests the result is prompt-sensitive and should be treated as a preliminary zero-shot observation rather than a fixed property of the model.

## Interpretation

The sampled run suggests that both CLIP and PLIP can show strong prediction bias on MHIST under simple zero-shot prompts:

- CLIP default prompts strongly favored `HP`.
- PLIP default prompts strongly favored `SSA`.
- PLIP remained sensitive to prompt wording.
- Accuracy alone hid important failure modes.

This does not mean that PLIP is clinically reliable, and it does not prove that PLIP solves MHIST. It shows why PathVLM-LiteBench reports balanced accuracy, macro-F1, per-class metrics, confusion matrices, predicted-label distributions, error files, and prediction-collapse warnings.

## Project Implication

This observation supports the current project direction:

- Keep CLIP as the general baseline.
- Use PLIP as a pathology-specific comparison model through the same wrapper interface.
- Evaluate model behavior with the same manifests, prompts, metrics, and report format.
- Treat prompt sensitivity as part of the benchmark rather than an afterthought.
- Keep the workflow laptop-friendly by using sampled manifests before scaling to larger local runs.

## Limitations

- This is one local sampled run, not a finalized benchmark.
- Prompt wording can change the result.
- Model versions, package versions, preprocessing, and sampling seed can affect metrics.
- MHIST labels are used only for research evaluation.
- PathVLM-LiteBench is for research and educational use only.
- Outputs under `dataset/` and `outputs/` should not be committed.
