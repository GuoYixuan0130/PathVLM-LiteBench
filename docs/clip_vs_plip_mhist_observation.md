# CLIP vs PLIP MHIST Preliminary Observation

This note records a local, low-compute zero-shot comparison between CLIP and PLIP on MHIST.

It is an engineering and research observation, not a clinical result. The goal is to document model behavior, prompt sensitivity, and prediction-collapse signals under the same PathVLM-LiteBench evaluation pipeline.

## Setup

Dataset:

- MHIST / Dartmouth Colorectal Polyps Histopathology Binary Classification Dataset
- Standard manifest format: `image_path,label,split,case_id`
- Full test manifest: `dataset/MHIST/manifest.csv`, split `test`
- Full test samples: 977 total, 617 `HP`, 360 `SSA`
- Sampled sanity-check manifest: `dataset/MHIST/manifest_test_50_per_class.csv`
- Sampled sanity-check samples: 100 total, 50 `HP`, 50 `SSA`

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

## Full Test Results

| Run | Accuracy | Balanced accuracy | Macro-F1 | HP recall | SSA recall | Predicted distribution |
|---|---:|---:|---:|---:|---:|---|
| CLIP, default prompts | 0.6018 | 0.4921 | 0.4322 | 0.9092 | 0.0750 | `HP=894, SSA=83` |
| PLIP, default prompts | 0.4893 | 0.5933 | 0.4582 | 0.1977 | 0.9889 | `SSA=851, HP=126` |

The full test split is class-imbalanced, so accuracy is not sufficient by itself. CLIP has higher accuracy because it predicts the majority class `HP` for most samples. PLIP has lower accuracy but higher balanced accuracy, macro precision, macro recall, macro-F1, and much higher `SSA` recall.

Both models triggered a prediction-collapse warning:

- CLIP predicted more than 80% of samples as `HP`.
- PLIP predicted more than 80% of samples as `SSA`.

## Full Test Confusion Matrices

Rows are true labels and columns are predicted labels.

CLIP, default prompts:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 561 | 56 |
| SSA | 333 | 27 |

PLIP, default prompts:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 122 | 495 |
| SSA | 4 | 356 |

## Sampled Sanity Checks

The sampled sanity-check run used a balanced 100-image manifest with 50 `HP` and 50 `SSA` samples.

| Run | Accuracy | Balanced accuracy | Macro-F1 | HP recall | SSA recall | Predicted distribution |
|---|---:|---:|---:|---:|---:|---|
| CLIP, default prompts | 0.5000 | 0.5000 | 0.4048 | 0.9000 | 0.1000 | `HP=90, SSA=10` |
| PLIP, default prompts | 0.5500 | 0.5500 | 0.4357 | 0.1000 | 1.0000 | `SSA=95, HP=5` |
| PLIP, class order swapped | 0.5500 | 0.5500 | 0.4357 | 0.1000 | 1.0000 | `SSA=95, HP=5` |
| PLIP, synonym prompts | 0.5200 | 0.5200 | 0.4792 | 0.2400 | 0.8000 | `SSA=78, HP=22` |

Synonym prompts for the sampled sanity check:

| Class | Prompt |
|---|---|
| `HP` | `an H&E histopathology patch showing a hyperplastic colorectal polyp` |
| `SSA` | `an H&E histopathology patch showing a sessile serrated lesion` |

Sampled confusion matrices:

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

The full and sampled runs suggest that both CLIP and PLIP can show strong prediction bias on MHIST under simple zero-shot prompts:

- CLIP default prompts strongly favored `HP`.
- PLIP default prompts strongly favored `SSA`.
- PLIP improved balanced accuracy and `SSA` recall in the full test run, but this came with a strong `SSA` prediction bias and much lower `HP` recall.
- PLIP remained sensitive to prompt wording.
- Accuracy alone hid important failure modes.

This does not mean that PLIP is clinically reliable, and it does not prove that PLIP solves MHIST. It shows why PathVLM-LiteBench reports balanced accuracy, macro-F1, per-class metrics, confusion matrices, predicted-label distributions, error files, and prediction-collapse warnings.

## Project Implication

This observation supports the current project direction:

- Keep CLIP as the general baseline.
- Use PLIP as a pathology-specific comparison model through the same wrapper interface.
- Evaluate model behavior with the same manifests, prompts, metrics, and report format.
- Treat prompt sensitivity as part of the benchmark rather than an afterthought.
- Keep the workflow laptop-friendly by using sampled manifests for debugging and full test runs for more stable observation.

## Limitations

- This is a local baseline observation, not a finalized benchmark.
- Prompt wording can change the result.
- Model versions, package versions, preprocessing, and sampling seed can affect metrics.
- The sampled run is useful for debugging, while the full test run is more representative of this local MHIST split.
- MHIST labels are used only for research evaluation.
- PathVLM-LiteBench is for research and educational use only.
- Outputs under `dataset/` and `outputs/` should not be committed.
