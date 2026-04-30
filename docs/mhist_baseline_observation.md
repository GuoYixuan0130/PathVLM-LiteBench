# MHIST CLIP Baseline Observation

This note records an example local zero-shot baseline observation on MHIST using the current CLIP pipeline in PathVLM-LiteBench.

It is intended as an engineering and research observation, not as a clinical result.

## Setup

Dataset:

- MHIST / Dartmouth Colorectal Polyps Histopathology Binary Classification Dataset
- Patch-level images with a converted standard manifest
- Test split only
- 977 test samples in this local run

Task:

- Zero-shot patch classification
- Classes: `HP` and `SSA`
- Model key: `clip`
- Resolved model: `openai/clip-vit-base-patch32`
- Frozen model inference only
- No fine-tuning

Example command:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest.csv \
  --image_root dataset/MHIST/images \
  --model clip \
  --device cuda \
  --split test \
  --class_names HP SSA \
  --class_prompts \
    "a histopathology image of hyperplastic polyp" \
    "a histopathology image of sessile serrated adenoma" \
  --top_k 2 \
  --save_report \
  --report_dir outputs/zero_shot_demo
```

`dataset/` and `outputs/` are local folders and should not be committed.

## Observed Class Distributions

True label distribution:

| Label | Count |
|---|---:|
| HP | 617 |
| SSA | 360 |

Predicted label distribution:

| Label | Count |
|---|---:|
| HP | 894 |
| SSA | 83 |

The model predicted `HP` for most test samples. This is a useful failure signal: the output is strongly class-skewed even though both classes are present in the test split.

## Observed Confusion Matrix

Rows are true labels and columns are predicted labels.

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 561 | 56 |
| SSA | 333 | 27 |

The main failure mode is low `SSA` recall. In this run, most `SSA` samples were predicted as `HP`.

## Observed Metrics

Approximate metrics from this local run:

| Metric | Value |
|---|---:|
| Accuracy | 0.6018 |
| Balanced accuracy | 0.4921 |
| Macro F1 | 0.4322 |
| HP recall | 0.9092 |
| SSA recall | 0.0750 |

Accuracy alone is misleading here because the dataset is class-imbalanced and the prediction distribution is even more skewed. Balanced accuracy, macro-F1, per-class recall, confusion matrix, and error analysis are more informative for this kind of fine-grained pathology task.

## Interpretation

This observation does not indicate a CSV or report-saving bug. The pipeline produced valid predictions, metrics, distributions, and error-analysis outputs.

The result instead suggests that a general CLIP baseline may be weak for fine-grained pathology semantics such as differentiating `HP` from `SSA` in MHIST. This is consistent with the project positioning: start with a general CLIP baseline to validate the evaluation workflow, then compare against pathology-specific VLMs through the same interface.

Useful outputs from the current zero-shot report include:

- `predictions.csv` for per-image predictions
- `errors.csv` for misclassified samples
- `metrics.json` for aggregate metrics, class distributions, confusion matrix, and prediction-collapse warning

## Project Implication

This failure mode supports the next engineering priorities:

- Keep CLIP as the implemented baseline rather than presenting it as a pathology diagnostic model.
- Emphasize balanced accuracy, macro-F1, per-class metrics, and confusion matrices.
- Use saved predictions and error reports to inspect class bias.
- Keep the workflow laptop-friendly by using sampled manifests for quick checks.
- Add PLIP / CONCH later through the same wrapper and reporting interface for fair comparison.

## Limitations

- This is a local baseline observation, not a finalized benchmark result.
- Prompt wording, preprocessing, model version, and dataset split can affect results.
- MHIST labels are used for research evaluation only.
- PathVLM-LiteBench is for research and educational use only.
- The output should not be interpreted as clinical diagnosis or clinical decision support.
