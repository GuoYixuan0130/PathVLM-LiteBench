# CLIP vs PLIP vs CONCH MHIST Sampled Observation

This note records a local post-v0.4.0 zero-shot comparison between CLIP, PLIP, and CONCH on a balanced MHIST test sample.

It is an engineering and research observation, not a clinical result. The goal is to check whether optional CONCH can run through the same PathVLM-LiteBench zero-shot interface and to document model behavior under identical prompts.

## Setup

Dataset:

- MHIST / Dartmouth Colorectal Polyps Histopathology Binary Classification Dataset
- Sampled test manifest: `dataset/MHIST/manifest_test_50_per_class.csv`
- Sample size: 100 patches
- True label distribution: `HP=50`, `SSA=50`

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
| `conch` | `MahmoodLab/CONCH` | Optional CONCH wrapper using gated Hugging Face access |

Class prompts:

| Class | Prompt |
|---|---|
| `HP` | `a histopathology image of hyperplastic polyp` |
| `SSA` | `a histopathology image of sessile serrated adenoma` |

## Commands

The same command shape was used for all models:

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest_test_50_per_class.csv \
  --image_root dataset/MHIST/images \
  --model <model_key> \
  --device cuda \
  --split test \
  --class_names HP SSA \
  --class_prompts \
    "a histopathology image of hyperplastic polyp" \
    "a histopathology image of sessile serrated adenoma" \
  --top_k 2 \
  --save_report \
  --report_dir outputs/zero_shot_<model_key>_mhist_sampled_v040
```

The generated output directories and comparison summary are local artifacts and should not be committed.

## Results

| Run | Accuracy | Balanced accuracy | Macro-F1 | HP recall | SSA recall | Predicted distribution |
|---|---:|---:|---:|---:|---:|---|
| CLIP | 0.5000 | 0.5000 | 0.4048 | 0.9000 | 0.1000 | `HP=90, SSA=10` |
| PLIP | 0.5500 | 0.5500 | 0.4357 | 0.1000 | 1.0000 | `HP=5, SSA=95` |
| CONCH | 0.5100 | 0.5100 | 0.4323 | 0.1400 | 0.8800 | `HP=13, SSA=87` |

All three runs triggered the prediction-collapse warning:

- CLIP predicted more than 80% of samples as `HP`.
- PLIP predicted more than 80% of samples as `SSA`.
- CONCH predicted more than 80% of samples as `SSA`.

## Confusion Matrices

Rows are true labels and columns are predicted labels.

CLIP:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 45 | 5 |
| SSA | 45 | 5 |

PLIP:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 5 | 45 |
| SSA | 0 | 50 |

CONCH:

| True label | Pred HP | Pred SSA |
|---|---:|---:|
| HP | 7 | 43 |
| SSA | 6 | 44 |

## Interpretation

The main result is not that one model is clinically better. On this small balanced sample and prompt pair, all three models are close to chance-level balanced accuracy and show strong class bias.

The useful engineering signal is that CLIP, PLIP, and CONCH can now be compared through the same zero-shot pipeline. The useful research signal is that pathology-specific model selection alone does not remove prompt and class-bias failure modes in this MHIST setup.

PLIP produced the highest balanced accuracy in this run, but the improvement is small and accompanied by a near-total `SSA` prediction bias. CONCH also favored `SSA`, though less extremely than PLIP. CLIP favored `HP`, matching the direction observed in earlier full-test CLIP runs.

## Implications

- v0.4.0 CONCH integration is usable for local saved-report comparisons.
- Future model comparisons should include prompt variants, predicted-label distributions, confusion matrices, and balanced metrics rather than accuracy alone.
- Any public claim should describe this as a local zero-shot behavior check, not a benchmark result.
- The next useful experiment is a prompt-grid comparison across CLIP, PLIP, and CONCH on the same sampled manifest.
