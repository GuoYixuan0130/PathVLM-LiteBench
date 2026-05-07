# CLIP vs PLIP vs CONCH MHIST Prompt-Grid Observation

This note records a local prompt-grid zero-shot comparison between CLIP, PLIP, and CONCH on the balanced MHIST sampled test manifest.

It is an engineering and research observation, not a clinical result. The goal is to check whether class bias remains stable across prompt wording when all models use the same PathVLM-LiteBench zero-shot pipeline.

## Setup

Dataset:

- MHIST / Dartmouth Colorectal Polyps Histopathology Binary Classification Dataset
- Sampled test manifest: `dataset/MHIST/manifest_test_50_per_class.csv`
- Sample size: 100 patches
- True label distribution: `HP=50`, `SSA=50`

Models:

| Model key | Model | Notes |
|---|---|---|
| `clip` | `openai/clip-vit-base-patch32` | General CLIP baseline |
| `plip` | `vinid/plip` | Pathology-specific CLIP-compatible wrapper |
| `conch` | `MahmoodLab/CONCH` | Optional CONCH wrapper using gated Hugging Face access |

Prompt pairs:

| Prompt key | HP prompt | SSA prompt |
|---|---|---|
| `default` | `a histopathology image of hyperplastic polyp` | `a histopathology image of sessile serrated adenoma` |
| `diagnosis` | `hyperplastic polyp tissue` | `sessile serrated adenoma tissue` |
| `patch` | `a pathology patch showing hyperplastic polyp` | `a pathology patch showing sessile serrated adenoma` |

All runs used `--device cuda`, frozen model inference, no fine-tuning, and saved reports under ignored local `outputs/` directories.

## Results

| Model | Prompt key | Accuracy | Balanced accuracy | Macro-F1 | HP recall | SSA recall | Predicted distribution | Collapse warning |
|---|---|---:|---:|---:|---:|---:|---|---|
| CLIP | `default` | 0.5000 | 0.5000 | 0.4048 | 0.9000 | 0.1000 | `HP=90, SSA=10` | yes |
| CLIP | `diagnosis` | 0.5000 | 0.5000 | 0.3658 | 0.9600 | 0.0400 | `HP=96, SSA=4` | yes |
| CLIP | `patch` | 0.4800 | 0.4800 | 0.4583 | 0.6800 | 0.2800 | `HP=70, SSA=30` | no |
| PLIP | `default` | 0.5500 | 0.5500 | 0.4357 | 0.1000 | 1.0000 | `HP=5, SSA=95` | yes |
| PLIP | `diagnosis` | 0.5800 | 0.5800 | 0.5175 | 0.9400 | 0.2200 | `HP=86, SSA=14` | yes |
| PLIP | `patch` | 0.5400 | 0.5400 | 0.4296 | 0.1000 | 0.9800 | `HP=6, SSA=94` | yes |
| CONCH | `default` | 0.5100 | 0.5100 | 0.4323 | 0.1400 | 0.8800 | `HP=13, SSA=87` | yes |
| CONCH | `diagnosis` | 0.5100 | 0.5100 | 0.3552 | 0.0200 | 1.0000 | `HP=1, SSA=99` | yes |
| CONCH | `patch` | 0.4600 | 0.4600 | 0.4565 | 0.5400 | 0.3800 | `HP=58, SSA=42` | no |

## Observations

Prompt wording changed the class-bias pattern.

- CLIP stayed HP-biased across all prompt pairs, but the `patch` prompts reduced the predicted HP share from 90-96% to 70%.
- PLIP changed direction: `default` and `patch` prompts were strongly SSA-biased, while the shorter `diagnosis` prompts became strongly HP-biased.
- CONCH was SSA-biased for `default` and `diagnosis`, but the `patch` prompts produced a more mixed prediction distribution.

Balanced accuracy stayed close to chance for all runs. The best run in this grid was PLIP with `diagnosis` prompts at 0.5800 balanced accuracy, but it still triggered a prediction-collapse warning with `HP=86, SSA=14`.

## Interpretation

The result suggests that prompt wording and model identity interact strongly in this MHIST zero-shot setup. A single prompt pair is not enough to characterize model behavior.

This is useful for PathVLM-LiteBench because it validates the need for saved report comparison, prompt-sensitivity checks, predicted-label distributions, and conservative interpretation. It also shows that adding a pathology-specific model wrapper does not by itself remove class-bias failure modes.
