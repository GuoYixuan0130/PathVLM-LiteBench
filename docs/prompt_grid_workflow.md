# Zero-Shot Prompt-Grid Workflow

This guide explains how to run a zero-shot prompt grid with `pathvlm-litebench run-zero-shot-grid`.

A prompt grid runs the same zero-shot classification task across multiple models and multiple prompt pairs. It is useful for checking whether model behavior is stable under prompt wording changes.

The workflow is intended for local research and engineering review, not clinical interpretation.

## When to Use It

Use a zero-shot prompt grid when you want to compare:

- one model across several prompt styles
- several models under the same prompt styles
- predicted-label distributions and class-bias warnings
- balanced accuracy, macro-F1, confusion matrices, and error rates from saved reports

Do not treat one prompt pair as a final model benchmark. Prompt-grid results should be interpreted as local behavior checks.

## Minimal Command

Preview a prompt grid without loading models:

```bash
pathvlm-litebench validate-config \
  configs/zero_shot_prompt_grid_mhist_sample.json
```

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json \
  --dry-run
```

Run the grid:

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json
```

To avoid overwriting a previous local run, override the output root at runtime:

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json \
  --output-root outputs/zero_shot_prompt_grid_mhist_sample_run2
```

When `--output-root` is provided without `--comparison-output`, the comparison Markdown is written under the overridden output root.

The example config uses CLIP, PLIP, and optional CONCH. CONCH requires the optional CONCH package, approved Hugging Face access, and local authentication.

## Config Format

Example:

```json
{
  "task": "zero_shot_grid",
  "models": ["clip", "plip", "conch"],
  "device": "cuda",
  "manifest": "dataset/MHIST/manifest_test_50_per_class.csv",
  "image_root": "dataset/MHIST/images",
  "split": "test",
  "class_names": ["HP", "SSA"],
  "prompt_pairs": [
    {
      "key": "default",
      "class_prompts": [
        "a histopathology image of hyperplastic polyp",
        "a histopathology image of sessile serrated adenoma"
      ]
    },
    {
      "key": "patch",
      "class_prompts": [
        "a pathology patch showing hyperplastic polyp",
        "a pathology patch showing sessile serrated adenoma"
      ]
    }
  ],
  "top_k": 2,
  "output_root": "outputs/zero_shot_prompt_grid_mhist_sample",
  "save_comparison": true,
  "comparison_output": "outputs/zero_shot_prompt_grid_mhist_sample/comparison.md",
  "write_logs": true
}
```

Fields:

| Field | Required | Meaning |
|---|---|---|
| `task` | yes | Must be `zero_shot_grid`. |
| `models` | yes | Model keys or model names to run, such as `clip`, `plip`, or `conch`. |
| `device` | no | `auto`, `cpu`, or `cuda`. Defaults to `auto`. |
| `manifest` | yes, unless `image_dir` is used | Standard patch manifest CSV. |
| `image_root` | no | Root folder used to resolve relative paths in the manifest. |
| `image_dir` | yes, unless `manifest` is used | Folder of patch images for unlabeled smoke runs. |
| `split` | no | Optional manifest split filter, such as `test`. |
| `max_images` | no | Optional image limit for quick smoke runs. |
| `class_names` | yes | Class labels used by zero-shot classification. |
| `prompt_pairs` | yes | List of named prompt sets. Each must contain one prompt per class. |
| `top_k` | no | Number of class predictions to save per image. Defaults to `2`. |
| `output_root` | no | Root folder for generated report directories. |
| `save_comparison` | no | Whether to write a Markdown comparison after all runs. Defaults to `true`. |
| `comparison_output` | no | Custom Markdown comparison path. |
| `write_logs` | no | Whether each run writes `run.log`. Defaults to `true`. |

Runtime overrides:

| CLI option | Effect |
|---|---|
| `--output-root` | Overrides the config `output_root` for generated run directories. |
| `--comparison-output` | Overrides the config `comparison_output` Markdown path. |

`prompt_pairs[*].class_prompts` must have the same length and order as `class_names`.

## Output Layout

For this config:

```json
"models": ["clip", "plip"],
"prompt_pairs": [{"key": "default"}, {"key": "patch"}],
"output_root": "outputs/zero_shot_prompt_grid"
```

The command writes:

```text
outputs/zero_shot_prompt_grid/
|-- clip/
|   |-- default/
|   |   |-- predictions.csv
|   |   |-- errors.csv
|   |   |-- metrics.json
|   |   `-- run.log
|   `-- patch/
|       |-- predictions.csv
|       |-- errors.csv
|       |-- metrics.json
|       `-- run.log
|-- plip/
|   |-- default/
|   `-- patch/
`-- comparison.md
```

Each run directory is the same zero-shot report format produced by `examples/02_zero_shot_classification_demo.py --save_report`.

Generated outputs under `outputs/` are ignored by Git and should not be committed if they contain dataset-specific paths or metrics.

## Comparison Summary

When `save_comparison` is enabled, the command writes a Markdown file using the existing zero-shot report comparison utility.

The comparison table includes:

- model
- split
- number of images
- accuracy
- balanced accuracy
- macro-F1
- error count
- error rate
- predicted-label distribution
- prediction-collapse warning, if present

For fine-grained pathology tasks, balanced accuracy, macro-F1, per-class recall, confusion matrices, and predicted-label distribution are usually more informative than raw accuracy alone.

## Interpreting Warnings

The zero-shot report may include a warning when more than 80% of samples are predicted as one class.

This does not always mean the code is wrong. It usually means one of these is true:

- the prompt wording is biasing predictions
- the model is weak for the class distinction
- the dataset sample is difficult or unbalanced
- the class labels are too fine-grained for the prompt pair

Treat the warning as a reason to inspect confusion matrices, per-class recall, and prompt variants.

## Practical Workflow

Recommended workflow:

1. Convert dataset annotations into the standard manifest format.
2. Create a small balanced sampled manifest.
3. Run `run-zero-shot-grid --dry-run` to confirm planned runs.
4. Run the prompt grid on `cpu`, `cuda`, or `auto`.
5. Read `comparison.md`.
6. Inspect individual `metrics.json` files for confusion matrices and per-class metrics.
7. Summarize findings conservatively as local model-behavior observations.

Example dry run:

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json \
  --dry-run
```

Example full run:

```bash
pathvlm-litebench run-zero-shot-grid \
  --config configs/zero_shot_prompt_grid_mhist_sample.json \
  --output-root outputs/zero_shot_prompt_grid_mhist_sample_run2
```

## Failure Modes

Common failures:

- missing dataset files under `dataset/`
- wrong manifest path or `image_root`
- `class_prompts` length does not match `class_names`
- CUDA requested but unavailable
- optional CONCH package not installed
- gated CONCH weights inaccessible because Hugging Face authentication is missing

If a run fails and `write_logs` is enabled, inspect the run-specific `run.log`.

## Claim Boundaries

Good wording:

- "This prompt grid showed class-bias changes under different prompt styles."
- "CONCH ran through the same zero-shot workflow as CLIP and PLIP."
- "Balanced accuracy remained close to chance in this local sampled run."

Avoid:

- "This model diagnoses HP or SSA."
- "This prompt grid proves one pathology VLM is clinically reliable."
- "These generated reports are benchmark certification."
