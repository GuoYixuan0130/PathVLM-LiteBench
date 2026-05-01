# CLIP vs PLIP MHIST Protocol

This protocol describes a local, low-compute comparison between the general CLIP baseline and the PLIP pathology-specific model on MHIST.

The goal is to compare model behavior under the same zero-shot evaluation pipeline, not to make clinical claims.

## Scope

This protocol uses:

- the same dataset
- the same sampled manifest
- the same class names
- the same class prompts
- the same zero-shot classification demo
- separate output folders for each model

Outputs are local artifacts and should not be committed.

## Prerequisites

Expected local dataset layout:

```text
dataset/
`-- MHIST/
    |-- annotations.csv
    |-- manifest.csv
    |-- manifest_test_50_per_class.csv
    `-- images/
        |-- MHIST_aaa.png
        |-- MHIST_aab.png
        `-- ...
```

If the sampled manifest does not exist, create it first:

```bash
pathvlm-litebench sample-manifest \
  --input dataset/MHIST/manifest.csv \
  --output dataset/MHIST/manifest_test_50_per_class.csv \
  --split test \
  --samples_per_label 50 \
  --seed 42
```

Use `--device cuda` if the local GPU is available. Use `--device auto` for a portable command.

## Prompts

Use the same prompts for both models:

```text
HP:  a histopathology image of hyperplastic polyp
SSA: a histopathology image of sessile serrated adenoma
```

Do not tune prompts separately for one model during the first comparison. The first comparison should isolate the model difference.

## Run CLIP

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest_test_50_per_class.csv \
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
  --report_dir outputs/zero_shot_clip_mhist_sampled
```

Expected outputs:

```text
outputs/zero_shot_clip_mhist_sampled/
|-- predictions.csv
|-- errors.csv
`-- metrics.json
```

## Run PLIP

```bash
python examples/02_zero_shot_classification_demo.py \
  --manifest dataset/MHIST/manifest_test_50_per_class.csv \
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
  --report_dir outputs/zero_shot_plip_mhist_sampled
```

Expected outputs:

```text
outputs/zero_shot_plip_mhist_sampled/
|-- predictions.csv
|-- errors.csv
`-- metrics.json
```

## Metrics to Compare

Compare these fields from each `metrics.json`:

- accuracy
- balanced accuracy
- macro precision
- macro recall
- macro-F1
- HP precision / recall / F1
- SSA precision / recall / F1
- confusion matrix
- true label distribution
- predicted label distribution
- error rate
- prediction-collapse warning, if present

For this task, balanced accuracy, macro-F1, and `SSA` recall are more informative than accuracy alone.

## Quick Metrics Extraction

Local helper command for printing a compact comparison:

```powershell
@'
import json
from pathlib import Path

paths = {
    "clip": Path("outputs/zero_shot_clip_mhist_sampled/metrics.json"),
    "plip": Path("outputs/zero_shot_plip_mhist_sampled/metrics.json"),
}

for name, path in paths.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    report = data["metrics"]["classification_report"]
    error_summary = data["metrics"]["error_summary"]
    print(f"\n{name.upper()}")
    print(f"accuracy: {report['accuracy']:.4f}")
    print(f"balanced_accuracy: {report['balanced_accuracy']:.4f}")
    print(f"macro_f1: {report['macro_f1']:.4f}")
    for label, metrics in report["per_class"].items():
        print(f"{label}_recall: {metrics['recall']:.4f}")
    print(f"predicted_distribution: {error_summary['predicted_label_distribution']}")
    warning = error_summary.get("warning")
    if warning:
        print(f"warning: {warning}")
'@ | python -
```

## Result Table

Record results in this format:

```text
Dataset:
Manifest:
Device:
Date:

| Model | Accuracy | Balanced accuracy | Macro-F1 | HP recall | SSA recall |
|---|---:|---:|---:|---:|---:|
| CLIP |  |  |  |  |  |
| PLIP |  |  |  |  |  |

CLIP predicted distribution:
PLIP predicted distribution:

CLIP confusion matrix:
PLIP confusion matrix:

Main observation:
Possible failure mode:
Next prompt/model check:
```

## Interpretation

Interpret results conservatively:

- Good: "PLIP improved balanced accuracy on this sampled MHIST run."
- Good: "PLIP reduced the HP prediction bias in this local zero-shot setup."
- Avoid: "PLIP diagnoses SSA."
- Avoid: "PLIP is clinically reliable."
- Avoid: "This proves pathology-specific VLMs solve MHIST."

Describe the result as a local zero-shot baseline comparison.

## Recording Results

After running both models, record the compact metrics output in the project notes or observation document.

The main observation document for this comparison is:

```text
docs/clip_vs_plip_mhist_observation.md
```

That document records:

- setup
- metrics table
- confusion matrices
- prediction distributions
- representative error-analysis notes
- limitations
- implications for future pathology-specific wrapper work
