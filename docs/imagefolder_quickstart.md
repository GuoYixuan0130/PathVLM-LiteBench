# ImageFolder Quickstart: From a Public Dataset to a Benchmark

This is the shortest path from a public ImageFolder-style pathology dataset to a runnable PathVLM-LiteBench benchmark. It assumes no annotation CSV and no manual manifest editing.

The example uses NCT-CRC-HE-100K because it is openly licensed (CC BY 4.0) and ships as a class-per-folder tree, which maps directly onto the `build-imagefolder-manifest` command. The same four steps apply to any dataset that is laid out as `<class>/<image>` or `<split>/<class>/<image>`.

This workflow is for research and engineering review only. It is not a clinical benchmark or a diagnostic workflow.

## Overview

```text
download a public ImageFolder dataset locally
-> build a standard manifest with build-imagefolder-manifest
-> sample a small balanced manifest
-> run a CLIP vs PLIP zero-shot comparison
```

Keep data and outputs local and out of Git:

```text
dataset/   # downloaded datasets, ignored by Git
outputs/   # generated reports, ignored by Git
```

## 1. Download a Public Dataset

Download NCT-CRC-HE-100K from its official Zenodo record (DOI [10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)) and extract it under a local, Git-ignored folder. Always follow the dataset's own license and citation terms; NCT-CRC-HE-100K is distributed under CC BY 4.0 and must be attributed.

After extraction you get a class-per-folder tree:

```text
dataset/NCT-CRC-HE-100K/
|-- ADI/    # adipose tissue
|-- BACK/   # background
|-- DEB/    # debris
|-- LYM/    # lymphocytes
|-- MUC/    # mucus
|-- MUS/    # smooth muscle
|-- NORM/   # normal colon mucosa
|-- STR/    # cancer-associated stroma
`-- TUM/    # colorectal adenocarcinoma epithelium
```

The repository does not include this or any other pathology dataset.

## 2. Build a Standard Manifest

Turn the folder tree directly into a manifest:

```bash
pathvlm-litebench build-imagefolder-manifest \
  --image-dir dataset/NCT-CRC-HE-100K \
  --output dataset/NCT-CRC-HE-100K/manifest.csv \
  --relative
```

Each leaf folder name (`ADI`, `BACK`, ...) becomes a `label`, and each image becomes one manifest row with `image_path,label,split` columns. `--relative` writes image paths relative to the manifest so the dataset folder stays portable. The command prints the record count, class count, and per-class distribution so you can confirm the scan before running anything heavy.

If your dataset has a leading split level (`train/<class>/<image>`, `test/<class>/<image>`), add `--has-split`.

## 3. Sample a Small Balanced Manifest

NCT-CRC-HE-100K has 100k patches, which is far more than a laptop smoke test needs. Sample a small balanced subset first:

```bash
pathvlm-litebench sample-manifest \
  --input dataset/NCT-CRC-HE-100K/manifest.csv \
  --output dataset/NCT-CRC-HE-100K/manifest_sample.csv \
  --samples_per_label 20 \
  --seed 42
```

For a first pass, 20-100 patches per class is enough to validate the workflow and keeps CPU runs fast.

## 4. Run a Zero-Shot Model Comparison

Compare a general CLIP baseline against the pathology-pretrained PLIP on the sampled manifest:

```bash
pathvlm-litebench compare-models \
  --manifest dataset/NCT-CRC-HE-100K/manifest_sample.csv \
  --models clip plip \
  --class-prompts \
    "an H&E image of adipose tissue." \
    "an H&E image of background." \
    "an H&E image of debris." \
    "an H&E image of lymphocytes." \
    "an H&E image of mucus." \
    "an H&E image of smooth muscle." \
    "an H&E image of normal colon mucosa." \
    "an H&E image of cancer-associated stroma." \
    "an H&E image of colorectal adenocarcinoma epithelium." \
  --output-dir outputs/nct_crc_he_compare
```

The class prompts are listed in the same order as the sorted unique manifest labels (`ADI`, `BACK`, `DEB`, `LYM`, `MUC`, `MUS`, `NORM`, `STR`, `TUM`), so prompt `i` describes class `i`. If you omit `--class-prompts`, the command falls back to the `an H&E image of {}.` template applied to the raw folder names.

Add `--dry-run` first to print the resolved class names, prompts, and output paths without loading any model.

Outputs are written under `outputs/nct_crc_he_compare/`:

```text
outputs/nct_crc_he_compare/
|-- model_comparison.csv            # per-model accuracy
|-- model_comparison_per_class.csv  # per-model, per-class accuracy
|-- model_comparison.png            # bar chart with a random-chance baseline
`-- metadata.json                   # run configuration and full results
```

## Interpret Results Conservatively

Domain pretraining usually dominates this comparison: a pathology-pretrained model such as PLIP typically scores well above a general CLIP baseline, which can sit near the random-chance line (`1 / number_of_classes`). Read the per-class CSV and the chart together, and prefer wording like "PLIP outperformed the general CLIP baseline on this sampled run" over any clinical or diagnostic claim.

Do not commit downloaded patches, the generated manifests, or the `outputs/` reports.

## Related Docs

- [data_preparation.md](data_preparation.md)
- [public_patch_benchmark_workflow.md](public_patch_benchmark_workflow.md)
- [small_dataset_quickstart.md](small_dataset_quickstart.md)
