# v0.14.1 Release Notes

## Summary

PathVLM-LiteBench v0.14.1 is a documentation patch release. It corrects the `linear-probe` usage example introduced in v0.14.0 and changes no code behavior. All v0.14.0 features remain as-is.

## What Is Fixed

`linear-probe` documentation example:

- the README and v0.14.0 release-notes examples pointed `linear-probe` at the bundled `CRC_VAL_HE_100_sample_manifest.csv`, which has only `image_path` and `label` columns
- `linear-probe` defaults to `--train-split train` and `--test-split test`, so that manifest produced the error `No manifest records matched train split 'train'`
- the examples now point at a manifest that carries a `split` column with `train`/`test` values, and the docs state the split requirement explicitly

## What Remains Unchanged

- all v0.14.0 functionality: bootstrap confidence intervals on accuracy, the `linear-probe` command, runtime-environment metadata, the `cli/` package structure, and the consistency fixes
- the CLI surface, command behavior, and import surface are unchanged
- the toolkit remains a low-compute, laptop-friendly research and educational benchmark for patch-level evaluation and visualization

## Verification

```powershell
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m pathvlm_litebench.cli version
.venv\Scripts\python.exe -m pathvlm_litebench.cli linear-probe --help
```

## Example Commands

Train and evaluate a linear probe on frozen embeddings (the manifest must carry a `split` column with `train`/`test` values):

```bash
pathvlm-litebench linear-probe --manifest dataset/CRC_VAL_HE_100_split_manifest.csv --model plip --train-split train --test-split test --output-dir outputs/linear_probe
```
