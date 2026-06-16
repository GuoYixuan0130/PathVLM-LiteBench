# v0.11.0 Release Notes

## Summary

PathVLM-LiteBench v0.11.0 is a usability and robustness release. It focuses on making the toolkit easier to install, run, and scale on laptop-class hardware, without changing the core patch-level evaluation workflows.

The headline change is bounded-batch image encoding: image embedding now processes patches in fixed-size batches with an optional progress bar, instead of a single forward pass over all images. This keeps memory usage predictable on consumer GPUs and CPUs and aligns the implementation with the project's low-compute positioning. The release also adds a `demo` CLI subcommand for running bundled examples by name, a `--batch_size` flag on the model demos, friendlier configuration error handling, clearer installation documentation, and a top-of-README quickstart.

This release does not add whole-slide image reading, slide tiling, tissue detection, bundled pathology data, bundled model weights, required CONCH access, Hugging Face token handling, clinical diagnosis, or clinical decision support. It remains a research and educational toolkit for patch-level evaluation and visualization.

## What Is Included

Bounded-batch image encoding:

- encodes patch images in fixed-size batches instead of one forward pass over all images
- adds a shared batch iterator used by the CLIP and CONCH wrappers
- shows an optional progress bar when encoding more than one batch
- validates that the batch size is positive
- keeps embeddings L2-normalized and identical to single-batch results
- removes a dead text-encoder fallback path
- removes the unused `torchvision` dependency

`demo` CLI subcommand:

- adds `pathvlm-litebench demo <name> [args...]` to run bundled examples by name
- supported names: `retrieval`, `zero-shot`, `prompt-sensitivity`, `retrieval-metrics`, `heatmap`
- forwards extra arguments to the underlying example script
- lists runnable demos when no name is given
- reports a clear error for unknown demo names
- launches demos in a subprocess so importing the CLI never loads `torch` or `transformers`
- cross-links the runner from the existing `demos` listing

Demo usability:

- adds `--batch_size` to the retrieval, zero-shot, and prompt-sensitivity demos so users can avoid out-of-memory errors without editing code
- wraps configuration merging in friendly error handling, printing a short message instead of a raw traceback on missing or malformed config files

Documentation:

- restructures the README installation section around `pip install -e .`, which installs dependencies and the `pathvlm-litebench` command in one step
- documents installing a CUDA-matched PyTorch wheel for laptop GPU acceleration
- adds a top-of-README TL;DR quickstart for install-and-run in a few commands
- corrects the quick start notebook so PLIP is described as implemented and CONCH as an optional, gated wrapper
- trims and refreshes user-facing documentation
- published v0.11.0 release notes

Test and CI coverage:

- adds CLIP wrapper tests for batched-versus-single-batch equality, L2 normalization, and batch-size validation
- adds CLI tests for the `demo` subcommand listing, unknown names, argument forwarding, and exit-code propagation
- updates configuration-merge test fixtures for the new `--batch_size` argument
- uses fake model wrappers and avoids model downloads in CI

## What Remains Unchanged

The toolkit remains focused on:

- frozen CLIP-style model inference for retrieval, classification, and patch scoring workflows
- patch-level images
- coordinate-aware patch manifests
- low-compute experiments
- CPU-compatible smoke tests
- CUDA acceleration when available
- local CSV, JSON, Markdown, HTML, and PNG artifacts
- offline CI tests that avoid model downloads

Existing workflows remain available:

- CLIP, PLIP, and optional CONCH model wrappers
- patch-text retrieval
- zero-shot classification
- prompt sensitivity analysis
- zero-shot prompt-grid runs
- manifest conversion and balanced sampling
- summary and comparison Markdown reports
- embedding cache and visualization utilities
- artifact-only `render-coordinate-heatmap`
- model-backed `score-coordinate-heatmap` and `score-coordinate-heatmap-prompt-set`
- artifact-only `compare-coordinate-heatmap-scores`
- synthetic patch-coordinate heatmap demo

CONCH remains optional and gated. v0.11.0 does not make CONCH a required dependency.

## What Is Intentionally Not Included

v0.11.0 does not include:

- whole-slide image file reading
- slide tiling or tissue detection
- stain normalization or slide pyramid rendering
- tumor boundary identification
- clinical diagnosis or clinical decision support
- model training or fine-tuning
- downloaded or bundled public datasets
- bundled model weights
- required model downloads in CI
- required CONCH access
- Hugging Face token handling
- generated heatmaps, score CSV files, metadata files, comparison files, or reports committed to the repository
- real pathology images or whole-slide image files committed to the repository

Generated outputs remain local. Do not commit datasets, model weights, Hugging Face caches, generated patches, heatmaps, score CSV files, metadata JSON files, comparison artifacts, reports, prediction files, or run logs.

## Verification

Recommended verification before tagging on Windows:

```powershell
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m pathvlm_litebench.cli version
.venv\Scripts\python.exe -m pathvlm_litebench.cli models
.venv\Scripts\python.exe -m pathvlm_litebench.cli demos
.venv\Scripts\python.exe -m pathvlm_litebench.cli demo
```

The inspection commands above do not download models. Model-based demos load weights only when a demo is actually run.

## Example Commands

List the runnable demos:

```bash
pathvlm-litebench demo
```

Run the patch-text retrieval demo by name:

```bash
pathvlm-litebench demo retrieval --model clip --device auto
```

Lower the batch size to reduce memory usage:

```bash
python examples/02_zero_shot_classification_demo.py --model clip --batch_size 8
```

## Release Highlights

PathVLM-LiteBench v0.11.0 makes the toolkit easier to adopt on laptop-class hardware:

- image encoding runs in bounded batches with predictable memory usage
- a single `pip install -e .` installs dependencies and the CLI
- `pathvlm-litebench demo <name>` runs bundled examples without remembering script paths
- `--batch_size` lets users dial memory usage without editing code
- configuration errors print short, actionable messages
- a top-of-README quickstart gets new users from clone to first demo quickly
- CI remains lightweight and avoids model downloads
