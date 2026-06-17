from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from . import version
from .config import load_benchmark_config
from .config.heatmap_config import (
    PatchCoordinateHeatmapConfig,
    PatchCoordinateHeatmapPromptSetConfig,
    PatchCoordinateHeatmapScoringConfig,
)
from .data.imagefolder import build_imagefolder_manifest
from .data.manifest_converter import convert_manifest, convert_mhist_manifest
from .data.manifest_sampler import sample_manifest, summarize_manifest
from .models.registry import list_available_models
from .visualization.report_summary import (
    save_experiment_comparison_summary,
    save_prompt_sensitivity_experiment_summary,
    save_retrieval_experiment_summary,
    save_zero_shot_experiment_summary,
)


DEMO_SCRIPTS: dict[str, str] = {
    "retrieval": "01_patch_text_retrieval_demo.py",
    "zero-shot": "02_zero_shot_classification_demo.py",
    "prompt-sensitivity": "03_prompt_sensitivity_demo.py",
    "retrieval-metrics": "04_retrieval_metrics_demo.py",
    "heatmap": "05_patch_coordinate_heatmap_demo.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathvlm-litebench",
        description="PathVLM-LiteBench lightweight command-line interface.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="Show toolkit version.")
    subparsers.add_parser("models", help="List registered models.")
    subparsers.add_parser("demos", help="Show available demo commands.")
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run a bundled example demo by name (forwards extra args to the script).",
    )
    demo_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Demo to run. One of: " + ", ".join(DEMO_SCRIPTS) + ". Omit to list demos.",
    )
    demo_parser.add_argument(
        "demo_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the demo script, e.g. --model clip --device auto.",
    )
    convert_manifest_parser = subparsers.add_parser(
        "convert-manifest",
        help="Convert dataset-specific annotations CSV to standard patch manifest CSV.",
    )
    imagefolder_manifest_parser = subparsers.add_parser(
        "build-imagefolder-manifest",
        help="Build a standard patch manifest from an ImageFolder-style directory tree.",
    )
    sample_manifest_parser = subparsers.add_parser(
        "sample-manifest",
        help="Sample a smaller balanced subset from a standard manifest CSV.",
    )
    summarize_report_parser = subparsers.add_parser(
        "summarize-report",
        help="Generate a Markdown summary from saved experiment report artifacts.",
    )
    compare_reports_parser = subparsers.add_parser(
        "compare-reports",
        help="Generate a Markdown comparison from multiple saved report directories.",
    )
    validate_config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a benchmark JSON config without running model inference.",
    )
    zero_shot_grid_parser = subparsers.add_parser(
        "run-zero-shot-grid",
        help="Run a zero-shot prompt grid from a JSON config file.",
    )
    render_heatmap_parser = subparsers.add_parser(
        "render-coordinate-heatmap",
        help="Render a patch-coordinate heatmap from a manifest and existing score CSV.",
    )
    score_heatmap_parser = subparsers.add_parser(
        "score-coordinate-heatmap",
        help="Score coordinate patches against a text prompt and render a heatmap.",
    )
    score_heatmap_prompt_set_parser = subparsers.add_parser(
        "score-coordinate-heatmap-prompt-set",
        help="Expand a prompt set for prompt-scored patch-coordinate heatmaps.",
    )
    compare_heatmap_scores_parser = subparsers.add_parser(
        "compare-coordinate-heatmap-scores",
        help="Compare saved patch-coordinate score CSV artifacts.",
    )
    compare_models_parser = subparsers.add_parser(
        "compare-models",
        help="Compare zero-shot tissue classification accuracy across models.",
    )
    linear_probe_parser = subparsers.add_parser(
        "linear-probe",
        help="Train a logistic-regression linear probe on frozen embeddings.",
    )
    convert_manifest_parser.add_argument(
        "--input",
        required=True,
        help="Input annotations CSV path.",
    )
    convert_manifest_parser.add_argument(
        "--output",
        required=True,
        help="Output manifest CSV path.",
    )
    convert_manifest_parser.add_argument(
        "--preset",
        choices=["mhist"],
        default=None,
        help="Optional dataset preset conversion rules.",
    )
    convert_manifest_parser.add_argument(
        "--path_column",
        default=None,
        help="Input CSV column name containing image paths (required without --preset).",
    )
    convert_manifest_parser.add_argument(
        "--label_column",
        default=None,
        help="Input CSV column name containing labels.",
    )
    convert_manifest_parser.add_argument(
        "--split_column",
        default=None,
        help="Input CSV column name containing split names.",
    )
    convert_manifest_parser.add_argument(
        "--case_id_column",
        default=None,
        help="Input CSV column name containing case IDs.",
    )
    convert_manifest_parser.add_argument(
        "--slide_id_column",
        default=None,
        help="Input CSV column name containing slide IDs.",
    )
    convert_manifest_parser.add_argument(
        "--image_root",
        default=None,
        help="Optional root folder used when validating relative image paths.",
    )
    convert_manifest_parser.add_argument(
        "--require_exists",
        action="store_true",
        help="Require image paths to exist during conversion.",
    )
    convert_manifest_parser.add_argument(
        "--no_copy_extra_columns",
        action="store_true",
        help="Do not copy unmapped extra columns to output manifest.",
    )
    convert_manifest_parser.add_argument(
        "--no_case_id_from_filename",
        action="store_true",
        help="Disable fallback case_id derivation from image filename stem.",
    )
    imagefolder_manifest_parser.add_argument(
        "--image-dir",
        required=True,
        help="Root of the ImageFolder tree (<class>/<image>, or <split>/<class>/<image>).",
    )
    imagefolder_manifest_parser.add_argument(
        "--output",
        required=True,
        help="Output manifest CSV path.",
    )
    imagefolder_manifest_parser.add_argument(
        "--has-split",
        action="store_true",
        help="Treat the first level as a split (<split>/<class>/<image>).",
    )
    imagefolder_manifest_parser.add_argument(
        "--extensions",
        nargs="+",
        default=None,
        help="Image extensions to include (e.g. png jpg tif). Defaults to common formats.",
    )
    imagefolder_manifest_parser.add_argument(
        "--relative",
        action="store_true",
        help="Write image paths relative to the output CSV directory instead of absolute.",
    )
    sample_manifest_parser.add_argument(
        "--input",
        required=True,
        help="Input manifest CSV path.",
    )
    sample_manifest_parser.add_argument(
        "--output",
        required=True,
        help="Output sampled manifest CSV path.",
    )
    sample_manifest_parser.add_argument(
        "--label_column",
        default="label",
        help="Label column name in input manifest.",
    )
    sample_manifest_parser.add_argument(
        "--split_column",
        default="split",
        help="Split column name in input manifest.",
    )
    sample_manifest_parser.add_argument(
        "--split",
        default=None,
        help="Optional split value to filter before sampling.",
    )
    sample_manifest_parser.add_argument(
        "--samples_per_label",
        type=int,
        default=None,
        help="Maximum samples per label for balanced sampling.",
    )
    sample_manifest_parser.add_argument(
        "--max_total",
        type=int,
        default=None,
        help="Optional maximum total sampled records.",
    )
    sample_manifest_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    summarize_report_parser.add_argument(
        "--task",
        choices=["zero-shot", "retrieval", "prompt-sensitivity"],
        default="zero-shot",
        help="Report type to summarize.",
    )
    summarize_report_parser.add_argument(
        "--report_dir",
        required=True,
        help="Directory containing saved report artifacts.",
    )
    summarize_report_parser.add_argument(
        "--output",
        default=None,
        help="Optional output Markdown path. Defaults to report_dir/experiment_summary.md.",
    )
    compare_reports_parser.add_argument(
        "--task",
        choices=["zero-shot", "retrieval", "prompt-sensitivity"],
        required=True,
        help="Report type to compare.",
    )
    compare_reports_parser.add_argument(
        "--report_dirs",
        nargs="+",
        required=True,
        help="Report directories to compare.",
    )
    compare_reports_parser.add_argument(
        "--run_names",
        nargs="*",
        default=None,
        help="Optional run labels. Provide one label per report directory.",
    )
    compare_reports_parser.add_argument(
        "--output",
        required=True,
        help="Output Markdown path for the comparison summary.",
    )
    validate_config_parser.add_argument(
        "config",
        help="Path to a benchmark JSON config file.",
    )
    zero_shot_grid_parser.add_argument(
        "--config",
        required=True,
        help="Path to a zero-shot prompt-grid JSON config file.",
    )
    zero_shot_grid_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded runs without running model inference.",
    )
    zero_shot_grid_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override for all generated grid report directories.",
    )
    zero_shot_grid_parser.add_argument(
        "--comparison-output",
        default=None,
        help="Optional comparison Markdown output override.",
    )
    render_heatmap_parser.add_argument(
        "--config",
        default=None,
        help="Optional patch-coordinate heatmap JSON config path.",
    )
    render_heatmap_parser.add_argument(
        "--manifest",
        default=None,
        help="Coordinate-aware patch manifest CSV path.",
    )
    render_heatmap_parser.add_argument(
        "--score-csv",
        default=None,
        help="CSV file containing one score per patch.",
    )
    render_heatmap_parser.add_argument(
        "--output",
        default=None,
        help="Output heatmap PNG path.",
    )
    render_heatmap_parser.add_argument(
        "--score-column",
        default=None,
        help="Score column name in --score-csv.",
    )
    render_heatmap_parser.add_argument(
        "--score-path-column",
        default=None,
        help="Image path column in --score-csv when --align-by image_path is used.",
    )
    render_heatmap_parser.add_argument(
        "--align-by",
        choices=["image_path", "order"],
        default=None,
        help="Align scores to manifest records by image_path or row order.",
    )
    render_heatmap_parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image root for resolving relative manifest image paths.",
    )
    render_heatmap_parser.add_argument(
        "--score-image-root",
        default=None,
        help="Optional root for resolving relative score CSV image paths.",
    )
    render_heatmap_parser.add_argument(
        "--path-column",
        default=None,
        help="Manifest image path column name.",
    )
    render_heatmap_parser.add_argument(
        "--x-column",
        default=None,
        help="Manifest x coordinate column name.",
    )
    render_heatmap_parser.add_argument(
        "--y-column",
        default=None,
        help="Manifest y coordinate column name.",
    )
    render_heatmap_parser.add_argument(
        "--require-exists",
        action="store_true",
        default=None,
        help="Require manifest image paths to exist.",
    )
    render_heatmap_parser.add_argument(
        "--title",
        default=None,
        help="Optional heatmap title.",
    )
    render_heatmap_parser.add_argument(
        "--cmap",
        default=None,
        help="Matplotlib colormap name.",
    )
    score_heatmap_parser.add_argument(
        "--config",
        default=None,
        help="Optional prompt-scored patch-coordinate heatmap JSON config path.",
    )
    score_heatmap_parser.add_argument(
        "--manifest",
        default=None,
        help="Coordinate-aware patch manifest CSV path.",
    )
    score_heatmap_parser.add_argument(
        "--prompt",
        default=None,
        help="Text prompt to score against each patch.",
    )
    score_heatmap_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated score CSV, heatmap PNG, and metadata JSON.",
    )
    score_heatmap_parser.add_argument(
        "--score-csv",
        default=None,
        help="Optional score CSV output path. Defaults to output-dir/scores.csv.",
    )
    score_heatmap_parser.add_argument(
        "--heatmap-output",
        default=None,
        help="Optional heatmap PNG output path. Defaults to output-dir/heatmap.png.",
    )
    score_heatmap_parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional metadata JSON output path. Defaults to output-dir/metadata.json.",
    )
    score_heatmap_parser.add_argument(
        "--model",
        default=None,
        help="Model key or Hugging Face model name.",
    )
    score_heatmap_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Device for model inference.",
    )
    score_heatmap_parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image root for resolving relative manifest image paths.",
    )
    score_heatmap_parser.add_argument(
        "--path-column",
        default=None,
        help="Manifest image path column name.",
    )
    score_heatmap_parser.add_argument(
        "--x-column",
        default=None,
        help="Manifest x coordinate column name.",
    )
    score_heatmap_parser.add_argument(
        "--y-column",
        default=None,
        help="Manifest y coordinate column name.",
    )
    score_heatmap_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of manifest records to score.",
    )
    score_heatmap_parser.add_argument(
        "--title",
        default=None,
        help="Optional heatmap title.",
    )
    score_heatmap_parser.add_argument(
        "--cmap",
        default=None,
        help="Matplotlib colormap name.",
    )
    score_heatmap_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and resolved output paths without loading a model.",
    )
    score_heatmap_prompt_set_parser.add_argument(
        "--config",
        required=True,
        help="Prompt-set patch-coordinate heatmap JSON config path.",
    )
    score_heatmap_prompt_set_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print expanded prompt runs and output paths without loading images or models.",
    )
    score_heatmap_prompt_set_parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root override for prompt output directories.",
    )
    score_heatmap_prompt_set_parser.add_argument(
        "--comparison-output-csv",
        default=None,
        help=(
            "Optional prompt-set comparison CSV path. "
            "Defaults to output-root/score_summary.csv."
        ),
    )
    score_heatmap_prompt_set_parser.add_argument(
        "--comparison-output-md",
        default=None,
        help=(
            "Optional prompt-set comparison Markdown path. "
            "Defaults to output-root/score_summary.md."
        ),
    )
    score_heatmap_prompt_set_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of manifest records to include per prompt.",
    )
    compare_heatmap_scores_parser.add_argument(
        "--score-csvs",
        nargs="+",
        required=True,
        help="Saved score CSV paths to compare.",
    )
    compare_heatmap_scores_parser.add_argument(
        "--metadata-jsons",
        nargs="*",
        default=None,
        help=(
            "Optional metadata JSON paths. Provide one path per score CSV. "
            "When omitted, sibling metadata.json files are loaded when present."
        ),
    )
    compare_heatmap_scores_parser.add_argument(
        "--run-names",
        nargs="*",
        default=None,
        help="Optional run labels. Provide one label per score CSV.",
    )
    compare_heatmap_scores_parser.add_argument(
        "--score-column",
        default="score",
        help="Score column name in each score CSV.",
    )
    compare_heatmap_scores_parser.add_argument(
        "--output-csv",
        required=True,
        help="Output comparison CSV path.",
    )
    compare_heatmap_scores_parser.add_argument(
        "--output-md",
        default=None,
        help="Optional output Markdown summary path.",
    )
    compare_heatmap_scores_parser.add_argument(
        "--allow-row-count-mismatch",
        action="store_true",
        help="Allow score CSVs with different row counts.",
    )
    compare_models_parser.add_argument(
        "--manifest",
        required=True,
        help="Standard patch manifest CSV path (image_path,label columns).",
    )
    compare_models_parser.add_argument(
        "--models",
        nargs="+",
        default=["clip", "plip"],
        help="Model keys or Hugging Face names to compare. Default: clip plip.",
    )
    compare_models_parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help=(
            "Class names in class-index order. Required for integer-label "
            "manifests; inferred from unique labels otherwise."
        ),
    )
    compare_models_parser.add_argument(
        "--prompt-template",
        default="an H&E image of {}.",
        help="Prompt template applied to each class name. Use '{}' as the slot.",
    )
    compare_models_parser.add_argument(
        "--class-prompts",
        nargs="+",
        default=None,
        help="Optional explicit per-class prompts that override --prompt-template.",
    )
    compare_models_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for model inference.",
    )
    compare_models_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional maximum number of manifest records to evaluate.",
    )
    compare_models_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Image-encoding batch size.",
    )
    compare_models_parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image root for resolving relative manifest image paths.",
    )
    compare_models_parser.add_argument(
        "--split",
        default=None,
        help="Optional split value to filter manifest records before evaluation.",
    )
    compare_models_parser.add_argument(
        "--output-dir",
        default="outputs/model_comparison",
        help="Directory for the comparison CSV, bar chart PNG, and metadata JSON.",
    )
    compare_models_parser.add_argument(
        "--title",
        default=None,
        help="Optional chart title.",
    )
    compare_models_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap accuracy intervals. Default: 0.95.",
    )
    compare_models_parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for accuracy intervals. Default: 2000.",
    )
    compare_models_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible bootstrap intervals. Default: 0.",
    )
    compare_models_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and resolved class prompts without loading models.",
    )

    linear_probe_parser.add_argument(
        "--manifest",
        required=True,
        help="Standard patch manifest CSV with image_path, label, and split columns.",
    )
    linear_probe_parser.add_argument(
        "--image-root",
        default=None,
        help="Optional image root for resolving relative manifest image paths.",
    )
    linear_probe_parser.add_argument(
        "--model",
        default="clip",
        help="Model key or Hugging Face name used to encode patches. Default: clip.",
    )
    linear_probe_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for model inference.",
    )
    linear_probe_parser.add_argument(
        "--train-split",
        default="train",
        help="Manifest split value used to fit the probe. Default: train.",
    )
    linear_probe_parser.add_argument(
        "--test-split",
        default="test",
        help="Manifest split value used to evaluate the probe. Default: test.",
    )
    linear_probe_parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Optional explicit class order. Inferred from train labels otherwise.",
    )
    linear_probe_parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse L2 regularization strength for logistic regression. Default: 1.0.",
    )
    linear_probe_parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations. Default: 1000.",
    )
    linear_probe_parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip L2-normalizing embeddings before fitting the probe.",
    )
    linear_probe_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Image-encoding batch size.",
    )
    linear_probe_parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on records per split for a quick smoke run.",
    )
    linear_probe_parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for the bootstrap accuracy interval. Default: 0.95.",
    )
    linear_probe_parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=2000,
        help="Number of bootstrap resamples for the accuracy interval. Default: 2000.",
    )
    linear_probe_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the probe fit and bootstrap interval. Default: 0.",
    )
    linear_probe_parser.add_argument(
        "--output-dir",
        default="outputs/linear_probe",
        help="Directory for predictions.csv, metrics.json, and errors.csv.",
    )
    linear_probe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report split sizes without loading models.",
    )

    return parser


def _handle_version() -> int:
    print(f"PathVLM-LiteBench version {version}")
    return 0


def _handle_models() -> int:
    print("Registered models:")
    for item in list_available_models():
        print(f"- key: {item['key']}")
        print(f"  model_name: {item['model_name']}")
        print(f"  implemented: {item['implemented']}")
        print(f"  description: {item['description']}")
    return 0


def _handle_demos() -> int:
    print("Tip: run a bundled demo directly with: pathvlm-litebench demo <name>")
    print("     (names: " + ", ".join(DEMO_SCRIPTS) + ")")
    print()
    print("Available demo commands:")
    print("python examples/01_patch_text_retrieval_demo.py --model clip")
    print("python examples/02_zero_shot_classification_demo.py --model clip")
    print("python examples/03_prompt_sensitivity_demo.py --model clip")
    print("python examples/04_retrieval_metrics_demo.py")
    print("python examples/05_patch_coordinate_heatmap_demo.py")
    print("python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_clip_sample.json")
    print("python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_plip_sample.json")
    print("pathvlm-litebench validate-config configs/zero_shot_mhist_clip_sample.json")
    print("pathvlm-litebench validate-config configs/zero_shot_mhist_plip_sample.json")
    print("pathvlm-litebench validate-config configs/zero_shot_prompt_grid_mhist_sample.json")
    print("pathvlm-litebench run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --dry-run")
    print("pathvlm-litebench run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --output-root outputs/zero_shot_prompt_grid_mhist_sample_run")
    print("pathvlm-litebench validate-config configs/patch_coordinate_heatmap_demo_config.json")
    print("pathvlm-litebench validate-config configs/patch_coordinate_heatmap_scoring_demo_config.json")
    print("pathvlm-litebench validate-config configs/patch_coordinate_heatmap_prompt_set_demo_config.json")
    print("pathvlm-litebench render-coordinate-heatmap --config configs/patch_coordinate_heatmap_demo_config.json")
    print("pathvlm-litebench score-coordinate-heatmap --config configs/patch_coordinate_heatmap_scoring_demo_config.json --dry-run")
    print("pathvlm-litebench score-coordinate-heatmap --config configs/patch_coordinate_heatmap_scoring_demo_config.json")
    print("pathvlm-litebench score-coordinate-heatmap-prompt-set --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json --dry-run")
    print("pathvlm-litebench score-coordinate-heatmap-prompt-set --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json")
    print("pathvlm-litebench compare-coordinate-heatmap-scores --score-csvs outputs/patch_coordinate_heatmap_scored_tumor/scores.csv outputs/patch_coordinate_heatmap_scored_lymphocyte/scores.csv --run-names tumor lymphocyte --output-csv outputs/patch_coordinate_heatmap_comparison/score_summary.csv --output-md outputs/patch_coordinate_heatmap_comparison/score_summary.md")
    print("pathvlm-litebench render-coordinate-heatmap --manifest dataset/patch_coordinates/coordinate_manifest.csv --score-csv outputs/patch_coordinate_heatmap_demo/scores.csv --output outputs/patch_coordinate_heatmap_demo/heatmap.png")
    print("pathvlm-litebench score-coordinate-heatmap --manifest dataset/patch_coordinates/coordinate_manifest.csv --prompt \"a histopathology image of tumor tissue\" --output-dir outputs/patch_coordinate_heatmap_scored --model clip")
    print("pathvlm-litebench compare-models --manifest dataset/CRC_VAL_HE_100_sample_manifest.csv --models clip plip conch --class-names \"adipose tissue\" background debris lymphocytes mucus \"smooth muscle\" \"normal colon mucosa\" \"cancer-associated stroma\" \"colorectal adenocarcinoma epithelium\" --output-dir outputs/model_comparison")
    print("pathvlm-litebench build-imagefolder-manifest --image-dir path/to/imagefolder --output dataset/imagefolder_manifest.csv")
    return 0


def _examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


def _print_demo_list() -> None:
    print("Available demos:")
    for name, script in DEMO_SCRIPTS.items():
        print(f"- {name}  (examples/{script})")
    print()
    print("Run one with: pathvlm-litebench demo <name> [demo args...]")
    print("Example: pathvlm-litebench demo retrieval --model clip --device auto")
    if not _examples_dir().exists():
        print()
        print(
            "Note: the bundled demo scripts ship with a source checkout, not the "
            "PyPI package."
        )
        print(
            "Clone https://github.com/GuoYixuan0130/PathVLM-LiteBench to run them."
        )


def _handle_demo(args: argparse.Namespace) -> int:
    if args.name is None:
        _print_demo_list()
        return 0

    script = DEMO_SCRIPTS.get(args.name)
    if script is None:
        print(f"Error: unknown demo '{args.name}'.")
        print(f"Available demos: {', '.join(DEMO_SCRIPTS)}")
        return 1

    script_path = _examples_dir() / script
    if not script_path.exists():
        print(f"Error: demo script not found: {script_path}")
        print("The 'demo' command runs from a source checkout of the repository.")
        return 1

    command = [sys.executable, str(script_path), *args.demo_args]
    return subprocess.call(command)


def _handle_convert_manifest(args: argparse.Namespace) -> int:
    if args.preset == "mhist":
        saved_path = convert_mhist_manifest(
            annotations_csv=args.input,
            output_csv=args.output,
            image_root=args.image_root,
            require_exists=args.require_exists,
        )
        print(f"Saved converted manifest to: {saved_path}")
        return 0

    if args.path_column is None:
        print("Error: --path_column is required when --preset is not provided.")
        return 1

    saved_path = convert_manifest(
        input_csv=args.input,
        output_csv=args.output,
        path_column=args.path_column,
        label_column=args.label_column,
        split_column=args.split_column,
        case_id_column=args.case_id_column,
        slide_id_column=args.slide_id_column,
        image_root=args.image_root,
        require_exists=args.require_exists,
        case_id_from_filename=not args.no_case_id_from_filename,
        copy_extra_columns=not args.no_copy_extra_columns,
    )
    print(f"Saved converted manifest to: {saved_path}")
    return 0


def _handle_build_imagefolder_manifest(args: argparse.Namespace) -> int:
    try:
        summary = build_imagefolder_manifest(
            image_dir=args.image_dir,
            output_csv=args.output,
            has_split=args.has_split,
            extensions=args.extensions,
            relative=args.relative,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved imagefolder manifest to: {summary['output_csv']}")
    print(f"Number of records: {summary['num_records']}")
    print(f"Number of classes: {summary['num_classes']}")
    print(f"Label distribution: {summary['label_distribution']}")
    if summary["split_distribution"]:
        print(f"Split distribution: {summary['split_distribution']}")
    return 0


def _handle_sample_manifest(args: argparse.Namespace) -> int:
    saved_path = sample_manifest(
        input_csv=args.input,
        output_csv=args.output,
        label_column=args.label_column,
        split_column=args.split_column,
        split=args.split,
        samples_per_label=args.samples_per_label,
        max_total=args.max_total,
        seed=args.seed,
    )
    summary = summarize_manifest(
        manifest_csv=saved_path,
        label_column=args.label_column,
        split_column=args.split_column,
    )
    print(f"Saved sampled manifest to: {saved_path}")
    print(f"Number of records: {summary['num_records']}")
    print(f"Label distribution: {summary['label_distribution']}")
    print(f"Split distribution: {summary['split_distribution']}")
    return 0


def _handle_summarize_report(args: argparse.Namespace) -> int:
    if args.task == "zero-shot":
        saved_path = save_zero_shot_experiment_summary(
            report_dir=args.report_dir,
            output_path=args.output,
        )
        print(f"Saved experiment summary to: {saved_path}")
        return 0

    if args.task == "retrieval":
        saved_path = save_retrieval_experiment_summary(
            report_dir=args.report_dir,
            output_path=args.output,
        )
        print(f"Saved experiment summary to: {saved_path}")
        return 0

    if args.task == "prompt-sensitivity":
        saved_path = save_prompt_sensitivity_experiment_summary(
            report_dir=args.report_dir,
            output_path=args.output,
        )
        print(f"Saved experiment summary to: {saved_path}")
        return 0

    print(f"Error: unsupported report task: {args.task}")
    return 1


def _handle_compare_reports(args: argparse.Namespace) -> int:
    try:
        saved_path = save_experiment_comparison_summary(
            task=args.task,
            report_dirs=args.report_dirs,
            run_names=args.run_names,
            output_path=args.output,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved comparison summary to: {saved_path}")
    return 0


def _apply_zero_shot_grid_overrides(
    config,
    *,
    output_root: str | None = None,
    comparison_output: str | None = None,
):
    """
    Apply runtime zero-shot grid path overrides without mutating the loaded config.
    """
    if output_root is None and comparison_output is None:
        return config

    return replace(
        config,
        output_root=output_root if output_root is not None else config.output_root,
        comparison_output=comparison_output,
    )


def _handle_validate_config(args: argparse.Namespace) -> int:
    try:
        with open(args.config, "r", encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
        task = raw_config.get("task")
        if task == "zero_shot_grid":
            from .evaluation.zero_shot_grid import (
                expand_zero_shot_grid_runs,
                load_zero_shot_grid_config,
            )

            config = load_zero_shot_grid_config(args.config)
            runs = expand_zero_shot_grid_runs(config)
            prompt_keys = [prompt_pair.key for prompt_pair in config.prompt_pairs]
            print("Config valid: zero_shot_grid")
            print(f"Models: {', '.join(config.models)}")
            print(f"Prompt pairs: {', '.join(prompt_keys)}")
            print(f"Runs: {len(runs)}")
            print(f"Device: {config.device}")
            print(f"Output root: {config.output_root}")
            return 0

        if task == "patch_coordinate_heatmap":
            from .config import load_patch_coordinate_heatmap_config

            config = load_patch_coordinate_heatmap_config(args.config)
            print("Config valid: patch_coordinate_heatmap")
            print(f"Manifest: {config.manifest}")
            print(f"Score CSV: {config.score_csv}")
            print(f"Output: {config.output}")
            print(f"Align by: {config.align_by}")
            print(f"Score column: {config.score_column}")
            return 0

        if task == "patch_coordinate_heatmap_scoring":
            from .config import load_patch_coordinate_heatmap_scoring_config

            config = load_patch_coordinate_heatmap_scoring_config(args.config)
            _, score_csv, heatmap_output, metadata_output = (
                _resolve_score_heatmap_output_paths(config)
            )
            print("Config valid: patch_coordinate_heatmap_scoring")
            print(f"Manifest: {config.manifest}")
            print(f"Prompt: {config.prompt}")
            print(f"Model: {config.model}")
            print(f"Device: {config.device}")
            print(f"Output dir: {config.output_dir}")
            print(f"Score CSV: {score_csv}")
            print(f"Heatmap output: {heatmap_output}")
            print(f"Metadata output: {metadata_output}")
            return 0

        if task == "patch_coordinate_heatmap_prompt_set":
            from .config import load_patch_coordinate_heatmap_prompt_set_config

            config = load_patch_coordinate_heatmap_prompt_set_config(args.config)
            prompt_keys = [prompt.key for prompt in config.prompts]
            comparison_csv, comparison_md = (
                _resolve_prompt_set_comparison_output_paths(config)
            )
            print("Config valid: patch_coordinate_heatmap_prompt_set")
            print(f"Manifest: {config.manifest}")
            print(f"Output root: {config.output_root}")
            print(f"Comparison CSV: {comparison_csv}")
            print(f"Comparison Markdown: {comparison_md}")
            print(f"Model: {config.model}")
            print(f"Device: {config.device}")
            print(f"Prompts: {len(config.prompts)}")
            print(f"Prompt keys: {', '.join(prompt_keys)}")
            print(f"Max images: {config.max_images}")
            return 0

        config = load_benchmark_config(args.config)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Config valid: {config.task}")
    print(f"Model: {config.model}")
    print(f"Device: {config.device}")
    if config.task == "retrieval":
        prompt_count = len(config.prompts) if config.prompts is not None else 0
        print(f"Prompts: {prompt_count}")
        print(f"Output dir: {config.output_dir}")
    elif config.task == "zero_shot":
        class_count = len(config.class_names) if config.class_names is not None else 0
        print(f"Classes: {class_count}")
        print(f"Report dir: {config.report_dir}")
    elif config.task == "prompt_sensitivity":
        concept_count = len(config.concepts) if config.concepts is not None else 0
        print(f"Concepts: {concept_count}")
        print(f"Report dir: {config.report_dir}")
    return 0


def _handle_run_zero_shot_grid(args: argparse.Namespace) -> int:
    from .evaluation.zero_shot_grid import (
        load_zero_shot_grid_config,
        run_zero_shot_grid,
    )

    try:
        config = load_zero_shot_grid_config(args.config)
        config = _apply_zero_shot_grid_overrides(
            config,
            output_root=args.output_root,
            comparison_output=args.comparison_output,
        )
        result = run_zero_shot_grid(config, dry_run=args.dry_run)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    runs = result["runs"]
    print(f"Zero-shot grid runs: {len(runs)}")
    for run in runs:
        print(f"- {run.run_name}:")
        print(f"  model: {run.model}")
        print(f"  prompt_key: {run.prompt_key}")
        print(f"  report_dir: {run.report_dir}")
        if run.log_path is not None:
            print(f"  log_path: {run.log_path}")

    comparison_path = result.get("comparison_path")
    if args.dry_run:
        print("Dry run only. No model inference was run.")
    elif comparison_path:
        print(f"Saved comparison summary to: {comparison_path}")
    return 0


def _load_heatmap_scores(
    *,
    records,
    score_csv: str,
    score_column: str,
    score_path_column: str,
    align_by: str,
    score_image_root: str | None,
) -> list[float]:
    score_csv_path = Path(score_csv)
    if not score_csv_path.exists():
        raise FileNotFoundError(f"Score CSV file not found: {score_csv_path}")

    with score_csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Score CSV is empty: {score_csv_path}")
        fieldnames = list(reader.fieldnames)

        if score_column not in fieldnames:
            raise ValueError(
                f"Score column '{score_column}' not found in score CSV. "
                f"Available columns: {', '.join(fieldnames)}"
            )

        rows = list(reader)

    if align_by == "order":
        if len(rows) != len(records):
            raise ValueError(
                f"Score CSV row count must match manifest record count when "
                f"aligning by order: {len(rows)} vs {len(records)}"
            )
        return [_parse_score(row, score_column, idx + 2) for idx, row in enumerate(rows)]

    if score_path_column not in fieldnames:
        raise ValueError(
            f"Score path column '{score_path_column}' not found in score CSV. "
            f"Available columns: {', '.join(fieldnames)}"
        )

    score_root = Path(score_image_root) if score_image_root is not None else score_csv_path.parent
    scores_by_path: dict[str, float] = {}
    for idx, row in enumerate(rows, start=2):
        raw_path = row.get(score_path_column)
        if raw_path is None or not raw_path.strip():
            raise ValueError(
                f"Empty score path column '{score_path_column}' at row {idx} "
                f"in score CSV: {score_csv_path}"
            )
        path_key = _resolve_path_key(raw_path, score_root)
        if path_key in scores_by_path:
            raise ValueError(f"Duplicate score path at row {idx}: {raw_path}")
        scores_by_path[path_key] = _parse_score(row, score_column, idx)

    scores: list[float] = []
    for record in records:
        record_key = str(Path(record.image_path).resolve())
        if record_key not in scores_by_path:
            raise ValueError(f"No score found for manifest image path: {record.image_path}")
        scores.append(scores_by_path[record_key])

    return scores


def _parse_score(row: dict[str, str], score_column: str, row_idx: int) -> float:
    raw_score = row.get(score_column)
    if raw_score is None or not raw_score.strip():
        raise ValueError(f"Empty score column '{score_column}' at row {row_idx}")

    try:
        return float(raw_score)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric score at row {row_idx}: {raw_score!r}"
        ) from exc


def _resolve_path_key(raw_path: str, root: Path) -> str:
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = root / path
    return str(path.resolve())


def _load_render_heatmap_args_config(
    args: argparse.Namespace,
) -> PatchCoordinateHeatmapConfig:
    from .config import load_patch_coordinate_heatmap_config

    if args.config is None:
        if args.manifest is None:
            raise ValueError("--manifest is required when --config is not provided.")
        if args.score_csv is None:
            raise ValueError("--score-csv is required when --config is not provided.")
        if args.output is None:
            raise ValueError("--output is required when --config is not provided.")
        return PatchCoordinateHeatmapConfig(
            manifest=args.manifest,
            score_csv=args.score_csv,
            output=args.output,
            score_column=args.score_column or "score",
            score_path_column=args.score_path_column or "image_path",
            align_by=args.align_by or "image_path",
            image_root=args.image_root,
            score_image_root=args.score_image_root,
            path_column=args.path_column or "image_path",
            x_column=args.x_column or "x",
            y_column=args.y_column or "y",
            require_exists=bool(args.require_exists),
            title=args.title,
            cmap=args.cmap or "viridis",
        )

    config = load_patch_coordinate_heatmap_config(args.config)
    return replace(
        config,
        manifest=args.manifest if args.manifest is not None else config.manifest,
        score_csv=args.score_csv if args.score_csv is not None else config.score_csv,
        output=args.output if args.output is not None else config.output,
        score_column=(
            args.score_column if args.score_column is not None else config.score_column
        ),
        score_path_column=(
            args.score_path_column
            if args.score_path_column is not None
            else config.score_path_column
        ),
        align_by=args.align_by if args.align_by is not None else config.align_by,
        image_root=args.image_root if args.image_root is not None else config.image_root,
        score_image_root=(
            args.score_image_root
            if args.score_image_root is not None
            else config.score_image_root
        ),
        path_column=args.path_column if args.path_column is not None else config.path_column,
        x_column=args.x_column if args.x_column is not None else config.x_column,
        y_column=args.y_column if args.y_column is not None else config.y_column,
        require_exists=(
            args.require_exists
            if args.require_exists is not None
            else config.require_exists
        ),
        title=args.title if args.title is not None else config.title,
        cmap=args.cmap if args.cmap is not None else config.cmap,
    )


def _handle_render_coordinate_heatmap(args: argparse.Namespace) -> int:
    from .data import load_coordinate_patch_manifest
    from .visualization import aggregate_patch_scores_to_grid, save_score_heatmap

    try:
        config = _load_render_heatmap_args_config(args)
        records = load_coordinate_patch_manifest(
            manifest_path=config.manifest,
            image_root=config.image_root,
            path_column=config.path_column,
            x_column=config.x_column,
            y_column=config.y_column,
            require_exists=config.require_exists,
        )
        scores = _load_heatmap_scores(
            records=records,
            score_csv=config.score_csv,
            score_column=config.score_column,
            score_path_column=config.score_path_column,
            align_by=config.align_by,
            score_image_root=config.score_image_root,
        )
        grid = aggregate_patch_scores_to_grid(records, scores)
        saved_path = save_score_heatmap(
            grid,
            output_path=config.output,
            title=config.title,
            cmap=config.cmap,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved patch-coordinate heatmap to: {saved_path}")
    print(f"Patches: {len(records)}")
    print(f"Grid shape: {len(grid.y_values)} rows x {len(grid.x_values)} columns")
    return 0


def _load_score_heatmap_args_config(
    args: argparse.Namespace,
) -> PatchCoordinateHeatmapScoringConfig:
    from .config import load_patch_coordinate_heatmap_scoring_config

    if args.config is None:
        if args.manifest is None:
            raise ValueError("--manifest is required when --config is not provided.")
        if args.prompt is None:
            raise ValueError("--prompt is required when --config is not provided.")
        return PatchCoordinateHeatmapScoringConfig(
            manifest=args.manifest,
            prompt=args.prompt,
            output_dir=args.output_dir or "outputs/patch_coordinate_heatmap_scored",
            score_csv=args.score_csv,
            heatmap_output=args.heatmap_output,
            metadata_output=args.metadata_output,
            model=args.model or "clip",
            device=args.device or "auto",
            image_root=args.image_root,
            path_column=args.path_column or "image_path",
            x_column=args.x_column or "x",
            y_column=args.y_column or "y",
            max_images=args.max_images,
            title=args.title,
            cmap=args.cmap or "viridis",
        )

    config = load_patch_coordinate_heatmap_scoring_config(args.config)
    return replace(
        config,
        manifest=args.manifest if args.manifest is not None else config.manifest,
        prompt=args.prompt if args.prompt is not None else config.prompt,
        output_dir=args.output_dir if args.output_dir is not None else config.output_dir,
        score_csv=args.score_csv if args.score_csv is not None else config.score_csv,
        heatmap_output=(
            args.heatmap_output
            if args.heatmap_output is not None
            else config.heatmap_output
        ),
        metadata_output=(
            args.metadata_output
            if args.metadata_output is not None
            else config.metadata_output
        ),
        model=args.model if args.model is not None else config.model,
        device=args.device if args.device is not None else config.device,
        image_root=args.image_root if args.image_root is not None else config.image_root,
        path_column=args.path_column if args.path_column is not None else config.path_column,
        x_column=args.x_column if args.x_column is not None else config.x_column,
        y_column=args.y_column if args.y_column is not None else config.y_column,
        max_images=args.max_images if args.max_images is not None else config.max_images,
        title=args.title if args.title is not None else config.title,
        cmap=args.cmap if args.cmap is not None else config.cmap,
    )


def _resolve_score_heatmap_output_paths(
    config: PatchCoordinateHeatmapScoringConfig,
) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(config.output_dir)
    score_csv = (
        Path(config.score_csv)
        if config.score_csv is not None
        else output_dir / "scores.csv"
    )
    heatmap_output = (
        Path(config.heatmap_output)
        if config.heatmap_output is not None
        else output_dir / "heatmap.png"
    )
    metadata_output = (
        Path(config.metadata_output)
        if config.metadata_output is not None
        else output_dir / "metadata.json"
    )
    return output_dir, score_csv, heatmap_output, metadata_output


def _apply_prompt_set_overrides(
    config: PatchCoordinateHeatmapPromptSetConfig,
    *,
    output_root: str | None = None,
    comparison_output_csv: str | None = None,
    comparison_output_md: str | None = None,
    max_images: int | None = None,
) -> PatchCoordinateHeatmapPromptSetConfig:
    if (
        output_root is None
        and comparison_output_csv is None
        and comparison_output_md is None
        and max_images is None
    ):
        return config

    return replace(
        config,
        output_root=output_root if output_root is not None else config.output_root,
        comparison_output_csv=(
            comparison_output_csv
            if comparison_output_csv is not None
            else config.comparison_output_csv
        ),
        comparison_output_md=(
            comparison_output_md
            if comparison_output_md is not None
            else config.comparison_output_md
        ),
        max_images=max_images if max_images is not None else config.max_images,
    )


def _resolve_prompt_set_comparison_output_paths(
    config: PatchCoordinateHeatmapPromptSetConfig,
) -> tuple[Path, Path]:
    output_root = Path(config.output_root)
    output_csv = (
        Path(config.comparison_output_csv)
        if config.comparison_output_csv is not None
        else output_root / "score_summary.csv"
    )
    output_md = (
        Path(config.comparison_output_md)
        if config.comparison_output_md is not None
        else output_root / "score_summary.md"
    )
    return output_csv, output_md


def _expand_prompt_set_scoring_configs(
    config: PatchCoordinateHeatmapPromptSetConfig,
) -> list[tuple[str, PatchCoordinateHeatmapScoringConfig]]:
    runs: list[tuple[str, PatchCoordinateHeatmapScoringConfig]] = []
    for prompt in config.prompts:
        output_dir = (
            Path(prompt.output_dir)
            if prompt.output_dir is not None
            else Path(config.output_root) / prompt.key
        )
        run_config = PatchCoordinateHeatmapScoringConfig(
            manifest=config.manifest,
            prompt=prompt.prompt,
            output_dir=str(output_dir),
            model=config.model,
            device=config.device,
            image_root=config.image_root,
            path_column=config.path_column,
            x_column=config.x_column,
            y_column=config.y_column,
            max_images=config.max_images,
            title=prompt.title,
            cmap=prompt.cmap or config.cmap,
        )
        runs.append((prompt.key, run_config))
    return runs


def _build_score_heatmap_metadata(
    *,
    config: PatchCoordinateHeatmapScoringConfig,
    patch_count: int,
    score_csv: Path,
    heatmap_output: Path,
    metadata_output: Path,
    prompt_key: str | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "task": "patch_coordinate_heatmap_scoring",
        "version": version,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "manifest": config.manifest,
        "image_root": config.image_root,
        "path_column": config.path_column,
        "x_column": config.x_column,
        "y_column": config.y_column,
        "prompt": config.prompt,
        "model": config.model,
        "device": config.device,
        "output_dir": config.output_dir,
        "score_csv": str(score_csv),
        "heatmap_output": str(heatmap_output),
        "metadata_output": str(metadata_output),
        "max_images": config.max_images,
        "patch_count": patch_count,
        "title": config.title,
        "cmap": config.cmap,
    }
    if prompt_key is not None:
        metadata["prompt_key"] = prompt_key
    return metadata


def _save_score_heatmap_metadata(
    *,
    config: PatchCoordinateHeatmapScoringConfig,
    patch_count: int,
    score_csv: Path,
    heatmap_output: Path,
    metadata_output: Path,
    prompt_key: str | None = None,
) -> str:
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata = _build_score_heatmap_metadata(
        config=config,
        patch_count=patch_count,
        score_csv=score_csv,
        heatmap_output=heatmap_output,
        metadata_output=metadata_output,
        prompt_key=prompt_key,
    )
    metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(metadata_output)


def _score_coordinate_heatmap_run(
    *,
    config: PatchCoordinateHeatmapScoringConfig,
    records,
    images,
    model,
    prompt_key: str | None = None,
) -> tuple[str, str, str]:
    from .evaluation import score_patch_images_for_prompt
    from .visualization import (
        aggregate_patch_scores_to_grid,
        save_patch_scores_csv,
        save_score_heatmap,
    )

    _, score_csv, heatmap_output, metadata_output = (
        _resolve_score_heatmap_output_paths(config)
    )

    scores = score_patch_images_for_prompt(
        images=images,
        prompt=config.prompt,
        model=model,
    )
    grid = aggregate_patch_scores_to_grid(records, scores)
    saved_scores = save_patch_scores_csv(
        records,
        scores,
        score_csv,
        prompt=config.prompt,
    )
    saved_heatmap = save_score_heatmap(
        grid,
        heatmap_output,
        title=config.title or config.prompt,
        cmap=config.cmap,
    )
    saved_metadata = _save_score_heatmap_metadata(
        config=config,
        patch_count=len(records),
        score_csv=score_csv,
        heatmap_output=heatmap_output,
        metadata_output=metadata_output,
        prompt_key=prompt_key,
    )
    return saved_scores, saved_heatmap, saved_metadata


def _save_prompt_set_comparison(
    *,
    saved_runs: list[tuple[str, PatchCoordinateHeatmapScoringConfig, str, str, str]],
    output_csv: Path,
    output_md: Path,
) -> tuple[str, str]:
    from .visualization import (
        compare_patch_score_csvs,
        save_patch_score_comparison_csv,
        save_patch_score_comparison_summary,
    )

    score_csvs = [saved_scores for _, _, saved_scores, _, _ in saved_runs]
    metadata_jsons = [saved_metadata for _, _, _, _, saved_metadata in saved_runs]
    run_names = [prompt_key for prompt_key, _, _, _, _ in saved_runs]

    summaries = compare_patch_score_csvs(
        score_csvs,
        metadata_jsons=metadata_jsons,
        run_names=run_names,
    )
    saved_csv = save_patch_score_comparison_csv(summaries, output_csv)
    saved_md = save_patch_score_comparison_summary(summaries, output_md)
    return saved_csv, saved_md


def _handle_score_coordinate_heatmap(args: argparse.Namespace) -> int:
    from .data import (
        coordinate_records_to_image_paths,
        load_coordinate_patch_manifest,
    )

    try:
        config = _load_score_heatmap_args_config(args)
        _, score_csv, heatmap_output, metadata_output = (
            _resolve_score_heatmap_output_paths(config)
        )

        records = load_coordinate_patch_manifest(
            manifest_path=config.manifest,
            image_root=config.image_root,
            path_column=config.path_column,
            x_column=config.x_column,
            y_column=config.y_column,
            require_exists=True,
        )
        if config.max_images is not None:
            records = records[: config.max_images]

        if args.dry_run:
            print("Dry run only. No model inference was run.")
            print(f"Manifest: {config.manifest}")
            print(f"Patches: {len(records)}")
            print(f"Score CSV: {score_csv}")
            print(f"Heatmap output: {heatmap_output}")
            print(f"Metadata output: {metadata_output}")
            print(f"Prompt: {config.prompt}")
            print(f"Model: {config.model}")
            print(f"Device: {config.device}")
            return 0

        from .data import load_patch_images_from_paths
        from .models import create_model

        image_paths = coordinate_records_to_image_paths(records)
        images, _ = load_patch_images_from_paths(image_paths)

        model = create_model(config.model, device=config.device)
        saved_scores, saved_heatmap, saved_metadata = _score_coordinate_heatmap_run(
            config=config,
            records=records,
            images=images,
            model=model,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved patch-coordinate scores to: {saved_scores}")
    print(f"Saved patch-coordinate heatmap to: {saved_heatmap}")
    print(f"Saved patch-coordinate metadata to: {saved_metadata}")
    print(f"Patches: {len(records)}")
    print(f"Prompt: {config.prompt}")
    print(f"Model: {config.model}")
    return 0


def _handle_score_coordinate_heatmap_prompt_set(args: argparse.Namespace) -> int:
    from .config import load_patch_coordinate_heatmap_prompt_set_config
    from .data import load_coordinate_patch_manifest

    try:
        config = load_patch_coordinate_heatmap_prompt_set_config(args.config)
        config = _apply_prompt_set_overrides(
            config,
            output_root=args.output_root,
            comparison_output_csv=args.comparison_output_csv,
            comparison_output_md=args.comparison_output_md,
            max_images=args.max_images,
        )
        runs = _expand_prompt_set_scoring_configs(config)
        comparison_csv, comparison_md = _resolve_prompt_set_comparison_output_paths(
            config
        )

        records = load_coordinate_patch_manifest(
            manifest_path=config.manifest,
            image_root=config.image_root,
            path_column=config.path_column,
            x_column=config.x_column,
            y_column=config.y_column,
            require_exists=not args.dry_run,
        )
        if config.max_images is not None:
            records = records[: config.max_images]

        if args.dry_run:
            print("Dry run only. No model inference was run.")
            print(f"Manifest: {config.manifest}")
            print(f"Patches per prompt: {len(records)}")
            print(f"Output root: {config.output_root}")
            print(f"Comparison CSV: {comparison_csv}")
            print(f"Comparison Markdown: {comparison_md}")
            print(f"Model: {config.model}")
            print(f"Device: {config.device}")
            print(f"Prompt-set runs: {len(runs)}")
            for prompt_key, run_config in runs:
                output_dir, score_csv, heatmap_output, metadata_output = (
                    _resolve_score_heatmap_output_paths(run_config)
                )
                print(f"- {prompt_key}:")
                print(f"  output_dir: {output_dir}")
                print(f"  score_csv: {score_csv}")
                print(f"  heatmap_output: {heatmap_output}")
                print(f"  metadata_output: {metadata_output}")
                print(f"  prompt: {run_config.prompt}")
                print(f"  title: {run_config.title or run_config.prompt}")
                print(f"  cmap: {run_config.cmap}")
            return 0

        from .data import (
            coordinate_records_to_image_paths,
            load_patch_images_from_paths,
        )
        from .models import create_model

        image_paths = coordinate_records_to_image_paths(records)
        images, _ = load_patch_images_from_paths(image_paths)
        model = create_model(config.model, device=config.device)

        saved_runs: list[
            tuple[str, PatchCoordinateHeatmapScoringConfig, str, str, str]
        ] = []
        for prompt_key, run_config in runs:
            saved_scores, saved_heatmap, saved_metadata = _score_coordinate_heatmap_run(
                config=run_config,
                records=records,
                images=images,
                model=model,
                prompt_key=prompt_key,
            )
            saved_runs.append(
                (
                    prompt_key,
                    run_config,
                    saved_scores,
                    saved_heatmap,
                    saved_metadata,
                )
            )
        saved_comparison_csv, saved_comparison_md = _save_prompt_set_comparison(
            saved_runs=saved_runs,
            output_csv=comparison_csv,
            output_md=comparison_md,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Saved prompt-set patch-coordinate heatmap outputs.")
    print(f"Manifest: {config.manifest}")
    print(f"Patches per prompt: {len(records)}")
    print(f"Output root: {config.output_root}")
    print(f"Saved comparison CSV to: {saved_comparison_csv}")
    print(f"Saved comparison Markdown to: {saved_comparison_md}")
    print(f"Model: {config.model}")
    print(f"Device: {config.device}")
    print(f"Prompt-set runs: {len(saved_runs)}")
    for (
        prompt_key,
        run_config,
        saved_scores,
        saved_heatmap,
        saved_metadata,
    ) in saved_runs:
        print(f"- {prompt_key}:")
        print(f"  score_csv: {saved_scores}")
        print(f"  heatmap_output: {saved_heatmap}")
        print(f"  metadata_output: {saved_metadata}")
        print(f"  prompt: {run_config.prompt}")
    return 0


def _handle_compare_coordinate_heatmap_scores(args: argparse.Namespace) -> int:
    from .visualization import (
        compare_patch_score_csvs,
        save_patch_score_comparison_csv,
        save_patch_score_comparison_summary,
    )

    try:
        summaries = compare_patch_score_csvs(
            args.score_csvs,
            score_column=args.score_column,
            metadata_jsons=args.metadata_jsons,
            run_names=args.run_names,
            require_equal_row_count=not args.allow_row_count_mismatch,
        )
        saved_csv = save_patch_score_comparison_csv(summaries, args.output_csv)
        saved_md = None
        if args.output_md is not None:
            saved_md = save_patch_score_comparison_summary(
                summaries,
                args.output_md,
            )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved patch-coordinate score comparison CSV to: {saved_csv}")
    if saved_md is not None:
        print(f"Saved patch-coordinate score comparison summary to: {saved_md}")
    print(f"Runs: {len(summaries)}")
    for summary in summaries:
        print(
            f"- {summary.run_name}: rows={summary.row_count}, "
            f"mean={summary.score_mean:.6g}, std={summary.score_std:.6g}"
        )
    return 0


def _resolve_compare_models_class_names(
    args: argparse.Namespace,
    records,
) -> list[str]:
    from .data import get_unique_labels

    if args.class_names is not None:
        return list(args.class_names)

    unique_labels = get_unique_labels(records)
    if not unique_labels:
        raise ValueError("Manifest has no labels; provide --class-names.")
    if any(label.strip().isdigit() for label in unique_labels):
        raise ValueError(
            "Manifest labels look like integer class indices; pass --class-names "
            "in class-index order so prompts can be built."
        )
    return unique_labels


def _resolve_compare_models_prompts(
    args: argparse.Namespace,
    class_names: list[str],
) -> list[str]:
    if args.class_prompts is not None:
        if len(args.class_prompts) != len(class_names):
            raise ValueError(
                f"--class-prompts count ({len(args.class_prompts)}) must match "
                f"the number of classes ({len(class_names)})."
            )
        return list(args.class_prompts)

    if "{}" not in args.prompt_template:
        raise ValueError(
            "--prompt-template must contain a '{}' slot for the class name, "
            "or pass explicit --class-prompts instead."
        )
    return [args.prompt_template.format(name) for name in class_names]


def _handle_compare_models(args: argparse.Namespace) -> int:
    from .data import (
        filter_records_by_split,
        load_patch_manifest,
        records_to_image_paths,
        records_to_labels,
    )

    try:
        records = load_patch_manifest(
            manifest_path=args.manifest,
            image_root=args.image_root,
            require_exists=True,
        )
        if args.split is not None:
            records = filter_records_by_split(records, args.split)
            if not records:
                raise ValueError(f"No manifest records matched split '{args.split}'.")
        if args.max_images is not None:
            records = records[: args.max_images]

        class_names = _resolve_compare_models_class_names(args, records)
        class_prompts = _resolve_compare_models_prompts(args, class_names)

        from .evaluation import resolve_true_indices

        labels = records_to_labels(records)
        true_indices = resolve_true_indices(labels, class_names)

        output_dir = Path(args.output_dir)
        csv_path = output_dir / "model_comparison.csv"
        per_class_csv_path = output_dir / "model_comparison_per_class.csv"
        chart_path = output_dir / "model_comparison.png"
        metadata_path = output_dir / "metadata.json"

        if args.dry_run:
            print("Dry run only. No model inference was run.")
            print(f"Manifest: {args.manifest}")
            print(f"Patches: {len(records)}")
            print(f"Models: {', '.join(args.models)}")
            print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
            print("Prompts:")
            for prompt in class_prompts:
                print(f"  - {prompt}")
            print(f"CSV output: {csv_path}")
            print(f"Per-class CSV output: {per_class_csv_path}")
            print(f"Chart output: {chart_path}")
            print(f"Metadata output: {metadata_path}")
            return 0

        from .data import load_patch_images_from_paths
        from .environment import collect_environment
        from .evaluation import evaluate_models_zero_shot
        from .visualization import (
            compute_model_accuracy_cis,
            save_model_comparison_chart,
            save_model_comparison_csv,
            save_model_comparison_per_class_csv,
        )

        image_paths = records_to_image_paths(records)
        images, _ = load_patch_images_from_paths(image_paths)

        results = evaluate_models_zero_shot(
            images,
            true_indices,
            class_prompts,
            args.models,
            device=args.device,
            batch_size=args.batch_size,
        )

        cis = compute_model_accuracy_cis(
            results,
            confidence=args.confidence,
            num_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

        random_baseline = 1.0 / len(class_names)
        subtitle = (
            f"{len(images)} patches · {len(class_names)} classes · frozen · "
            f"shared prompt template"
        )
        save_model_comparison_csv(results, csv_path, cis=cis)
        save_model_comparison_per_class_csv(results, class_names, per_class_csv_path)
        save_model_comparison_chart(
            results,
            chart_path,
            title=args.title,
            subtitle=subtitle,
            random_baseline=random_baseline,
            cis=cis,
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "manifest": args.manifest,
                    "num_images": len(images),
                    "models": list(args.models),
                    "class_names": class_names,
                    "class_prompts": class_prompts,
                    "prompt_template": (
                        None if args.class_prompts is not None else args.prompt_template
                    ),
                    "device": args.device,
                    "batch_size": args.batch_size,
                    "split": args.split,
                    "random_baseline": random_baseline,
                    "bootstrap": {
                        "confidence": args.confidence,
                        "num_resamples": args.bootstrap_resamples,
                        "seed": args.seed,
                    },
                    "environment": collect_environment(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "results": [
                        {
                            "model": result.model,
                            "accuracy": result.accuracy,
                            "accuracy_ci": cis[result_index],
                            "correct": result.correct,
                            "total": result.total,
                            "per_class": [
                                {
                                    "class_index": index,
                                    "class_name": class_names[index],
                                    "correct": result.per_class_correct[index],
                                    "total": result.per_class_total[index],
                                    "accuracy": (
                                        None
                                        if result.per_class_total[index] == 0
                                        else result.per_class_correct[index]
                                        / result.per_class_total[index]
                                    ),
                                }
                                for index in range(len(class_names))
                            ],
                        }
                        for result_index, result in enumerate(results)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved model comparison CSV to: {csv_path}")
    print(f"Saved per-class comparison CSV to: {per_class_csv_path}")
    print(f"Saved model comparison chart to: {chart_path}")
    print(f"Saved model comparison metadata to: {metadata_path}")
    print(f"Patches: {len(images)}")
    for result, ci in zip(results, cis):
        line = f"- {result.model}: {result.accuracy:.1%} ({result.correct}/{result.total})"
        if ci is not None:
            line += (
                f"  {ci['confidence']:.0%} CI [{ci['ci_low']:.1%}, {ci['ci_high']:.1%}]"
            )
        print(line)
    return 0


def _handle_linear_probe(args: argparse.Namespace) -> int:
    from .data import (
        filter_records_by_split,
        load_patch_manifest,
        records_to_image_paths,
        records_to_labels,
    )

    try:
        records = load_patch_manifest(
            manifest_path=args.manifest,
            image_root=args.image_root,
            require_exists=True,
        )
        train_records = filter_records_by_split(records, args.train_split)
        test_records = filter_records_by_split(records, args.test_split)
        if not train_records:
            raise ValueError(
                f"No manifest records matched train split '{args.train_split}'."
            )
        if not test_records:
            raise ValueError(
                f"No manifest records matched test split '{args.test_split}'."
            )
        if args.max_images is not None:
            train_records = train_records[: args.max_images]
            test_records = test_records[: args.max_images]

        train_labels = records_to_labels(train_records)
        if any(label is None or not str(label).strip() for label in train_labels):
            raise ValueError(
                "Every train record must be labeled to fit a linear probe."
            )
        test_labels = records_to_labels(test_records)

        output_dir = Path(args.output_dir)
        predictions_path = output_dir / "predictions.csv"
        errors_path = output_dir / "errors.csv"
        metrics_path = output_dir / "metrics.json"

        if args.dry_run:
            print("Dry run only. No model inference was run.")
            print(f"Manifest: {args.manifest}")
            print(f"Model: {args.model}")
            print(f"Train split '{args.train_split}': {len(train_records)} patches")
            print(f"Test split '{args.test_split}': {len(test_records)} patches")
            print(f"Predictions output: {predictions_path}")
            print(f"Metrics output: {metrics_path}")
            return 0

        from .data import load_patch_images_from_paths
        from .environment import collect_environment
        from .evaluation import (
            accuracy_ci_from_labels,
            compute_classification_report,
            run_linear_probe,
        )
        from .models import create_model
        from .visualization import (
            save_classification_metrics_json,
            save_zero_shot_errors_csv,
            save_zero_shot_predictions_csv,
        )

        model = create_model(args.model, args.device)
        train_images, _ = load_patch_images_from_paths(
            records_to_image_paths(train_records)
        )
        test_images, test_image_paths = load_patch_images_from_paths(
            records_to_image_paths(test_records)
        )

        train_embeddings = model.encode_images(train_images, batch_size=args.batch_size)
        test_embeddings = model.encode_images(test_images, batch_size=args.batch_size)

        probe = run_linear_probe(
            train_embeddings,
            train_labels,
            test_embeddings,
            class_names=args.class_names,
            C=args.C,
            max_iter=args.max_iter,
            seed=args.seed,
            normalize=not args.no_normalize,
        )

        predicted_labels = probe["predicted_labels"]
        results = [
            {
                "image_index": index,
                "predicted_label": predicted_labels[index],
                "predicted_index": probe["predicted_indices"][index],
                "confidence": probe["confidences"][index],
                "top_predictions": [],
            }
            for index in range(len(predicted_labels))
        ]

        labeled_pairs = [
            (str(true), predicted_labels[index])
            for index, true in enumerate(test_labels)
            if true is not None and str(true).strip()
        ]
        if not labeled_pairs:
            raise ValueError(
                "No labeled test records; cannot evaluate the linear probe."
            )
        labeled_true = [pair[0] for pair in labeled_pairs]
        labeled_pred = [pair[1] for pair in labeled_pairs]

        report = compute_classification_report(
            labeled_true,
            labeled_pred,
            class_names=args.class_names,
        )
        report["accuracy_ci"] = accuracy_ci_from_labels(
            test_labels,
            predicted_labels,
            confidence=args.confidence,
            num_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

        save_zero_shot_predictions_csv(
            test_image_paths, results, predictions_path, true_labels=test_labels
        )
        save_zero_shot_errors_csv(
            test_image_paths, results, errors_path, true_labels=test_labels
        )
        save_classification_metrics_json(
            report,
            metrics_path,
            metadata={
                "task": "linear-probe",
                "manifest": args.manifest,
                "model": args.model,
                "device": args.device,
                "train_split": args.train_split,
                "test_split": args.test_split,
                "num_train": probe["num_train"],
                "num_test": probe["num_test"],
                "embedding_dim": probe["embedding_dim"],
                "probe": {
                    "classifier": "logistic_regression",
                    "C": probe["C"],
                    "max_iter": probe["max_iter"],
                    "normalize": probe["normalize"],
                    "seed": probe["seed"],
                },
                "bootstrap": {
                    "confidence": args.confidence,
                    "num_resamples": args.bootstrap_resamples,
                    "seed": args.seed,
                },
                "environment": collect_environment(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    accuracy_ci = report["accuracy_ci"]
    print(f"Saved linear-probe predictions to: {predictions_path}")
    print(f"Saved linear-probe errors to: {errors_path}")
    print(f"Saved linear-probe metrics to: {metrics_path}")
    print(f"Train patches: {probe['num_train']} · Test patches: {probe['num_test']}")
    print(
        f"Accuracy: {report['accuracy']:.1%} "
        f"({accuracy_ci['confidence']:.0%} CI "
        f"[{accuracy_ci['ci_low']:.1%}, {accuracy_ci['ci_high']:.1%}])"
    )
    print(f"Balanced accuracy: {report['balanced_accuracy']:.1%}")
    print(f"Macro F1: {report['macro_f1']:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "version":
        return _handle_version()

    if args.command == "models":
        return _handle_models()

    if args.command == "demos":
        return _handle_demos()

    if args.command == "demo":
        return _handle_demo(args)

    if args.command == "convert-manifest":
        return _handle_convert_manifest(args)

    if args.command == "build-imagefolder-manifest":
        return _handle_build_imagefolder_manifest(args)

    if args.command == "sample-manifest":
        return _handle_sample_manifest(args)

    if args.command == "summarize-report":
        return _handle_summarize_report(args)

    if args.command == "compare-reports":
        return _handle_compare_reports(args)

    if args.command == "validate-config":
        return _handle_validate_config(args)

    if args.command == "run-zero-shot-grid":
        return _handle_run_zero_shot_grid(args)

    if args.command == "render-coordinate-heatmap":
        return _handle_render_coordinate_heatmap(args)

    if args.command == "score-coordinate-heatmap":
        return _handle_score_coordinate_heatmap(args)

    if args.command == "score-coordinate-heatmap-prompt-set":
        return _handle_score_coordinate_heatmap_prompt_set(args)

    if args.command == "compare-coordinate-heatmap-scores":
        return _handle_compare_coordinate_heatmap_scores(args)

    if args.command == "compare-models":
        return _handle_compare_models(args)

    if args.command == "linear-probe":
        return _handle_linear_probe(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
