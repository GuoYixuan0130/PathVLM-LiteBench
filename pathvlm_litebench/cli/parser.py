from __future__ import annotations

import argparse

from .constants import DEMO_SCRIPTS


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
