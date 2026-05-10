from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

from . import version
from .config import load_benchmark_config
from .config.heatmap_config import (
    PatchCoordinateHeatmapConfig,
    PatchCoordinateHeatmapScoringConfig,
)
from .data.manifest_converter import convert_manifest, convert_mhist_manifest
from .data.manifest_sampler import sample_manifest, summarize_manifest
from .models.registry import list_available_models
from .visualization.report_summary import (
    save_experiment_comparison_summary,
    save_prompt_sensitivity_experiment_summary,
    save_retrieval_experiment_summary,
    save_zero_shot_experiment_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pathvlm-litebench",
        description="PathVLM-LiteBench lightweight command-line interface.",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="Show toolkit version.")
    subparsers.add_parser("models", help="List registered models.")
    subparsers.add_parser("demos", help="Show available demo commands.")
    convert_manifest_parser = subparsers.add_parser(
        "convert-manifest",
        help="Convert dataset-specific annotations CSV to standard patch manifest CSV.",
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
        help="Directory for generated score CSV and heatmap PNG.",
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
    print("pathvlm-litebench render-coordinate-heatmap --config configs/patch_coordinate_heatmap_demo_config.json")
    print("pathvlm-litebench score-coordinate-heatmap --config configs/patch_coordinate_heatmap_scoring_demo_config.json")
    print("pathvlm-litebench render-coordinate-heatmap --manifest dataset/patch_coordinates/coordinate_manifest.csv --score-csv outputs/patch_coordinate_heatmap_demo/scores.csv --output outputs/patch_coordinate_heatmap_demo/heatmap.png")
    print("pathvlm-litebench score-coordinate-heatmap --manifest dataset/patch_coordinates/coordinate_manifest.csv --prompt \"a histopathology image of tumor tissue\" --output-dir outputs/patch_coordinate_heatmap_scored --model clip")
    return 0


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
            print("Config valid: patch_coordinate_heatmap_scoring")
            print(f"Manifest: {config.manifest}")
            print(f"Prompt: {config.prompt}")
            print(f"Model: {config.model}")
            print(f"Device: {config.device}")
            print(f"Output dir: {config.output_dir}")
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


def _handle_score_coordinate_heatmap(args: argparse.Namespace) -> int:
    from .data import (
        coordinate_records_to_image_paths,
        load_coordinate_patch_manifest,
        load_patch_images_from_paths,
    )
    from .evaluation import score_patch_images_for_prompt
    from .models import create_model
    from .visualization import (
        aggregate_patch_scores_to_grid,
        save_patch_scores_csv,
        save_score_heatmap,
    )

    try:
        config = _load_score_heatmap_args_config(args)

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

        image_paths = coordinate_records_to_image_paths(records)
        images, _ = load_patch_images_from_paths(image_paths)

        model = create_model(config.model, device=config.device)
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
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved patch-coordinate scores to: {saved_scores}")
    print(f"Saved patch-coordinate heatmap to: {saved_heatmap}")
    print(f"Patches: {len(records)}")
    print(f"Prompt: {config.prompt}")
    print(f"Model: {config.model}")
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

    if args.command == "convert-manifest":
        return _handle_convert_manifest(args)

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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
