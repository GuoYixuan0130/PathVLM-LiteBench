from __future__ import annotations

import argparse

from . import version
from .data.manifest_converter import convert_manifest, convert_mhist_manifest
from .data.manifest_sampler import sample_manifest, summarize_manifest
from .models.registry import list_available_models
from .visualization.report_summary import (
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
