from __future__ import annotations

import argparse

from . import version
from .data import convert_manifest, convert_mhist_manifest
from .models import list_available_models


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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
