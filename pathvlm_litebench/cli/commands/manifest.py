from __future__ import annotations

import argparse

from ...data.imagefolder import build_imagefolder_manifest
from ...data.manifest_converter import convert_manifest, convert_mhist_manifest
from ...data.manifest_sampler import sample_manifest, summarize_manifest


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
