from __future__ import annotations

import argparse

from ...visualization.report_summary import (
    save_experiment_comparison_summary,
    save_prompt_sensitivity_experiment_summary,
    save_retrieval_experiment_summary,
    save_zero_shot_experiment_summary,
)


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
