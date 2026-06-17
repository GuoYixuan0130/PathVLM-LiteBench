from __future__ import annotations

import argparse
import json
from dataclasses import replace

from ...config import load_benchmark_config
from .heatmap import (
    _resolve_prompt_set_comparison_output_paths,
    _resolve_score_heatmap_output_paths,
)


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
            from ...evaluation.zero_shot_grid import (
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
            from ...config import load_patch_coordinate_heatmap_config

            config = load_patch_coordinate_heatmap_config(args.config)
            print("Config valid: patch_coordinate_heatmap")
            print(f"Manifest: {config.manifest}")
            print(f"Score CSV: {config.score_csv}")
            print(f"Output: {config.output}")
            print(f"Align by: {config.align_by}")
            print(f"Score column: {config.score_column}")
            return 0

        if task == "patch_coordinate_heatmap_scoring":
            from ...config import load_patch_coordinate_heatmap_scoring_config

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
            from ...config import load_patch_coordinate_heatmap_prompt_set_config

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
    from ...evaluation.zero_shot_grid import (
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
