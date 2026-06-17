from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ... import version
from ...config.heatmap_config import (
    PatchCoordinateHeatmapConfig,
    PatchCoordinateHeatmapPromptSetConfig,
    PatchCoordinateHeatmapScoringConfig,
)


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
    from ...config import load_patch_coordinate_heatmap_config

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
    from ...data import load_coordinate_patch_manifest
    from ...visualization import aggregate_patch_scores_to_grid, save_score_heatmap

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
    from ...config import load_patch_coordinate_heatmap_scoring_config

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
    from ...evaluation import score_patch_images_for_prompt
    from ...visualization import (
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
    from ...visualization import (
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
    from ...data import (
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

        from ...data import load_patch_images_from_paths
        from ...models import create_model

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
    from ...config import load_patch_coordinate_heatmap_prompt_set_config
    from ...data import load_coordinate_patch_manifest

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

        from ...data import (
            coordinate_records_to_image_paths,
            load_patch_images_from_paths,
        )
        from ...models import create_model

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
    from ...visualization import (
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
