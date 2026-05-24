from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class PatchScoreRunSummary:
    run_name: str
    score_csv: str
    metadata_json: str | None
    prompt: str | None
    model: str | None
    device: str | None
    manifest: str | None
    heatmap_output: str | None
    version: str | None
    created_at_utc: str | None
    row_count: int
    metadata_patch_count: int | None
    score_mean: float
    score_std: float
    score_min: float
    score_max: float


def summarize_patch_score_csv(
    score_csv: str | Path,
    *,
    score_column: str = "score",
    metadata_json: str | Path | None = None,
    run_name: str | None = None,
    run_index: int = 0,
) -> PatchScoreRunSummary:
    """
    Summarize one saved prompt-scored patch-coordinate score CSV.

    If metadata_json is omitted and a sibling metadata.json file exists next to
    the score CSV, it is loaded automatically.
    """
    score_csv_path = Path(score_csv)
    if not score_csv_path.exists():
        raise FileNotFoundError(f"Score CSV file not found: {score_csv_path}")

    rows, fieldnames = _read_score_rows(score_csv_path)
    if score_column not in fieldnames:
        raise ValueError(
            f"Score column '{score_column}' not found in score CSV. "
            f"Available columns: {', '.join(fieldnames)}"
        )
    if not rows:
        raise ValueError(f"Score CSV contains no rows: {score_csv_path}")

    scores = _parse_scores(rows, score_column, score_csv_path)
    metadata_path = _resolve_metadata_path(score_csv_path, metadata_json)
    metadata = _load_metadata(metadata_path) if metadata_path is not None else {}
    metadata_patch_count = _metadata_patch_count(metadata, metadata_path)
    if metadata_patch_count is not None and metadata_patch_count != len(rows):
        raise ValueError(
            "Metadata patch_count does not match score CSV row count for "
            f"{score_csv_path}: {metadata_patch_count} vs {len(rows)}"
        )

    return PatchScoreRunSummary(
        run_name=_resolve_run_name(score_csv_path, run_index, run_name),
        score_csv=str(score_csv_path),
        metadata_json=str(metadata_path) if metadata_path is not None else None,
        prompt=_metadata_text(metadata, "prompt") or _first_non_empty(rows, "prompt"),
        model=_metadata_text(metadata, "model"),
        device=_metadata_text(metadata, "device"),
        manifest=_metadata_text(metadata, "manifest"),
        heatmap_output=_metadata_text(metadata, "heatmap_output"),
        version=_metadata_text(metadata, "version"),
        created_at_utc=_metadata_text(metadata, "created_at_utc"),
        row_count=len(rows),
        metadata_patch_count=metadata_patch_count,
        score_mean=sum(scores) / len(scores),
        score_std=_population_std(scores),
        score_min=min(scores),
        score_max=max(scores),
    )


def compare_patch_score_csvs(
    score_csvs: Sequence[str | Path],
    *,
    score_column: str = "score",
    metadata_jsons: Sequence[str | Path] | None = None,
    run_names: Sequence[str] | None = None,
    require_equal_row_count: bool = True,
) -> list[PatchScoreRunSummary]:
    """
    Summarize multiple saved prompt-scored patch-coordinate score CSV artifacts.
    """
    if not score_csvs:
        raise ValueError("At least one score CSV is required.")
    if metadata_jsons is not None and len(metadata_jsons) != len(score_csvs):
        raise ValueError(
            "metadata_jsons must contain exactly one path per score CSV."
        )
    if run_names is not None and len(run_names) != len(score_csvs):
        raise ValueError("run_names must contain exactly one label per score CSV.")

    summaries: list[PatchScoreRunSummary] = []
    for index, score_csv in enumerate(score_csvs):
        metadata_json = metadata_jsons[index] if metadata_jsons is not None else None
        run_name = run_names[index] if run_names is not None else None
        summaries.append(
            summarize_patch_score_csv(
                score_csv,
                score_column=score_column,
                metadata_json=metadata_json,
                run_name=run_name,
                run_index=index,
            )
        )

    if require_equal_row_count and len(summaries) > 1:
        expected_rows = summaries[0].row_count
        mismatched = [
            f"{summary.run_name}={summary.row_count}"
            for summary in summaries
            if summary.row_count != expected_rows
        ]
        if mismatched:
            raise ValueError(
                "All score CSVs must contain the same row count. "
                f"Expected {expected_rows}; mismatched runs: {', '.join(mismatched)}"
            )

    return summaries


def save_patch_score_comparison_csv(
    summaries: Sequence[PatchScoreRunSummary],
    output_csv: str | Path,
) -> str:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_summary_fieldnames())
        writer.writeheader()
        for summary in summaries:
            writer.writerow(_summary_csv_row(summary))

    return str(output_csv)


def build_patch_score_comparison_summary(
    summaries: Sequence[PatchScoreRunSummary],
) -> str:
    if not summaries:
        raise ValueError("At least one run summary is required.")

    lines: list[str] = [
        "# Patch-Coordinate Score Comparison",
        "",
        "This summary compares saved PathVLM-LiteBench patch-coordinate score artifacts.",
        "It is artifact-only: it reads saved CSV and metadata files and does not load models or images.",
        "",
        "## Score Summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "Run",
                "Rows",
                "Mean",
                "Std",
                "Min",
                "Max",
                "Prompt",
                "Model",
                "Device",
            ],
            [
                [
                    summary.run_name,
                    summary.row_count,
                    _format_float(summary.score_mean),
                    _format_float(summary.score_std),
                    _format_float(summary.score_min),
                    _format_float(summary.score_max),
                    summary.prompt,
                    summary.model,
                    summary.device,
                ]
                for summary in summaries
            ],
        )
    )
    lines.extend(["", "## Artifacts", ""])
    lines.extend(
        _markdown_table(
            [
                "Run",
                "Score CSV",
                "Metadata JSON",
                "Manifest",
                "Heatmap",
            ],
            [
                [
                    summary.run_name,
                    summary.score_csv,
                    summary.metadata_json,
                    summary.manifest,
                    summary.heatmap_output,
                ]
                for summary in summaries
            ],
        )
    )
    lines.extend(
        [
            "",
            "Use these aggregate score statistics for prompt-run review only. They are not clinical interpretations.",
            "",
        ]
    )
    return "\n".join(lines)


def save_patch_score_comparison_summary(
    summaries: Sequence[PatchScoreRunSummary],
    output_md: str | Path,
) -> str:
    output_md = Path(output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        build_patch_score_comparison_summary(summaries),
        encoding="utf-8",
    )
    return str(output_md)


def _read_score_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Score CSV is empty: {path}")
        return list(reader), list(reader.fieldnames)


def _parse_scores(
    rows: Sequence[dict[str, str]],
    score_column: str,
    path: Path,
) -> list[float]:
    scores: list[float] = []
    for row_idx, row in enumerate(rows, start=2):
        raw_score = row.get(score_column)
        if raw_score is None or not raw_score.strip():
            raise ValueError(
                f"Empty score column '{score_column}' at row {row_idx} in {path}"
            )
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric score at row {row_idx} in {path}: {raw_score!r}"
            ) from exc
        if not math.isfinite(score):
            raise ValueError(f"Score at row {row_idx} in {path} is not finite.")
        scores.append(score)
    return scores


def _resolve_metadata_path(
    score_csv: Path,
    metadata_json: str | Path | None,
) -> Path | None:
    if metadata_json is not None:
        metadata_path = Path(metadata_json)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata JSON file not found: {metadata_path}")
        return metadata_path

    sibling = score_csv.parent / "metadata.json"
    if sibling.exists():
        return sibling
    return None


def _load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata JSON must be an object: {path}")
    return data


def _metadata_patch_count(metadata: dict[str, Any], path: Path | None) -> int | None:
    value = metadata.get("patch_count")
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        location = "" if path is None else f" in {path}"
        raise ValueError(f"metadata patch_count must be an integer{location}.") from exc


def _metadata_text(metadata: dict[str, Any], field: str) -> str | None:
    value = metadata.get(field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _first_non_empty(rows: Sequence[dict[str, str]], field: str) -> str | None:
    for row in rows:
        value = row.get(field)
        if value is not None and value.strip():
            return value.strip()
    return None


def _resolve_run_name(path: Path, run_index: int, run_name: str | None) -> str:
    if run_name is not None and run_name.strip():
        return run_name.strip()
    if path.parent.name:
        return path.parent.name
    if path.stem:
        return path.stem
    return f"run_{run_index + 1}"


def _population_std(scores: Sequence[float]) -> float:
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return math.sqrt(variance)


def _summary_fieldnames() -> list[str]:
    return [
        "run_name",
        "score_csv",
        "metadata_json",
        "prompt",
        "model",
        "device",
        "manifest",
        "heatmap_output",
        "version",
        "created_at_utc",
        "row_count",
        "metadata_patch_count",
        "score_mean",
        "score_std",
        "score_min",
        "score_max",
    ]


def _summary_csv_row(summary: PatchScoreRunSummary) -> dict[str, object]:
    return {
        "run_name": summary.run_name,
        "score_csv": summary.score_csv,
        "metadata_json": summary.metadata_json or "",
        "prompt": summary.prompt or "",
        "model": summary.model or "",
        "device": summary.device or "",
        "manifest": summary.manifest or "",
        "heatmap_output": summary.heatmap_output or "",
        "version": summary.version or "",
        "created_at_utc": summary.created_at_utc or "",
        "row_count": summary.row_count,
        "metadata_patch_count": (
            "" if summary.metadata_patch_count is None else summary.metadata_patch_count
        ),
        "score_mean": _format_float(summary.score_mean),
        "score_std": _format_float(summary.score_std),
        "score_min": _format_float(summary.score_min),
        "score_max": _format_float(summary.score_max),
    }


def _format_float(value: float) -> str:
    return f"{value:.6g}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(_escape_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return lines


def _escape_markdown_cell(value: Any) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)
    return text.replace("\\", "/").replace("|", "\\|").replace("\n", " ")
