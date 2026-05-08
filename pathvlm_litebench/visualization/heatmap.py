from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

from pathvlm_litebench.data.coordinate_manifest import CoordinatePatchRecord

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class ScoreHeatmapGrid:
    values: np.ndarray
    counts: np.ndarray
    x_values: list[float]
    y_values: list[float]


def _validate_scores(scores: Sequence[float], expected_length: int) -> list[float]:
    if len(scores) != expected_length:
        raise ValueError(
            f"records and scores must have the same length: "
            f"{expected_length} vs {len(scores)}"
        )

    parsed_scores: list[float] = []
    for idx, score in enumerate(scores):
        parsed = float(score)
        if not math.isfinite(parsed):
            raise ValueError(f"Score at index {idx} is not finite: {score!r}")
        parsed_scores.append(parsed)

    return parsed_scores


def aggregate_patch_scores_to_grid(
    records: Sequence[CoordinatePatchRecord],
    scores: Sequence[float],
    fill_value: float = np.nan,
) -> ScoreHeatmapGrid:
    """
    Aggregate patch scores into a coordinate grid.

    Repeated coordinates are averaged. Rows follow sorted y coordinates and
    columns follow sorted x coordinates, matching image-style heatmap display.
    """
    if len(records) == 0:
        raise ValueError("records must contain at least one item")

    parsed_scores = _validate_scores(scores, len(records))

    x_values = sorted({float(record.x) for record in records})
    y_values = sorted({float(record.y) for record in records})

    x_to_col = {value: idx for idx, value in enumerate(x_values)}
    y_to_row = {value: idx for idx, value in enumerate(y_values)}

    sums = np.zeros((len(y_values), len(x_values)), dtype=float)
    counts = np.zeros((len(y_values), len(x_values)), dtype=int)

    for record, score in zip(records, parsed_scores):
        row = y_to_row[float(record.y)]
        col = x_to_col[float(record.x)]
        sums[row, col] += score
        counts[row, col] += 1

    values = np.full((len(y_values), len(x_values)), fill_value, dtype=float)
    np.divide(sums, counts, out=values, where=counts > 0)

    return ScoreHeatmapGrid(
        values=values,
        counts=counts,
        x_values=x_values,
        y_values=y_values,
    )


def save_score_heatmap(
    grid: ScoreHeatmapGrid,
    output_path: str | Path,
    title: str | None = None,
    cmap: str = "viridis",
) -> str:
    """
    Save an aggregated score heatmap as a PNG image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    masked_values = np.ma.masked_invalid(grid.values)

    fig_width = max(4.0, min(12.0, 0.6 * len(grid.x_values) + 2.0))
    fig_height = max(3.0, min(10.0, 0.6 * len(grid.y_values) + 2.0))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(masked_values, cmap=cmap, origin="upper", aspect="auto")

    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_xticks(range(len(grid.x_values)))
    ax.set_yticks(range(len(grid.y_values)))
    ax.set_xticklabels([_format_coordinate(value) for value in grid.x_values])
    ax.set_yticklabels([_format_coordinate(value) for value in grid.y_values])

    if len(grid.x_values) > 8:
        ax.tick_params(axis="x", labelrotation=45)

    if title:
        ax.set_title(title)

    fig.colorbar(image, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return str(output_path)


def save_patch_scores_csv(
    records: Sequence[CoordinatePatchRecord],
    scores: Sequence[float],
    output_csv_path: str | Path,
    prompt: str | None = None,
) -> str:
    """
    Save coordinate-aware patch scores as CSV for later inspection.
    """
    parsed_scores = _validate_scores(scores, len(records))

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "slide_id",
        "x",
        "y",
        "width",
        "height",
        "label",
        "score",
        "prompt",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for record, score in zip(records, parsed_scores):
            writer.writerow(
                {
                    "image_path": record.image_path,
                    "slide_id": "" if record.slide_id is None else record.slide_id,
                    "x": record.x,
                    "y": record.y,
                    "width": "" if record.width is None else record.width,
                    "height": "" if record.height is None else record.height,
                    "label": "" if record.label is None else record.label,
                    "score": score,
                    "prompt": "" if prompt is None else prompt,
                }
            )

    return str(output_csv_path)


def _format_coordinate(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"
