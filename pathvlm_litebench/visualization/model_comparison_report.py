from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

if TYPE_CHECKING:
    from pathvlm_litebench.evaluation.model_comparison import ModelZeroShotResult


def save_model_comparison_csv(
    results: Sequence[ModelZeroShotResult],
    output_csv_path: str | Path,
) -> str:
    """
    Save per-model zero-shot accuracy as CSV.
    """
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "accuracy", "correct", "total"]
    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result.model,
                    "accuracy": result.accuracy,
                    "correct": result.correct,
                    "total": result.total,
                }
            )

    return str(output_csv_path)


def save_model_comparison_chart(
    results: Sequence[ModelZeroShotResult],
    output_path: str | Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
    random_baseline: float | None = None,
) -> str:
    """
    Save a bar chart of per-model zero-shot accuracy.
    """
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [result.model for result in results]
    accuracies = [result.accuracy for result in results]

    fig_width = max(5.0, min(14.0, 2.0 + 1.8 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.13)

    x = np.arange(len(labels))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(idx % 10) for idx in range(len(labels))]
    bars = ax.bar(x, accuracies, width=0.6, color=colors, edgecolor="white", linewidth=1.0)

    if random_baseline is not None:
        ax.axhline(random_baseline, color="#cc4444", linestyle="--", linewidth=1.1)
        ax.text(
            len(labels) - 0.5,
            random_baseline + 0.012,
            f"random baseline ({random_baseline:.0%})",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#cc4444",
        )

    for rect, acc in zip(bars, accuracies):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            acc + 0.015,
            f"{acc:.0%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Overall zero-shot accuracy", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#dddddd")
    ax.set_axisbelow(True)

    fig.suptitle(
        title or "Zero-shot tissue classification by model",
        fontsize=13,
        fontweight="bold",
        x=0.5,
        y=0.955,
    )
    if subtitle:
        fig.text(0.5, 0.885, subtitle, fontsize=9.5, color="#444444", ha="center")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return str(output_path)
