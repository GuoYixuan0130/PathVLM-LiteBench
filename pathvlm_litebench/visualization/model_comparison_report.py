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


def compute_model_accuracy_cis(
    results: Sequence["ModelZeroShotResult"],
    *,
    confidence: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 0,
) -> list[dict | None]:
    """
    Bootstrap accuracy confidence intervals for each model result.

    Returns one interval dict per result, in input order, or ``None`` for any
    result that carries no per-sample ``correct_flags`` (so an interval cannot
    be estimated). See :func:`pathvlm_litebench.evaluation.bootstrap_proportion_ci`.
    """
    from pathvlm_litebench.evaluation.bootstrap import bootstrap_proportion_ci

    cis: list[dict | None] = []
    for result in results:
        if result.correct_flags:
            cis.append(
                bootstrap_proportion_ci(
                    result.correct_flags,
                    num_resamples=num_resamples,
                    confidence=confidence,
                    seed=seed,
                )
            )
        else:
            cis.append(None)
    return cis


def save_model_comparison_csv(
    results: Sequence[ModelZeroShotResult],
    output_csv_path: str | Path,
    *,
    cis: Sequence[dict | None] | None = None,
) -> str:
    """
    Save per-model zero-shot accuracy as CSV.

    When ``cis`` is provided (one entry per result), the ``ci_low``/``ci_high``
    columns hold the bootstrap accuracy interval bounds; they are left blank for
    results without an interval.
    """
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    if cis is not None and len(cis) != len(results):
        raise ValueError(
            f"cis and results must have the same length: {len(cis)} vs {len(results)}"
        )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "accuracy", "correct", "total", "ci_low", "ci_high"]
    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, result in enumerate(results):
            ci = cis[index] if cis is not None else None
            writer.writerow(
                {
                    "model": result.model,
                    "accuracy": result.accuracy,
                    "correct": result.correct,
                    "total": result.total,
                    "ci_low": "" if ci is None else ci["ci_low"],
                    "ci_high": "" if ci is None else ci["ci_high"],
                }
            )

    return str(output_csv_path)


def save_model_comparison_per_class_csv(
    results: Sequence["ModelZeroShotResult"],
    class_names: Sequence[str],
    output_csv_path: str | Path,
) -> str:
    """
    Save per-model, per-class zero-shot accuracy as a long-format CSV.

    One row per (model, class). Accuracy is left blank for classes that have no
    patches in the evaluated set.
    """
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    num_classes = len(class_names)
    for result in results:
        if len(result.per_class_total) != num_classes:
            raise ValueError(
                f"Model {result.model!r} has {len(result.per_class_total)} "
                f"per-class entries but {num_classes} class names were given."
            )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "class_index", "class_name", "correct", "total", "accuracy"]
    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for index, name in enumerate(class_names):
                total = result.per_class_total[index]
                correct = result.per_class_correct[index]
                writer.writerow(
                    {
                        "model": result.model,
                        "class_index": index,
                        "class_name": name,
                        "correct": correct,
                        "total": total,
                        "accuracy": "" if total == 0 else correct / total,
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
    cis: Sequence[dict | None] | None = None,
) -> str:
    """
    Save a bar chart of per-model zero-shot accuracy.

    When ``cis`` is provided (one entry per result), bootstrap confidence
    intervals are drawn as error bars on each bar.
    """
    if len(results) == 0:
        raise ValueError("results must not be empty.")

    if cis is not None and len(cis) != len(results):
        raise ValueError(
            f"cis and results must have the same length: {len(cis)} vs {len(results)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [result.model for result in results]
    accuracies = [result.accuracy for result in results]

    yerr = None
    if cis is not None and any(ci is not None for ci in cis):
        lower = [
            (result.accuracy - ci["ci_low"]) if ci is not None else 0.0
            for result, ci in zip(results, cis)
        ]
        upper = [
            (ci["ci_high"] - result.accuracy) if ci is not None else 0.0
            for result, ci in zip(results, cis)
        ]
        yerr = np.array([lower, upper])

    fig_width = max(5.0, min(14.0, 2.0 + 1.8 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.13)

    x = np.arange(len(labels))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(idx % 10) for idx in range(len(labels))]
    bars = ax.bar(
        x,
        accuracies,
        width=0.6,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        yerr=yerr,
        capsize=5 if yerr is not None else 0,
        error_kw={"ecolor": "#333333", "elinewidth": 1.2},
    )

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

    for index, (rect, acc) in enumerate(zip(bars, accuracies)):
        upper_offset = float(yerr[1][index]) if yerr is not None else 0.0
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            acc + upper_offset + 0.015,
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
