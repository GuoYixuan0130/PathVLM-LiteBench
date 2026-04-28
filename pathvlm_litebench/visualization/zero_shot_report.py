from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def save_zero_shot_predictions_csv(
    image_paths: list[str],
    results: list[dict],
    output_csv_path: str | Path,
    true_labels: list[str | None] | None = None,
) -> str:
    """
    Save per-image zero-shot prediction results as CSV.

    Args:
        image_paths:
            Image paths corresponding to predictions.
        results:
            Zero-shot result dictionaries from zero_shot_predict.
        output_csv_path:
            Output CSV file path.
        true_labels:
            Optional ground-truth labels. When provided, length must match
            image_paths/results and `correct` will be filled.

    Returns:
        Saved CSV path as string.
    """
    if len(image_paths) != len(results):
        raise ValueError(
            f"image_paths and results must have the same length: "
            f"{len(image_paths)} vs {len(results)}"
        )

    if true_labels is not None and len(true_labels) != len(results):
        raise ValueError(
            f"true_labels and results must have the same length: "
            f"{len(true_labels)} vs {len(results)}"
        )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_index",
        "image_path",
        "true_label",
        "predicted_label",
        "predicted_index",
        "confidence",
        "correct",
        "top_predictions_json",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (image_path, result) in enumerate(zip(image_paths, results)):
            predicted_label = result.get("predicted_label", "")
            if true_labels is None:
                true_label: str | None = None
                correct: str | bool = ""
            else:
                true_label = true_labels[idx]
                if true_label is None:
                    correct = ""
                else:
                    correct = predicted_label == true_label

            writer.writerow(
                {
                    "image_index": result.get("image_index", idx),
                    "image_path": image_path,
                    "true_label": "" if true_label is None else true_label,
                    "predicted_label": predicted_label,
                    "predicted_index": result.get("predicted_index", ""),
                    "confidence": result.get("confidence", ""),
                    "correct": correct,
                    "top_predictions_json": json.dumps(
                        result.get("top_predictions", []),
                        ensure_ascii=False,
                    ),
                }
            )

    return str(output_csv_path)


def compute_zero_shot_error_summary(
    results: list[dict],
    true_labels: list[str | None] | None = None,
) -> dict[str, Any]:
    """
    Compute distribution and error-analysis summary for zero-shot results.
    """
    if true_labels is None:
        return {"note": "true_labels were not provided; error analysis was skipped."}

    if len(true_labels) != len(results):
        raise ValueError(
            f"true_labels and results must have the same length: "
            f"{len(true_labels)} vs {len(results)}"
        )

    true_label_distribution: dict[str, int] = {}
    predicted_label_distribution: dict[str, int] = {}
    per_class_correct: dict[str, int] = {}
    per_class_incorrect: dict[str, int] = {}

    num_errors = 0
    labeled_count = 0
    unlabeled_count = 0

    for result, true_label in zip(results, true_labels):
        predicted_label = str(result.get("predicted_label", ""))
        predicted_label_distribution[predicted_label] = (
            predicted_label_distribution.get(predicted_label, 0) + 1
        )

        if true_label is None:
            unlabeled_count += 1
            continue

        labeled_count += 1
        true_label_str = str(true_label)
        true_label_distribution[true_label_str] = true_label_distribution.get(true_label_str, 0) + 1

        if predicted_label == true_label_str:
            per_class_correct[true_label_str] = per_class_correct.get(true_label_str, 0) + 1
        else:
            per_class_incorrect[true_label_str] = per_class_incorrect.get(true_label_str, 0) + 1
            num_errors += 1

    for class_name in true_label_distribution:
        per_class_correct.setdefault(class_name, 0)
        per_class_incorrect.setdefault(class_name, 0)

    num_samples = len(results)
    error_rate = num_errors / labeled_count if labeled_count > 0 else 0.0

    summary: dict[str, Any] = {
        "num_samples": num_samples,
        "labeled_count": labeled_count,
        "unlabeled_count": unlabeled_count,
        "num_errors": num_errors,
        "error_rate": error_rate,
        "true_label_distribution": true_label_distribution,
        "predicted_label_distribution": predicted_label_distribution,
        "per_class_correct": per_class_correct,
        "per_class_incorrect": per_class_incorrect,
    }

    if num_samples > 0 and len(predicted_label_distribution) > 0:
        majority_label, majority_count = max(
            predicted_label_distribution.items(),
            key=lambda item: item[1],
        )
        if majority_count / num_samples > 0.8:
            summary["warning"] = (
                f"More than 80% of samples were predicted as '{majority_label}'. "
                "This may indicate prediction collapse or class bias."
            )

    return summary


def save_zero_shot_errors_csv(
    image_paths: list[str],
    results: list[dict],
    output_csv_path: str | Path,
    true_labels: list[str | None],
) -> str:
    """
    Save misclassified, labeled samples to CSV.
    """
    if len(image_paths) != len(results):
        raise ValueError(
            f"image_paths and results must have the same length: "
            f"{len(image_paths)} vs {len(results)}"
        )

    if len(true_labels) != len(results):
        raise ValueError(
            f"true_labels and results must have the same length: "
            f"{len(true_labels)} vs {len(results)}"
        )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_index",
        "image_path",
        "true_label",
        "predicted_label",
        "predicted_index",
        "confidence",
        "top_predictions_json",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (image_path, result, true_label) in enumerate(
            zip(image_paths, results, true_labels)
        ):
            if true_label is None:
                continue

            predicted_label = str(result.get("predicted_label", ""))
            if predicted_label == str(true_label):
                continue

            writer.writerow(
                {
                    "image_index": result.get("image_index", idx),
                    "image_path": image_path,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "predicted_index": result.get("predicted_index", ""),
                    "confidence": result.get("confidence", ""),
                    "top_predictions_json": json.dumps(
                        result.get("top_predictions", []),
                        ensure_ascii=False,
                    ),
                }
            )

    return str(output_csv_path)


def save_classification_metrics_json(
    metrics: dict[str, Any],
    output_json_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Save aggregate classification metrics to JSON.

    Args:
        metrics:
            Classification metrics dictionary.
        output_json_path:
            Output JSON file path.
        metadata:
            Optional metadata for the run.

    Returns:
        Saved JSON path as string.
    """
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "metrics": metrics,
    }

    output_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return str(output_json_path)
