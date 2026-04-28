import csv
import json
from pathlib import Path

import pytest

from pathvlm_litebench.visualization import (
    save_zero_shot_predictions_csv,
    save_classification_metrics_json,
)
from pathvlm_litebench.visualization.zero_shot_report import (
    compute_zero_shot_error_summary,
    save_zero_shot_errors_csv,
)


def test_save_zero_shot_predictions_csv_basic(tmp_path: Path):
    image_paths = ["a.png", "b.png"]
    results = [
        {
            "image_index": 0,
            "predicted_index": 0,
            "predicted_label": "HP",
            "confidence": 0.8,
            "top_predictions": [
                {"class_index": 0, "class_name": "HP", "probability": 0.8, "logit": 1.2}
            ],
        },
        {
            "image_index": 1,
            "predicted_index": 1,
            "predicted_label": "SSA",
            "confidence": 0.7,
            "top_predictions": [
                {"class_index": 1, "class_name": "SSA", "probability": 0.7, "logit": 1.0}
            ],
        },
    ]
    true_labels = ["HP", "HP"]

    output_csv_path = tmp_path / "reports" / "predictions.csv"
    saved_path = save_zero_shot_predictions_csv(
        image_paths=image_paths,
        results=results,
        output_csv_path=output_csv_path,
        true_labels=true_labels,
    )

    assert Path(saved_path).exists()

    with Path(saved_path).open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert "image_path" in rows[0]
    assert "predicted_label" in rows[0]
    assert "correct" in rows[0]
    assert rows[0]["correct"] == "True"
    assert rows[1]["correct"] == "False"


def test_save_zero_shot_predictions_csv_length_mismatch(tmp_path: Path):
    image_paths = ["a.png", "b.png"]
    results = [
        {
            "image_index": 0,
            "predicted_index": 0,
            "predicted_label": "HP",
            "confidence": 0.8,
            "top_predictions": [],
        },
        {
            "image_index": 1,
            "predicted_index": 1,
            "predicted_label": "SSA",
            "confidence": 0.7,
            "top_predictions": [],
        },
    ]
    true_labels = ["HP"]

    with pytest.raises(ValueError):
        save_zero_shot_predictions_csv(
            image_paths=image_paths,
            results=results,
            output_csv_path=tmp_path / "predictions.csv",
            true_labels=true_labels,
        )


def test_save_classification_metrics_json(tmp_path: Path):
    metrics = {"accuracy": 0.5}
    metadata = {"model": "clip"}
    output_json_path = tmp_path / "reports" / "metrics.json"

    saved_path = save_classification_metrics_json(
        metrics=metrics,
        output_json_path=output_json_path,
        metadata=metadata,
    )

    payload = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "clip"
    assert payload["metrics"]["accuracy"] == 0.5


def test_compute_zero_shot_error_summary_basic():
    results = [
        {"predicted_label": "HP"},
        {"predicted_label": "HP"},
        {"predicted_label": "SSA"},
    ]
    true_labels = ["HP", "SSA", "SSA"]

    summary = compute_zero_shot_error_summary(
        results=results,
        true_labels=true_labels,
    )

    assert summary["num_samples"] == 3
    assert summary["num_errors"] == 1
    assert summary["true_label_distribution"]["HP"] == 1
    assert summary["true_label_distribution"]["SSA"] == 2
    assert summary["predicted_label_distribution"]["HP"] == 2
    assert summary["predicted_label_distribution"]["SSA"] == 1
    assert summary["per_class_correct"]["HP"] == 1
    assert summary["per_class_correct"]["SSA"] == 1
    assert summary["per_class_incorrect"]["SSA"] == 1


def test_compute_zero_shot_error_summary_warning():
    results = [{"predicted_label": "HP"} for _ in range(9)] + [{"predicted_label": "SSA"}]
    true_labels = ["HP"] * 10

    summary = compute_zero_shot_error_summary(
        results=results,
        true_labels=true_labels,
    )

    assert "warning" in summary


def test_save_zero_shot_errors_csv(tmp_path: Path):
    image_paths = ["a.png", "b.png"]
    results = [
        {
            "image_index": 0,
            "predicted_index": 0,
            "predicted_label": "HP",
            "confidence": 0.8,
            "top_predictions": [{"class_name": "HP", "probability": 0.8}],
        },
        {
            "image_index": 1,
            "predicted_index": 1,
            "predicted_label": "SSA",
            "confidence": 0.7,
            "top_predictions": [{"class_name": "SSA", "probability": 0.7}],
        },
    ]
    true_labels = ["HP", "HP"]

    output_csv_path = tmp_path / "reports" / "errors.csv"
    saved_path = save_zero_shot_errors_csv(
        image_paths=image_paths,
        results=results,
        output_csv_path=output_csv_path,
        true_labels=true_labels,
    )

    assert Path(saved_path).exists()
    with Path(saved_path).open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["true_label"] == "HP"
    assert rows[0]["predicted_label"] == "SSA"


def test_save_zero_shot_errors_csv_length_mismatch(tmp_path: Path):
    image_paths = ["a.png"]
    results = [{"predicted_label": "HP"}]
    true_labels = ["HP", "SSA"]

    with pytest.raises(ValueError):
        save_zero_shot_errors_csv(
            image_paths=image_paths,
            results=results,
            output_csv_path=tmp_path / "errors.csv",
            true_labels=true_labels,
        )
