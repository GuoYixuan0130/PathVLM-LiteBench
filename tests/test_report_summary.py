import json
from pathlib import Path

import pytest

from pathvlm_litebench.visualization.report_summary import (
    build_zero_shot_experiment_summary,
    save_zero_shot_experiment_summary,
)


def _write_zero_shot_report(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "metadata": {
            "model": "clip",
            "device": "cpu",
            "split": "test",
            "manifest": "dataset/MHIST/manifest.csv",
            "image_dir": "dataset/MHIST/images",
            "class_names": ["HP", "SSA"],
            "top_k": 2,
            "num_images": 4,
        },
        "metrics": {
            "classification_report": {
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "macro_precision": 0.5,
                "macro_recall": 0.5,
                "macro_f1": 0.5,
                "per_class": {
                    "HP": {
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "support": 2,
                    },
                    "SSA": {
                        "precision": 0.5,
                        "recall": 0.5,
                        "f1": 0.5,
                        "support": 2,
                    },
                },
                "confusion_matrix": {
                    "class_names": ["HP", "SSA"],
                    "matrix": [[1, 1], [1, 1]],
                },
            },
            "error_summary": {
                "num_samples": 4,
                "labeled_count": 4,
                "unlabeled_count": 0,
                "num_errors": 2,
                "error_rate": 0.5,
                "true_label_distribution": {"HP": 2, "SSA": 2},
                "predicted_label_distribution": {"HP": 2, "SSA": 2},
                "warning": "Example warning.",
            },
        },
    }
    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_payload),
        encoding="utf-8",
    )
    (report_dir / "predictions.csv").write_text(
        "image_index,image_path,true_label,predicted_label,predicted_index,confidence,correct,top_predictions_json\n"
        "0,a.png,HP,HP,0,0.8,True,[]\n"
        "1,b.png,HP,SSA,1,0.7,False,[]\n"
        "2,c.png,SSA,HP,0,0.6,False,[]\n"
        "3,d.png,SSA,SSA,1,0.9,True,[]\n",
        encoding="utf-8",
    )
    (report_dir / "errors.csv").write_text(
        "image_index,image_path,true_label,predicted_label,predicted_index,confidence,top_predictions_json\n"
        "1,b.png,HP,SSA,1,0.7,[]\n"
        "2,c.png,SSA,HP,0,0.6,[]\n",
        encoding="utf-8",
    )


def test_build_zero_shot_experiment_summary(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)

    markdown = build_zero_shot_experiment_summary(report_dir)

    assert "# Zero-Shot Experiment Summary" in markdown
    assert "| Model | clip |" in markdown
    assert "| Accuracy | 0.5000 |" in markdown
    assert "| HP | 0.5000 | 0.5000 | 0.5000 | 2 |" in markdown
    assert "| True \\ Pred | HP | SSA |" in markdown
    assert "| predictions.csv | found (4 rows) |" in markdown
    assert "Example warning." in markdown
    assert "not clinical interpretation" in markdown


def test_save_zero_shot_experiment_summary_default_path(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)

    saved_path = save_zero_shot_experiment_summary(report_dir)

    assert saved_path == str(report_dir / "experiment_summary.md")
    assert Path(saved_path).exists()
    assert "Balanced accuracy" in Path(saved_path).read_text(encoding="utf-8")


def test_save_zero_shot_experiment_summary_custom_path(tmp_path: Path):
    report_dir = tmp_path / "zero_shot_report"
    _write_zero_shot_report(report_dir)
    output_path = tmp_path / "summaries" / "summary.md"

    saved_path = save_zero_shot_experiment_summary(
        report_dir=report_dir,
        output_path=output_path,
    )

    assert saved_path == str(output_path)
    assert output_path.exists()


def test_build_zero_shot_experiment_summary_requires_metrics(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_zero_shot_experiment_summary(tmp_path / "missing_report")
