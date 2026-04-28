import csv
import json
from pathlib import Path

import pytest

from pathvlm_litebench.visualization import (
    save_retrieval_results_csv,
    save_retrieval_metrics_json,
)


def test_save_retrieval_results_csv_basic(tmp_path: Path):
    prompts = ["prompt tumor", "prompt normal"]
    retrieval_results = [
        [
            {
                "index": 0,
                "score": 0.9,
                "path": "a.png",
                "label": "tumor",
                "target_label": "tumor",
                "is_positive": True,
            },
            {
                "index": 1,
                "score": 0.7,
                "path": "b.png",
                "label": "normal",
                "target_label": "tumor",
                "is_positive": False,
            },
        ],
        [
            {
                "index": 1,
                "score": 0.8,
                "path": "b.png",
                "label": "normal",
                "target_label": "normal",
                "is_positive": True,
            },
        ],
    ]
    label_prompts = ["tumor", "normal"]

    output_csv_path = tmp_path / "reports" / "retrieval_results.csv"
    saved_path = save_retrieval_results_csv(
        prompts=prompts,
        retrieval_results=retrieval_results,
        output_csv_path=output_csv_path,
        label_prompts=label_prompts,
    )

    assert Path(saved_path).exists()

    with Path(saved_path).open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 3
    assert rows[0]["prompt_index"] == "0"
    assert rows[0]["rank"] == "1"
    assert rows[0]["is_positive"] in {"True", "true", "1"}
    assert "label" in rows[0]
    assert "target_label" in rows[0]
    assert "score" in rows[0]


def test_save_retrieval_results_csv_length_mismatch(tmp_path: Path):
    prompts = ["prompt tumor"]
    retrieval_results = [[], []]

    with pytest.raises(ValueError):
        save_retrieval_results_csv(
            prompts=prompts,
            retrieval_results=retrieval_results,
            output_csv_path=tmp_path / "retrieval_results.csv",
        )


def test_save_retrieval_results_csv_label_prompt_mismatch(tmp_path: Path):
    prompts = ["prompt tumor", "prompt normal"]
    retrieval_results = [[], []]
    label_prompts = ["tumor"]

    with pytest.raises(ValueError):
        save_retrieval_results_csv(
            prompts=prompts,
            retrieval_results=retrieval_results,
            output_csv_path=tmp_path / "retrieval_results.csv",
            label_prompts=label_prompts,
        )


def test_save_retrieval_metrics_json(tmp_path: Path):
    metrics = {
        "recall_at_k": {"R@1": 0.5, "R@5": 1.0},
        "mean_recall": 0.75,
    }
    metadata = {"model": "clip"}

    output_json_path = tmp_path / "reports" / "retrieval_metrics.json"
    saved_path = save_retrieval_metrics_json(
        metrics=metrics,
        output_json_path=output_json_path,
        metadata=metadata,
    )

    payload = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "clip"
    assert payload["metrics"]["mean_recall"] == 0.75
