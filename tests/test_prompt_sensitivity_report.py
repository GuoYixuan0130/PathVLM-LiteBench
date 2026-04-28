import csv
import json
from pathlib import Path

from pathvlm_litebench.visualization import (
    save_prompt_sensitivity_summary_csv,
    save_prompt_sensitivity_details_csv,
    save_prompt_sensitivity_metrics_json,
)


def test_save_prompt_sensitivity_summary_csv(tmp_path: Path):
    results = [
        {
            "concept_name": "tumor",
            "num_prompts": 2,
            "mean_topk_overlap": 0.5,
            "mean_similarity_std": 0.03,
            "prompt_results": [],
        }
    ]

    output_csv_path = tmp_path / "reports" / "prompt_sensitivity_summary.csv"
    saved_path = save_prompt_sensitivity_summary_csv(
        results=results,
        output_csv_path=output_csv_path,
    )

    assert Path(saved_path).exists()
    content = Path(saved_path).read_text(encoding="utf-8")
    assert "concept_name" in content
    assert "tumor" in content
    assert "mean_topk_overlap" in content


def test_save_prompt_sensitivity_details_csv(tmp_path: Path):
    results = [
        {
            "concept_name": "tumor",
            "num_prompts": 2,
            "mean_topk_overlap": 0.5,
            "mean_similarity_std": 0.03,
            "prompt_results": [
                {
                    "prompt_index": 0,
                    "prompt_text": "tumor tissue",
                    "top_indices": [3, 5],
                    "top_scores": [0.9, 0.8],
                }
            ],
        }
    ]

    output_csv_path = tmp_path / "reports" / "prompt_sensitivity_details.csv"
    saved_path = save_prompt_sensitivity_details_csv(
        results=results,
        output_csv_path=output_csv_path,
    )

    assert Path(saved_path).exists()

    with Path(saved_path).open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["concept_name"] == "tumor"
    assert rows[0]["rank"] == "1"
    assert rows[0]["image_index"] == "3"
    assert float(rows[0]["score"]) == 0.9


def test_save_prompt_sensitivity_metrics_json(tmp_path: Path):
    results = [
        {
            "concept_name": "tumor",
            "num_prompts": 2,
            "mean_topk_overlap": 0.5,
            "mean_similarity_std": 0.03,
            "prompt_results": [],
        }
    ]
    metadata = {"model": "clip"}
    output_json_path = tmp_path / "reports" / "prompt_sensitivity_metrics.json"

    saved_path = save_prompt_sensitivity_metrics_json(
        results=results,
        output_json_path=output_json_path,
        metadata=metadata,
    )

    payload = json.loads(Path(saved_path).read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "clip"
    assert payload["results"][0]["concept_name"] == "tumor"
