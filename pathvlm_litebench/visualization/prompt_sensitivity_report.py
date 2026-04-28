from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def save_prompt_sensitivity_summary_csv(
    results: list[dict],
    output_csv_path: str | Path,
) -> str:
    """
    Save concept-level prompt sensitivity summary metrics to CSV.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "concept_name",
        "num_prompts",
        "mean_topk_overlap",
        "mean_similarity_std",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in results:
            writer.writerow(
                {
                    "concept_name": item.get("concept_name", ""),
                    "num_prompts": item.get("num_prompts", ""),
                    "mean_topk_overlap": item.get("mean_topk_overlap", ""),
                    "mean_similarity_std": item.get("mean_similarity_std", ""),
                }
            )

    return str(output_csv_path)


def save_prompt_sensitivity_details_csv(
    results: list[dict],
    output_csv_path: str | Path,
) -> str:
    """
    Save prompt-level top-k retrieval details to CSV.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "concept_name",
        "prompt_index",
        "prompt_text",
        "rank",
        "image_index",
        "score",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for concept_result in results:
            concept_name = concept_result.get("concept_name", "")
            prompt_results = concept_result.get("prompt_results", [])

            for prompt_result in prompt_results:
                prompt_index = prompt_result.get("prompt_index", "")
                prompt_text = prompt_result.get("prompt_text", "")
                top_indices = prompt_result.get("top_indices", []) or []
                top_scores = prompt_result.get("top_scores", []) or []

                max_len = max(len(top_indices), len(top_scores))

                for rank_idx in range(max_len):
                    image_index = top_indices[rank_idx] if rank_idx < len(top_indices) else ""
                    score = top_scores[rank_idx] if rank_idx < len(top_scores) else ""

                    writer.writerow(
                        {
                            "concept_name": concept_name,
                            "prompt_index": prompt_index,
                            "prompt_text": prompt_text,
                            "rank": rank_idx + 1,
                            "image_index": image_index,
                            "score": score,
                        }
                    )

    return str(output_csv_path)


def save_prompt_sensitivity_metrics_json(
    results: list[dict[str, Any]],
    output_json_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Save full prompt sensitivity results and metadata to JSON.
    """
    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata or {},
        "results": results,
    }

    output_json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return str(output_json_path)
