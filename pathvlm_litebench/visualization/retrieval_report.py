from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def save_retrieval_results_csv(
    prompts: list[str],
    retrieval_results: list[list[dict]],
    output_csv_path: str | Path,
    label_prompts: list[str] | None = None,
) -> str:
    """
    Save prompt-level top-k retrieval results to CSV.

    Args:
        prompts:
            Retrieval prompts.
        retrieval_results:
            Nested top-k retrieval results, one list per prompt.
        output_csv_path:
            Output CSV path.
        label_prompts:
            Optional target labels aligned with prompts.

    Returns:
        Saved CSV path as string.
    """
    if len(prompts) != len(retrieval_results):
        raise ValueError(
            f"prompts and retrieval_results must have the same length: "
            f"{len(prompts)} vs {len(retrieval_results)}"
        )

    if label_prompts is not None and len(label_prompts) != len(prompts):
        raise ValueError(
            f"label_prompts and prompts must have the same length: "
            f"{len(label_prompts)} vs {len(prompts)}"
        )

    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "prompt_index",
        "prompt",
        "target_label",
        "rank",
        "image_index",
        "image_path",
        "score",
        "label",
        "is_positive",
    ]

    with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for prompt_idx, (prompt, prompt_results) in enumerate(zip(prompts, retrieval_results)):
            target_label = ""
            if label_prompts is not None:
                target_label = label_prompts[prompt_idx]

            for rank, item in enumerate(prompt_results, start=1):
                score_value: float | str = ""
                raw_score = item.get("score")
                if raw_score is not None:
                    score_value = float(raw_score)

                writer.writerow(
                    {
                        "prompt_index": prompt_idx,
                        "prompt": prompt,
                        "target_label": item.get("target_label", target_label),
                        "rank": rank,
                        "image_index": item.get("index", ""),
                        "image_path": item.get("path", ""),
                        "score": score_value,
                        "label": item.get("label", ""),
                        "is_positive": item.get("is_positive", ""),
                    }
                )

    return str(output_csv_path)


def save_retrieval_metrics_json(
    metrics: dict[str, Any],
    output_json_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Save retrieval metrics and metadata to JSON.

    Args:
        metrics:
            Retrieval metrics payload.
        output_json_path:
            Output JSON path.
        metadata:
            Optional metadata payload.

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
