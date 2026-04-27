import torch
import torch.nn.functional as F

from pathvlm_litebench.evaluation import (
    analyze_prompt_sensitivity,
    compute_jaccard_overlap,
)


def test_compute_jaccard_overlap():
    overlap = compute_jaccard_overlap([0, 1], [1, 2])
    assert abs(overlap - (1 / 3)) < 1e-6


def test_analyze_prompt_sensitivity_basic():
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
        ]
    )

    prompt_embeddings_by_concept = [
        torch.tensor(
            [
                [1.0, 0.0],
                [0.95, 0.05],
            ]
        )
    ]

    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    prompt_embeddings_by_concept = [
        F.normalize(item, p=2, dim=-1) for item in prompt_embeddings_by_concept
    ]

    results = analyze_prompt_sensitivity(
        image_embeddings=image_embeddings,
        prompt_embeddings_by_concept=prompt_embeddings_by_concept,
        concept_names=["tumor_like"],
        prompt_texts_by_concept=[["tumor tissue", "malignant tissue"]],
        top_k=2,
    )

    assert len(results) == 1
    assert results[0]["concept_name"] == "tumor_like"
    assert results[0]["num_prompts"] == 2
    assert "mean_topk_overlap" in results[0]
    assert "mean_similarity_std" in results[0]
    assert len(results[0]["prompt_results"]) == 2
