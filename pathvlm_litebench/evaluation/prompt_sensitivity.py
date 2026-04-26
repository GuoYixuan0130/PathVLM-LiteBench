from __future__ import annotations

from itertools import combinations
from typing import Any

import torch
import torch.nn.functional as F


def compute_jaccard_overlap(indices_a: list[int], indices_b: list[int]) -> float:
    """
    Compute Jaccard overlap between two lists of retrieved indices.

    Args:
        indices_a:
            First list of retrieved image indices.
        indices_b:
            Second list of retrieved image indices.

    Returns:
        Jaccard overlap between 0 and 1.
    """
    set_a = set(indices_a)
    set_b = set(indices_b)

    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0

    union = set_a | set_b

    if len(union) == 0:
        return 0.0

    intersection = set_a & set_b

    return len(intersection) / len(union)


def _mean_pairwise_overlap(topk_indices: list[list[int]]) -> float:
    """
    Compute mean pairwise Jaccard overlap among multiple top-k index lists.
    """
    if len(topk_indices) <= 1:
        return 1.0

    overlaps = []

    for indices_a, indices_b in combinations(topk_indices, 2):
        overlaps.append(compute_jaccard_overlap(indices_a, indices_b))

    return sum(overlaps) / len(overlaps)


def analyze_prompt_sensitivity(
    image_embeddings: torch.Tensor,
    prompt_embeddings_by_concept: list[torch.Tensor],
    concept_names: list[str],
    prompt_texts_by_concept: list[list[str]] | None = None,
    top_k: int = 5,
    normalize: bool = True,
) -> list[dict[str, Any]]:
    """
    Analyze prompt sensitivity for each concept.

    For each concept, this function compares multiple prompt variants by:
    - computing similarity between all images and each prompt variant
    - retrieving top-k images for each prompt variant
    - computing mean pairwise Jaccard overlap among top-k results
    - computing average standard deviation of similarity scores across prompt variants

    Args:
        image_embeddings:
            Tensor of shape [num_images, embedding_dim].
        prompt_embeddings_by_concept:
            A list where each item is a tensor of shape [num_prompts, embedding_dim].
            Each tensor contains prompt embeddings for one concept.
        concept_names:
            Names of concepts, such as ["tumor", "normal", "necrosis"].
        prompt_texts_by_concept:
            Optional prompt texts for each concept.
        top_k:
            Number of top retrieved images used to measure overlap.
        normalize:
            Whether to L2-normalize embeddings before similarity computation.

    Returns:
        A list of dictionaries. Each dictionary contains:
        - concept_name
        - num_prompts
        - mean_topk_overlap
        - mean_similarity_std
        - prompt_results
    """
    if image_embeddings.ndim != 2:
        raise ValueError("image_embeddings must have shape [num_images, embedding_dim].")

    if len(prompt_embeddings_by_concept) != len(concept_names):
        raise ValueError(
            f"prompt_embeddings_by_concept and concept_names must have the same length: "
            f"{len(prompt_embeddings_by_concept)} vs {len(concept_names)}"
        )

    if prompt_texts_by_concept is not None and len(prompt_texts_by_concept) != len(concept_names):
        raise ValueError(
            f"prompt_texts_by_concept and concept_names must have the same length: "
            f"{len(prompt_texts_by_concept)} vs {len(concept_names)}"
        )

    num_images = image_embeddings.shape[0]
    top_k = min(top_k, num_images)

    if normalize:
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    results: list[dict[str, Any]] = []

    for concept_idx, concept_name in enumerate(concept_names):
        prompt_embeddings = prompt_embeddings_by_concept[concept_idx]

        if prompt_embeddings.ndim != 2:
            raise ValueError(
                f"Prompt embeddings for concept '{concept_name}' must have shape "
                f"[num_prompts, embedding_dim]."
            )

        if prompt_embeddings.shape[1] != image_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dimensions do not match for concept '{concept_name}': "
                f"image dim = {image_embeddings.shape[1]}, "
                f"prompt dim = {prompt_embeddings.shape[1]}"
            )

        if normalize:
            prompt_embeddings = F.normalize(prompt_embeddings, p=2, dim=-1)

        num_prompts = prompt_embeddings.shape[0]

        if prompt_texts_by_concept is not None:
            prompt_texts = prompt_texts_by_concept[concept_idx]
            if len(prompt_texts) != num_prompts:
                raise ValueError(
                    f"Number of prompt texts must match number of prompt embeddings "
                    f"for concept '{concept_name}': {len(prompt_texts)} vs {num_prompts}"
                )
        else:
            prompt_texts = [f"prompt_{i}" for i in range(num_prompts)]

        similarity = image_embeddings @ prompt_embeddings.T
        # similarity shape: [num_images, num_prompts]

        topk_indices_by_prompt: list[list[int]] = []
        prompt_results: list[dict[str, Any]] = []

        for prompt_idx in range(num_prompts):
            scores = similarity[:, prompt_idx]
            top_scores, top_indices = torch.topk(scores, k=top_k)

            top_indices_list = [int(index.item()) for index in top_indices]
            top_scores_list = [float(score.item()) for score in top_scores]

            topk_indices_by_prompt.append(top_indices_list)

            prompt_results.append(
                {
                    "prompt_index": prompt_idx,
                    "prompt_text": prompt_texts[prompt_idx],
                    "top_indices": top_indices_list,
                    "top_scores": top_scores_list,
                }
            )

        mean_topk_overlap = _mean_pairwise_overlap(topk_indices_by_prompt)

        if num_prompts > 1:
            similarity_std = torch.std(similarity, dim=1)
            mean_similarity_std = float(similarity_std.mean().item())
        else:
            mean_similarity_std = 0.0

        results.append(
            {
                "concept_name": concept_name,
                "num_prompts": num_prompts,
                "mean_topk_overlap": float(mean_topk_overlap),
                "mean_similarity_std": mean_similarity_std,
                "prompt_results": prompt_results,
            }
        )

    return results
