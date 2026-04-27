from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def _validate_embeddings(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> None:
    """
    Validate image and text embedding tensors.
    """
    if image_embeddings.ndim != 2:
        raise ValueError("image_embeddings must have shape [num_images, embedding_dim].")

    if text_embeddings.ndim != 2:
        raise ValueError("text_embeddings must have shape [num_texts, embedding_dim].")

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match: "
            f"image dim = {image_embeddings.shape[1]}, "
            f"text dim = {text_embeddings.shape[1]}"
        )


def _validate_positive_pairs(
    positive_pairs: dict[int, set[int]],
    num_queries: int,
    num_targets: int,
) -> None:
    """
    Validate positive pair mapping.

    The mapping is:
    query_index -> set of positive target indices
    """
    for query_idx, target_indices in positive_pairs.items():
        if query_idx < 0 or query_idx >= num_queries:
            raise ValueError(
                f"Query index out of range: {query_idx}. "
                f"Valid range is [0, {num_queries - 1}]."
            )

        if len(target_indices) == 0:
            raise ValueError(f"Positive target set for query {query_idx} is empty.")

        for target_idx in target_indices:
            if target_idx < 0 or target_idx >= num_targets:
                raise ValueError(
                    f"Target index out of range: {target_idx}. "
                    f"Valid range is [0, {num_targets - 1}]."
                )


def compute_recall_at_k_from_similarity(
    similarity: torch.Tensor,
    positive_pairs: dict[int, set[int]],
    k_values: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """
    Compute Recall@K from a similarity matrix.

    Args:
        similarity:
            Tensor of shape [num_queries, num_targets].
            Higher score means more similar.
        positive_pairs:
            Mapping from query index to a set of positive target indices.
        k_values:
            Iterable of K values.

    Returns:
        Dictionary such as:
        {
            "R@1": 0.25,
            "R@5": 0.75,
            "R@10": 0.90
        }

    Notes:
        A query is counted as recalled at K if at least one of its positive
        targets appears in the top-K retrieved targets.
    """
    if similarity.ndim != 2:
        raise ValueError("similarity must have shape [num_queries, num_targets].")

    num_queries, num_targets = similarity.shape

    if len(positive_pairs) == 0:
        raise ValueError("positive_pairs must not be empty.")

    _validate_positive_pairs(
        positive_pairs=positive_pairs,
        num_queries=num_queries,
        num_targets=num_targets,
    )

    metrics: dict[str, float] = {}

    for k in k_values:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        k_eff = min(k, num_targets)

        recalled = 0
        evaluated_queries = 0

        for query_idx, positives in positive_pairs.items():
            scores = similarity[query_idx]
            top_indices = torch.topk(scores, k=k_eff).indices.tolist()

            if any(index in positives for index in top_indices):
                recalled += 1

            evaluated_queries += 1

        metrics[f"R@{k}"] = recalled / evaluated_queries

    return metrics


def compute_text_to_image_recall_at_k(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    positive_pairs: dict[int, set[int]],
    k_values: Iterable[int] = (1, 5, 10),
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute text-to-image Recall@K.

    Args:
        image_embeddings:
            Tensor of shape [num_images, embedding_dim].
        text_embeddings:
            Tensor of shape [num_texts, embedding_dim].
        positive_pairs:
            Mapping from text index to a set of positive image indices.
        k_values:
            Iterable of K values.
        normalize:
            Whether to L2-normalize embeddings before similarity computation.

    Returns:
        Recall@K metrics for text-to-image retrieval.
    """
    _validate_embeddings(image_embeddings, text_embeddings)

    if normalize:
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    similarity = text_embeddings @ image_embeddings.T
    # shape: [num_texts, num_images]

    return compute_recall_at_k_from_similarity(
        similarity=similarity,
        positive_pairs=positive_pairs,
        k_values=k_values,
    )


def compute_image_to_text_recall_at_k(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    positive_pairs: dict[int, set[int]],
    k_values: Iterable[int] = (1, 5, 10),
    normalize: bool = True,
) -> dict[str, float]:
    """
    Compute image-to-text Recall@K.

    Args:
        image_embeddings:
            Tensor of shape [num_images, embedding_dim].
        text_embeddings:
            Tensor of shape [num_texts, embedding_dim].
        positive_pairs:
            Mapping from image index to a set of positive text indices.
        k_values:
            Iterable of K values.
        normalize:
            Whether to L2-normalize embeddings before similarity computation.

    Returns:
        Recall@K metrics for image-to-text retrieval.
    """
    _validate_embeddings(image_embeddings, text_embeddings)

    if normalize:
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    similarity = image_embeddings @ text_embeddings.T
    # shape: [num_images, num_texts]

    return compute_recall_at_k_from_similarity(
        similarity=similarity,
        positive_pairs=positive_pairs,
        k_values=k_values,
    )


def compute_mean_recall(metrics: dict[str, float]) -> float:
    """
    Compute the mean value of a Recall@K metric dictionary.

    Args:
        metrics:
            Dictionary containing recall metrics.

    Returns:
        Mean recall value.
    """
    if len(metrics) == 0:
        raise ValueError("metrics must not be empty.")

    return sum(metrics.values()) / len(metrics)
