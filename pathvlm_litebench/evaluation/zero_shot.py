from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def zero_shot_predict(
    image_embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    class_names: list[str],
    top_k: int = 1,
    temperature: float = 1.0,
    normalize: bool = True,
) -> list[dict[str, Any]]:
    """
    Perform zero-shot classification using image and class text embeddings.

    Args:
        image_embeddings:
            Tensor of shape [num_images, embedding_dim].
        class_embeddings:
            Tensor of shape [num_classes, embedding_dim].
        class_names:
            A list of class names. Length must match num_classes.
        top_k:
            Number of top class predictions to return for each image.
        temperature:
            Temperature used before softmax. Lower values make predictions sharper.
        normalize:
            Whether to L2-normalize embeddings before similarity computation.

    Returns:
        A list of dictionaries. Each dictionary contains:
        - image_index
        - predicted_index
        - predicted_label
        - confidence
        - top_predictions
    """
    if image_embeddings.ndim != 2:
        raise ValueError("image_embeddings must have shape [num_images, embedding_dim].")

    if class_embeddings.ndim != 2:
        raise ValueError("class_embeddings must have shape [num_classes, embedding_dim].")

    if image_embeddings.shape[1] != class_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match: "
            f"image dim = {image_embeddings.shape[1]}, "
            f"class dim = {class_embeddings.shape[1]}"
        )

    num_classes = class_embeddings.shape[0]

    if len(class_names) != num_classes:
        raise ValueError(
            f"Length of class_names must match number of class embeddings: "
            f"{len(class_names)} vs {num_classes}"
        )

    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    top_k = min(top_k, num_classes)

    if normalize:
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)

    logits = image_embeddings @ class_embeddings.T
    logits = logits / temperature

    probabilities = torch.softmax(logits, dim=-1)

    top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=-1)

    results: list[dict[str, Any]] = []

    for image_idx in range(image_embeddings.shape[0]):
        top_predictions = []

        for prob, class_idx in zip(top_probs[image_idx], top_indices[image_idx]):
            idx = int(class_idx.item())
            top_predictions.append(
                {
                    "class_index": idx,
                    "class_name": class_names[idx],
                    "probability": float(prob.item()),
                    "logit": float(logits[image_idx, idx].item()),
                }
            )

        best_prediction = top_predictions[0]

        results.append(
            {
                "image_index": image_idx,
                "predicted_index": best_prediction["class_index"],
                "predicted_label": best_prediction["class_name"],
                "confidence": best_prediction["probability"],
                "top_predictions": top_predictions,
            }
        )

    return results


def compute_accuracy(
    predicted_labels: list[str],
    true_labels: list[str],
) -> float:
    """
    Compute classification accuracy.

    Args:
        predicted_labels:
            Predicted class labels.
        true_labels:
            Ground-truth class labels.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    if len(predicted_labels) != len(true_labels):
        raise ValueError(
            f"predicted_labels and true_labels must have the same length: "
            f"{len(predicted_labels)} vs {len(true_labels)}"
        )

    if len(true_labels) == 0:
        raise ValueError("true_labels must not be empty.")

    correct = sum(pred == true for pred, true in zip(predicted_labels, true_labels))

    return correct / len(true_labels)
