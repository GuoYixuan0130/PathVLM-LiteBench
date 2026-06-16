from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


@dataclass
class ModelZeroShotResult:
    """Zero-shot accuracy of a single model on a shared patch set."""

    model: str
    accuracy: float
    correct: int
    total: int


def resolve_true_indices(
    labels: Sequence[str | None],
    class_names: Sequence[str],
) -> list[int]:
    """
    Map manifest labels to class indices.

    Each label is resolved either as an integer class index (e.g. "0".."8" for a
    9-class manifest) or as a case-insensitive class-name match against
    ``class_names``. Integer-index interpretation takes precedence so that
    integer-string manifests resolve unambiguously.

    Raises:
        ValueError: if ``class_names`` is empty, or a label is missing or cannot
            be mapped to exactly one class.
    """
    if len(class_names) == 0:
        raise ValueError("class_names must not be empty.")

    num_classes = len(class_names)
    name_to_index = {name.strip().lower(): idx for idx, name in enumerate(class_names)}

    indices: list[int] = []
    for position, label in enumerate(labels):
        if label is None or not str(label).strip():
            raise ValueError(
                f"Missing label at position {position}; every patch must be "
                f"labeled to evaluate zero-shot accuracy."
            )

        text = str(label).strip()

        if text.isdigit():
            index = int(text)
            if 0 <= index < num_classes:
                indices.append(index)
                continue

        name_index = name_to_index.get(text.lower())
        if name_index is not None:
            indices.append(name_index)
            continue

        raise ValueError(
            f"Could not map label {label!r} at position {position} to a class. "
            f"Expected an integer in [0, {num_classes - 1}] or one of: "
            f"{', '.join(class_names)}."
        )

    return indices


def evaluate_models_zero_shot(
    images: Sequence[Any],
    true_indices: Sequence[int],
    class_prompts: Sequence[str],
    models: Sequence[str],
    *,
    device: str | None = "auto",
    batch_size: int = 32,
    show_progress: bool = False,
    model_factory: Callable[[str, str | None], Any] | None = None,
) -> list[ModelZeroShotResult]:
    """
    Run zero-shot tissue classification for several models on a shared patch set.

    Each model encodes the same images and the same class prompts; the predicted
    class is the argmax cosine similarity, and accuracy is the fraction of
    patches whose prediction matches ``true_indices``.

    Args:
        images: Loaded patch images (e.g. PIL images).
        true_indices: Ground-truth class index per image.
        class_prompts: One text prompt per class, in class-index order.
        models: Registry keys or Hugging Face model names to evaluate.
        device: Device option forwarded to the model factory.
        batch_size: Image-encoding batch size.
        show_progress: Whether the model factory's encoder shows a progress bar.
        model_factory: Callable ``(model_key, device) -> wrapper`` used to build
            each model; defaults to :func:`pathvlm_litebench.models.create_model`.
            Injectable for testing without loading real weights.

    Returns:
        One :class:`ModelZeroShotResult` per model, in the order of ``models``.
    """
    if len(images) == 0:
        raise ValueError("images must not be empty.")

    if len(true_indices) != len(images):
        raise ValueError(
            f"true_indices and images must have the same length: "
            f"{len(true_indices)} vs {len(images)}"
        )

    if len(class_prompts) == 0:
        raise ValueError("class_prompts must not be empty.")

    if len(models) == 0:
        raise ValueError("models must not be empty.")

    num_classes = len(class_prompts)
    for position, index in enumerate(true_indices):
        if not 0 <= index < num_classes:
            raise ValueError(
                f"true_indices[{position}] = {index} is out of range for "
                f"{num_classes} classes."
            )

    if model_factory is None:
        from pathvlm_litebench.models import create_model

        model_factory = create_model

    truth = torch.tensor(list(true_indices), dtype=torch.long)
    total = len(images)

    results: list[ModelZeroShotResult] = []
    for model_key in models:
        model = model_factory(model_key, device)
        image_emb = model.encode_images(
            images,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        text_emb = model.encode_text(list(class_prompts))

        similarity = image_emb @ text_emb.T
        predictions = similarity.argmax(dim=1).to(torch.long)
        correct = int((predictions == truth).sum().item())

        results.append(
            ModelZeroShotResult(
                model=str(model_key),
                accuracy=correct / total,
                correct=correct,
                total=total,
            )
        )

    return results
