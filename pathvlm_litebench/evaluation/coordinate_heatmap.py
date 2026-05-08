from __future__ import annotations

from typing import Protocol

import torch
from PIL import Image


class PatchTextScorer(Protocol):
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        ...

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        ...


def score_patch_images_for_prompt(
    images: list[Image.Image],
    prompt: str,
    model: PatchTextScorer,
) -> list[float]:
    """
    Compute one image-text similarity score per patch image.
    """
    if len(images) == 0:
        raise ValueError("images must contain at least one item")

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    image_embeddings = model.encode_images(images)
    text_embeddings = model.encode_text([prompt.strip()])

    if image_embeddings.ndim != 2:
        raise ValueError("image embeddings must have shape [num_images, embedding_dim]")

    if text_embeddings.ndim != 2:
        raise ValueError("text embeddings must have shape [num_texts, embedding_dim]")

    if image_embeddings.shape[0] != len(images):
        raise ValueError(
            f"image embedding count must match image count: "
            f"{image_embeddings.shape[0]} vs {len(images)}"
        )

    if text_embeddings.shape[0] != 1:
        raise ValueError(
            f"expected exactly one text embedding for one prompt. "
            f"Got: {text_embeddings.shape[0]}"
        )

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match: "
            f"image dim = {image_embeddings.shape[1]}, "
            f"text dim = {text_embeddings.shape[1]}"
        )

    scores = image_embeddings @ text_embeddings.T
    return [float(score) for score in scores[:, 0].detach().cpu().tolist()]
