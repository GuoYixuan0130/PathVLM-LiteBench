from PIL import Image
import pytest
import torch

from pathvlm_litebench.evaluation import score_patch_images_for_prompt


class FakeScorer:
    def __init__(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> None:
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        return self.image_embeddings

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.text_embeddings


def test_score_patch_images_for_prompt_computes_similarity():
    images = [
        Image.new("RGB", (8, 8), color="red"),
        Image.new("RGB", (8, 8), color="blue"),
    ]
    model = FakeScorer(
        image_embeddings=torch.tensor([[1.0, 0.0], [0.25, 0.75]]),
        text_embeddings=torch.tensor([[1.0, 0.0]]),
    )

    scores = score_patch_images_for_prompt(
        images=images,
        prompt="synthetic red score",
        model=model,
    )

    assert scores == [1.0, 0.25]


def test_score_patch_images_for_prompt_rejects_empty_prompt():
    model = FakeScorer(
        image_embeddings=torch.tensor([[1.0, 0.0]]),
        text_embeddings=torch.tensor([[1.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="prompt"):
        score_patch_images_for_prompt(
            images=[Image.new("RGB", (8, 8))],
            prompt=" ",
            model=model,
        )


def test_score_patch_images_for_prompt_rejects_dimension_mismatch():
    model = FakeScorer(
        image_embeddings=torch.tensor([[1.0, 0.0]]),
        text_embeddings=torch.tensor([[1.0, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="Embedding dimensions"):
        score_patch_images_for_prompt(
            images=[Image.new("RGB", (8, 8))],
            prompt="synthetic prompt",
            model=model,
        )
