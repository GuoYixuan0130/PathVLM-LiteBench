from __future__ import annotations

import pytest
import torch
from PIL import Image

import pathvlm_litebench.models.clip_wrapper as clip_wrapper
from pathvlm_litebench.models.clip_wrapper import CLIPWrapper


class _FakeInputs(dict):
    def to(self, device):
        return self


class _FakeProcessor:
    def __call__(self, images=None, text=None, return_tensors=None, padding=None, truncation=None):
        if images is not None:
            return _FakeInputs(count=len(images), offset=0)
        return _FakeInputs(count=len(text), offset=1000)


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def get_image_features(self, count, offset):
        # Distinct, non-normalized rows so batching/order can be verified.
        base = torch.arange(count, dtype=torch.float32) + offset + 1
        return torch.stack([base, base * 2, base * 3], dim=1)

    def get_text_features(self, count, offset):
        base = torch.arange(count, dtype=torch.float32) + offset + 1
        return torch.stack([base, base * 2, base * 3], dim=1)


@pytest.fixture
def fake_clip(monkeypatch):
    monkeypatch.setattr(
        clip_wrapper.CLIPModel,
        "from_pretrained",
        staticmethod(lambda name: _FakeModel()),
    )
    monkeypatch.setattr(
        clip_wrapper.CLIPProcessor,
        "from_pretrained",
        staticmethod(lambda name: _FakeProcessor()),
    )
    return CLIPWrapper(device="cpu")


def _images(n):
    return [Image.new("RGB", (4, 4)) for _ in range(n)]


def test_encode_images_batched_matches_single_batch(fake_clip):
    images = _images(5)

    single = fake_clip.encode_images(images, batch_size=10, show_progress=False)
    batched = fake_clip.encode_images(images, batch_size=2, show_progress=False)

    assert single.shape == (5, 3)
    assert batched.shape == (5, 3)
    assert torch.allclose(single, batched)


def test_encode_images_outputs_are_l2_normalized(fake_clip):
    embeddings = fake_clip.encode_images(_images(7), batch_size=3, show_progress=False)

    norms = torch.linalg.norm(embeddings, dim=-1)
    assert torch.allclose(norms, torch.ones(7), atol=1e-5)


def test_encode_images_rejects_non_positive_batch_size(fake_clip):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        fake_clip.encode_images(_images(2), batch_size=0)
