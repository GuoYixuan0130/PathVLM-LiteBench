from __future__ import annotations

import pytest
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel

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


# --- Dependency-contract regression guards (transformers 5.x compatibility) ---
#
# The wrapper passes the output of CLIPModel.get_text_features / get_image_features
# straight into F.normalize, so it relies on those returning plain tensors.
# transformers 5.x changed them to return BaseModelOutputWithPooling objects, which
# silently shipped a broken 0.11.0. The mocked tests above can't catch that because
# they fake the model. These tests run the real transformers code path on a tiny,
# randomly-initialized model (no weight download) to keep that contract honest.

_TEXT_PROJECTION_DIM = 16


def _tiny_clip_config() -> CLIPConfig:
    return CLIPConfig(
        projection_dim=_TEXT_PROJECTION_DIM,
        text_config={
            "vocab_size": 99,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 16,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 30,
            "patch_size": 15,
        },
    )


class _RealInputs(dict):
    def to(self, device):
        return _RealInputs({key: value.to(device) for key, value in self.items()})


class _RealTensorProcessor:
    """Produces the real tensors a tiny CLIPModel expects, without a tokenizer."""

    def __init__(self, vocab_size: int, image_size: int, seq_len: int = 8):
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.seq_len = seq_len

    def __call__(self, images=None, text=None, return_tensors=None, padding=None, truncation=None):
        if images is not None:
            n = len(images)
            return _RealInputs(
                pixel_values=torch.randn(n, 3, self.image_size, self.image_size)
            )
        n = len(text)
        return _RealInputs(
            input_ids=torch.randint(0, self.vocab_size, (n, self.seq_len)),
            attention_mask=torch.ones(n, self.seq_len, dtype=torch.long),
        )


@pytest.fixture
def tiny_real_clip(monkeypatch):
    config = _tiny_clip_config()
    model = CLIPModel(config)
    processor = _RealTensorProcessor(
        vocab_size=config.text_config.vocab_size,
        image_size=config.vision_config.image_size,
    )
    monkeypatch.setattr(
        clip_wrapper.CLIPModel,
        "from_pretrained",
        staticmethod(lambda name: model),
    )
    monkeypatch.setattr(
        clip_wrapper.CLIPProcessor,
        "from_pretrained",
        staticmethod(lambda name: processor),
    )
    return CLIPWrapper(device="cpu")


def test_installed_transformers_returns_feature_tensors():
    config = _tiny_clip_config()
    model = CLIPModel(config).eval()
    input_ids = torch.randint(0, config.text_config.vocab_size, (3, 8))
    pixel_values = torch.randn(2, 3, config.vision_config.image_size, config.vision_config.image_size)

    with torch.no_grad():
        text_features = model.get_text_features(input_ids=input_ids)
        image_features = model.get_image_features(pixel_values=pixel_values)

    assert isinstance(text_features, torch.Tensor)
    assert isinstance(image_features, torch.Tensor)
    assert text_features.shape == (3, config.projection_dim)
    assert image_features.shape == (2, config.projection_dim)


def test_wrapper_encode_runs_against_real_transformers_model(tiny_real_clip):
    text_embeddings = tiny_real_clip.encode_text(["alpha", "beta", "gamma"])
    image_embeddings = tiny_real_clip.encode_images(_images(4), batch_size=2, show_progress=False)

    assert text_embeddings.shape == (3, _TEXT_PROJECTION_DIM)
    assert image_embeddings.shape == (4, _TEXT_PROJECTION_DIM)
    assert torch.allclose(torch.linalg.norm(text_embeddings, dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(image_embeddings, dim=-1), torch.ones(4), atol=1e-5)
