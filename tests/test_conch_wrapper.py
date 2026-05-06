from __future__ import annotations

import sys
import types
import builtins

import pytest
import torch
from PIL import Image


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        rows = []
        for index, _text in enumerate(texts, start=1):
            row = [1, index + 10, 2]
            row.extend([0] * 124)
            rows.append(row)
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


class _FakeModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def encode_text(self, tokens):
        rows = []
        for index in range(tokens.shape[0]):
            rows.append([float(index + 1), 0.0, 0.0])
        return torch.tensor(rows, dtype=torch.float32, device=tokens.device)

    def encode_image(self, images):
        rows = []
        for index in range(images.shape[0]):
            rows.append([0.0, float(index + 1), 0.0])
        return torch.tensor(rows, dtype=torch.float32, device=images.device)


def _install_fake_conch_modules(monkeypatch, create_model):
    conch_module = types.ModuleType("conch")
    open_clip_module = types.ModuleType("conch.open_clip_custom")
    tokenizer_module = types.ModuleType("conch.open_clip_custom.custom_tokenizer")

    open_clip_module.create_model_from_pretrained = create_model
    tokenizer_module.get_tokenizer = lambda: _FakeTokenizer()

    monkeypatch.setitem(sys.modules, "conch", conch_module)
    monkeypatch.setitem(sys.modules, "conch.open_clip_custom", open_clip_module)
    monkeypatch.setitem(
        sys.modules,
        "conch.open_clip_custom.custom_tokenizer",
        tokenizer_module,
    )


def test_conch_wrapper_encodes_text_and_images_with_fake_model(monkeypatch):
    fake_model = _FakeModel()

    def create_model_from_pretrained(model_cfg, checkpoint_path, device):
        def preprocess(image):
            assert isinstance(image, Image.Image)
            return torch.ones(3, 4, 4)

        return fake_model, preprocess

    _install_fake_conch_modules(monkeypatch, create_model_from_pretrained)

    from pathvlm_litebench.models.conch_wrapper import CONCHWrapper

    wrapper = CONCHWrapper(device="cpu")

    text_embeddings = wrapper.encode_text(["prompt a", "prompt b"])
    image_embeddings = wrapper.encode_images(
        [
            Image.new("RGB", (4, 4), "white"),
            Image.new("RGB", (4, 4), "purple"),
        ]
    )
    similarity = wrapper.compute_similarity(image_embeddings, text_embeddings)

    assert fake_model.eval_called is True
    assert text_embeddings.shape == (2, 3)
    assert image_embeddings.shape == (2, 3)
    assert torch.allclose(torch.linalg.norm(text_embeddings, dim=-1), torch.ones(2))
    assert torch.allclose(torch.linalg.norm(image_embeddings, dim=-1), torch.ones(2))
    assert similarity.shape == (2, 2)


def test_conch_wrapper_missing_dependency_error(monkeypatch):
    for module_name in [
        "conch",
        "conch.open_clip_custom",
        "conch.open_clip_custom.custom_tokenizer",
    ]:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("conch.open_clip_custom"):
            raise ImportError("No module named conch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from pathvlm_litebench.models.conch_wrapper import CONCHWrapper

    with pytest.raises(ImportError, match="optional CONCH package"):
        CONCHWrapper(device="cpu")


def test_conch_wrapper_load_error_is_clear(monkeypatch):
    def create_model_from_pretrained(model_cfg, checkpoint_path, device):
        raise RuntimeError("access denied")

    _install_fake_conch_modules(monkeypatch, create_model_from_pretrained)

    from pathvlm_litebench.models.conch_wrapper import CONCHWrapper

    with pytest.raises(RuntimeError, match="Failed to load CONCH"):
        CONCHWrapper(device="cpu")


def test_create_model_uses_conch_wrapper_for_model_key(monkeypatch):
    fake_model = _FakeModel()

    def create_model_from_pretrained(model_cfg, checkpoint_path, device):
        return fake_model, lambda image: torch.ones(3, 4, 4)

    _install_fake_conch_modules(monkeypatch, create_model_from_pretrained)

    from pathvlm_litebench.models import create_model
    from pathvlm_litebench.models.conch_wrapper import CONCHWrapper

    model = create_model("conch", device="cpu")

    assert isinstance(model, CONCHWrapper)
    assert model.model_name == "MahmoodLab/CONCH"


def test_create_model_uses_conch_wrapper_for_hf_name(monkeypatch):
    fake_model = _FakeModel()

    def create_model_from_pretrained(model_cfg, checkpoint_path, device):
        return fake_model, lambda image: torch.ones(3, 4, 4)

    _install_fake_conch_modules(monkeypatch, create_model_from_pretrained)

    from pathvlm_litebench.models import create_model
    from pathvlm_litebench.models.conch_wrapper import CONCHWrapper

    model = create_model("MahmoodLab/CONCH", device="cpu")

    assert isinstance(model, CONCHWrapper)
    assert model.model_name == "MahmoodLab/CONCH"
