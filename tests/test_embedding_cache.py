from pathlib import Path

import torch

from pathvlm_litebench.data import (
    save_embeddings,
    load_embeddings,
    save_metadata,
    load_metadata,
)


def test_embedding_cache_roundtrip(tmp_path: Path):
    embeddings = torch.randn(3, 8)

    embedding_path = tmp_path / "embeddings.pt"

    saved_path = save_embeddings(embeddings, embedding_path)
    loaded_embeddings = load_embeddings(saved_path)

    assert torch.allclose(embeddings, loaded_embeddings)


def test_metadata_cache_roundtrip(tmp_path: Path):
    metadata = {
        "image_paths": ["a.png", "b.png"],
        "model_name": "openai/clip-vit-base-patch32",
    }

    metadata_path = tmp_path / "metadata.json"

    saved_path = save_metadata(metadata, metadata_path)
    loaded_metadata = load_metadata(saved_path)

    assert loaded_metadata == metadata
