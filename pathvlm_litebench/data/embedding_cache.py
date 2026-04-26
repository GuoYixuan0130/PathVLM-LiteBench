from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_embeddings(
    embeddings: torch.Tensor,
    save_path: str | Path,
) -> str:
    """
    Save tensor embeddings to a .pt file.

    Args:
        embeddings:
            Tensor embeddings to save.
        save_path:
            Output .pt file path.

    Returns:
        The saved file path as a string.
    """
    save_path = Path(save_path)

    if save_path.suffix != ".pt":
        raise ValueError(f"Embedding cache file must use .pt extension: {save_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.cpu(), save_path)

    return str(save_path)


def load_embeddings(
    load_path: str | Path,
    map_location: str = "cpu",
) -> torch.Tensor:
    """
    Load tensor embeddings from a .pt file.

    Args:
        load_path:
            Input .pt file path.
        map_location:
            Device mapping for torch.load.

    Returns:
        Loaded tensor embeddings.
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Embedding cache file does not exist: {load_path}")

    if load_path.suffix != ".pt":
        raise ValueError(f"Embedding cache file must use .pt extension: {load_path}")

    embeddings = torch.load(load_path, map_location=map_location)

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError(f"Loaded object is not a torch.Tensor: {type(embeddings)}")

    return embeddings


def save_metadata(
    metadata: Any,
    save_path: str | Path,
) -> str:
    """
    Save metadata to a JSON file.

    Args:
        metadata:
            JSON-serializable metadata, such as image paths or prompts.
        save_path:
            Output .json file path.

    Returns:
        The saved file path as a string.
    """
    save_path = Path(save_path)

    if save_path.suffix != ".json":
        raise ValueError(f"Metadata cache file must use .json extension: {save_path}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(save_path)


def load_metadata(
    load_path: str | Path,
) -> Any:
    """
    Load metadata from a JSON file.

    Args:
        load_path:
            Input .json file path.

    Returns:
        Loaded metadata.
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Metadata cache file does not exist: {load_path}")

    if load_path.suffix != ".json":
        raise ValueError(f"Metadata cache file must use .json extension: {load_path}")

    with load_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata
