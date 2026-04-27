from __future__ import annotations

from typing import Any

import torch

from .clip_wrapper import CLIPWrapper


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "clip": {
        "model_name": "openai/clip-vit-base-patch32",
        "implemented": True,
        "description": "OpenAI CLIP ViT-B/32 baseline model.",
    },
    "clip-vit-base-patch32": {
        "model_name": "openai/clip-vit-base-patch32",
        "implemented": True,
        "description": "Alias for OpenAI CLIP ViT-B/32.",
    },
    "plip": {
        "model_name": "vinid/plip",
        "implemented": False,
        "description": "Placeholder for PLIP pathology vision-language model.",
    },
    "conch": {
        "model_name": "MahmoodLab/CONCH",
        "implemented": False,
        "description": "Placeholder for CONCH pathology vision-language model.",
    },
}


def list_available_models() -> list[dict[str, Any]]:
    """
    List registered model keys and metadata.

    Returns:
        A list of model metadata dictionaries.
    """
    models = []

    for key, metadata in MODEL_REGISTRY.items():
        models.append(
            {
                "key": key,
                "model_name": metadata["model_name"],
                "implemented": metadata["implemented"],
                "description": metadata["description"],
            }
        )

    return models


def resolve_model_name(model_key_or_name: str) -> str:
    """
    Resolve a model key or Hugging Face model name to a concrete model name.

    Args:
        model_key_or_name:
            A registered model key, such as "clip", or a Hugging Face model name.

    Returns:
        A Hugging Face model name.

    Raises:
        NotImplementedError:
            If the model key is registered but not implemented.
        ValueError:
            If the key is unknown and does not look like a Hugging Face model name.
    """
    if model_key_or_name in MODEL_REGISTRY:
        metadata = MODEL_REGISTRY[model_key_or_name]

        if not metadata["implemented"]:
            raise NotImplementedError(
                f"Model key '{model_key_or_name}' is registered but not implemented yet. "
                f"Description: {metadata['description']}"
            )

        return metadata["model_name"]

    if "/" in model_key_or_name:
        return model_key_or_name

    available = ", ".join(MODEL_REGISTRY.keys())

    raise ValueError(
        f"Unknown model key: '{model_key_or_name}'. "
        f"Available registered keys: {available}. "
        f"You can also pass a Hugging Face model name such as 'openai/clip-vit-base-patch32'."
    )


def resolve_device(device: str | None = None) -> str | None:
    """
    Resolve a device option for model creation.

    Args:
        device:
            One of None, "auto", "cpu", or "cuda".

    Returns:
        A device string supported by CLIPWrapper, or None for auto selection.

    Raises:
        ValueError:
            If an unsupported device value is provided or CUDA is unavailable.
    """
    if device is None or device == "auto":
        return None

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise ValueError("CUDA was requested but is not available")

    raise ValueError(
        f"Invalid device '{device}'; allowed values are: auto, cpu, cuda."
    )


def create_model(
    model_key_or_name: str = "clip",
    device: str | None = None,
) -> CLIPWrapper:
    """
    Create a model wrapper from a model key or Hugging Face model name.

    Args:
        model_key_or_name:
            Registered model key or Hugging Face model name.
        device:
            Optional device string, such as "cpu" or "cuda".

    Returns:
        A CLIPWrapper instance.
    """
    model_name = resolve_model_name(model_key_or_name)
    resolved_device = resolve_device(device)

    return CLIPWrapper(
        model_name=model_name,
        device=resolved_device,
    )
