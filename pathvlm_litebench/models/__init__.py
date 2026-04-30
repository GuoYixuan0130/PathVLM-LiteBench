from .registry import (
    MODEL_REGISTRY,
    create_model,
    list_available_models,
    resolve_device,
    resolve_model_name,
)

__all__ = [
    "CLIPWrapper",
    "MODEL_REGISTRY",
    "create_model",
    "list_available_models",
    "resolve_device",
    "resolve_model_name",
]


def __getattr__(name: str):
    if name == "CLIPWrapper":
        from .clip_wrapper import CLIPWrapper

        return CLIPWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
