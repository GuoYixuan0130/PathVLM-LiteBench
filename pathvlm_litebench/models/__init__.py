from .registry import (
    MODEL_REGISTRY,
    create_model,
    list_available_models,
    resolve_device,
    resolve_model_name,
)

__all__ = [
    "CLIPWrapper",
    "PLIPWrapper",
    "CONCHWrapper",
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
    if name == "PLIPWrapper":
        from .plip_wrapper import PLIPWrapper

        return PLIPWrapper
    if name == "CONCHWrapper":
        from .conch_wrapper import CONCHWrapper

        return CONCHWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
