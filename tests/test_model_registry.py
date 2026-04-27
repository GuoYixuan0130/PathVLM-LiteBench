from pathvlm_litebench.models import (
    list_available_models,
    resolve_model_name,
)


def test_resolve_clip_model_key():
    assert resolve_model_name("clip") == "openai/clip-vit-base-patch32"


def test_resolve_huggingface_model_name():
    assert resolve_model_name("openai/clip-vit-base-patch32") == "openai/clip-vit-base-patch32"


def test_list_available_models():
    models = list_available_models()
    keys = {item["key"] for item in models}

    assert "clip" in keys
    assert "plip" in keys
    assert "conch" in keys


def test_placeholder_model_raises_not_implemented():
    try:
        resolve_model_name("plip")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("Expected NotImplementedError for PLIP placeholder.")


def test_unknown_model_key_raises_value_error():
    try:
        resolve_model_name("unknown_model")
    except ValueError as exc:
        assert "Unknown model key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown model key.")
