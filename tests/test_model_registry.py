from pathvlm_litebench.models import (
    list_available_models,
    resolve_model_name,
)


def test_resolve_clip_model_key():
    assert resolve_model_name("clip") == "openai/clip-vit-base-patch32"


def test_resolve_huggingface_model_name():
    assert resolve_model_name("openai/clip-vit-base-patch32") == "openai/clip-vit-base-patch32"


def test_resolve_plip_model_key():
    assert resolve_model_name("plip") == "vinid/plip"


def test_list_available_models():
    models = list_available_models()
    keys = {item["key"] for item in models}

    assert "clip" in keys
    assert "plip" in keys
    assert "conch" in keys


def test_plip_is_marked_implemented():
    models = list_available_models()
    plip = next(item for item in models if item["key"] == "plip")

    assert plip["implemented"] is True
    assert plip["model_name"] == "vinid/plip"


def test_resolve_conch_model_key():
    assert resolve_model_name("conch") == "MahmoodLab/CONCH"


def test_conch_is_marked_implemented():
    models = list_available_models()
    conch = next(item for item in models if item["key"] == "conch")

    assert conch["implemented"] is True
    assert conch["model_name"] == "MahmoodLab/CONCH"
    assert "gated Hugging Face access" in conch["description"]


def test_unknown_model_key_raises_value_error():
    try:
        resolve_model_name("unknown_model")
    except ValueError as exc:
        assert "Unknown model key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown model key.")
