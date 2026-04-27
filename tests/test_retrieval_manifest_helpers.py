import importlib.util
from pathlib import Path

import pytest


def load_retrieval_demo_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "01_patch_text_retrieval_demo.py"
    spec = importlib.util.spec_from_file_location("retrieval_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load retrieval demo module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_text_to_image_positive_pairs_basic():
    module = load_retrieval_demo_module()

    labels = ["tumor", "normal", "tumor", "necrosis"]
    label_prompts = ["tumor", "normal"]

    positive_pairs = module.build_text_to_image_positive_pairs(labels, label_prompts)
    assert positive_pairs == {0: {0, 2}, 1: {1}}


def test_build_text_to_image_positive_pairs_incomplete_labels():
    module = load_retrieval_demo_module()

    labels = ["tumor", None, "normal"]
    label_prompts = ["tumor"]

    with pytest.raises(ValueError):
        module.build_text_to_image_positive_pairs(labels, label_prompts)


def test_build_text_to_image_positive_pairs_missing_label_name():
    module = load_retrieval_demo_module()

    labels = ["tumor", "normal", "tumor"]
    label_prompts = ["necrosis"]

    with pytest.raises(ValueError):
        module.build_text_to_image_positive_pairs(labels, label_prompts)


def test_build_text_to_image_positive_pairs_empty_label_prompts():
    module = load_retrieval_demo_module()

    labels = ["tumor", "normal", "tumor"]

    with pytest.raises(ValueError):
        module.build_text_to_image_positive_pairs(labels, [])
