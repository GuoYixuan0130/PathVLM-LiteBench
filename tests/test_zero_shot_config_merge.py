import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


def load_zero_shot_demo_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "02_zero_shot_classification_demo.py"
    spec = importlib.util.spec_from_file_location("zero_shot_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load zero-shot demo module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_args(**overrides) -> Namespace:
    base = {
        "config": None,
        "image_dir": None,
        "manifest": None,
        "image_root": None,
        "split": None,
        "max_images": None,
        "class_names": None,
        "class_prompts": None,
        "top_k": None,
        "model": None,
        "device": None,
        "save_report": False,
        "report_dir": None,
    }
    base.update(overrides)
    return Namespace(**base)


def test_merge_args_with_config_zero_shot(tmp_path: Path):
    module = load_zero_shot_demo_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "zero_shot",
                "model": "clip",
                "device": "auto",
                "manifest": "dataset/MHIST/manifest.csv",
                "image_root": "dataset/MHIST/images",
                "split": "test",
                "class_names": ["HP", "SSA"],
                "class_prompts": [
                    "a histopathology image of hyperplastic polyp",
                    "a histopathology image of sessile serrated adenoma",
                ],
                "top_k": 2,
                "save_report": True,
                "report_dir": "outputs/zero_shot_demo",
            }
        ),
        encoding="utf-8",
    )

    args = _build_args(config=str(config_path))
    merged = module.merge_args_with_config(args)

    assert merged["model"] == "clip"
    assert merged["device"] == "auto"
    assert merged["manifest"] == "dataset/MHIST/manifest.csv"
    assert merged["image_root"] == "dataset/MHIST/images"
    assert merged["split"] == "test"
    assert merged["class_names"] == ["HP", "SSA"]
    assert merged["top_k"] == 2
    assert merged["save_report"] is True
    assert merged["report_dir"] == "outputs/zero_shot_demo"


def test_merge_args_with_config_cli_override_top_k(tmp_path: Path):
    module = load_zero_shot_demo_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "zero_shot",
                "model": "clip",
                "device": "auto",
                "top_k": 5,
            }
        ),
        encoding="utf-8",
    )

    args = _build_args(config=str(config_path), top_k=2)
    merged = module.merge_args_with_config(args)

    assert merged["top_k"] == 2


def test_merge_args_with_config_wrong_task(tmp_path: Path):
    module = load_zero_shot_demo_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "retrieval",
                "model": "clip",
                "device": "auto",
                "top_k": 5,
            }
        ),
        encoding="utf-8",
    )

    args = _build_args(config=str(config_path))
    with pytest.raises(ValueError):
        module.merge_args_with_config(args)


def test_merge_args_without_config_uses_defaults():
    module = load_zero_shot_demo_module()
    args = _build_args()
    merged = module.merge_args_with_config(args)

    assert merged["model"] == "clip"
    assert merged["device"] == "auto"
    assert merged["top_k"] == 3
    assert merged["report_dir"] == "outputs/zero_shot_demo"
