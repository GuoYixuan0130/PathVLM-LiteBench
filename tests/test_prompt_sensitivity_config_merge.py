import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest


def load_prompt_sensitivity_demo_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "03_prompt_sensitivity_demo.py"
    spec = importlib.util.spec_from_file_location("prompt_sensitivity_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load prompt sensitivity demo module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_args(**overrides) -> Namespace:
    base = {
        "config": None,
        "image_dir": None,
        "top_k": None,
        "model": None,
        "device": None,
        "use_pathology_prompts": False,
        "concepts": None,
        "save_report": False,
        "report_dir": None,
    }
    base.update(overrides)
    return Namespace(**base)


def test_merge_args_with_config_prompt_sensitivity(tmp_path: Path):
    module = load_prompt_sensitivity_demo_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "prompt_sensitivity",
                "model": "clip",
                "device": "auto",
                "image_dir": "dataset/MHIST/images",
                "top_k": 5,
                "use_pathology_prompts": True,
                "concepts": ["tumor", "normal"],
                "save_report": True,
                "report_dir": "outputs/prompt_sensitivity_demo",
            }
        ),
        encoding="utf-8",
    )

    args = _build_args(config=str(config_path))
    merged = module.merge_args_with_config(args)

    assert merged["model"] == "clip"
    assert merged["device"] == "auto"
    assert merged["top_k"] == 5
    assert merged["use_pathology_prompts"] is True
    assert merged["save_report"] is True


def test_merge_args_with_config_cli_override_top_k(tmp_path: Path):
    module = load_prompt_sensitivity_demo_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "prompt_sensitivity",
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
    module = load_prompt_sensitivity_demo_module()
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
    module = load_prompt_sensitivity_demo_module()
    args = _build_args()
    merged = module.merge_args_with_config(args)
    assert merged["model"] == "clip"
    assert merged["device"] == "auto"
    assert merged["top_k"] == 3


def test_merge_args_use_pathology_prompts_from_config(tmp_path: Path):
    module = load_prompt_sensitivity_demo_module()
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "prompt_sensitivity",
                "model": "clip",
                "device": "auto",
                "top_k": 5,
                "use_pathology_prompts": True,
            }
        ),
        encoding="utf-8",
    )

    args = _build_args(config=str(config_path), use_pathology_prompts=False)
    merged = module.merge_args_with_config(args)
    assert merged["use_pathology_prompts"] is True
