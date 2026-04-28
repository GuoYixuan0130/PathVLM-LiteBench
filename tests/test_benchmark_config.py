from pathlib import Path

import pytest

from pathvlm_litebench.config import (
    BenchmarkConfig,
    benchmark_config_from_dict,
    benchmark_config_to_dict,
    create_default_retrieval_config,
    load_benchmark_config,
    save_benchmark_config,
)


def test_default_retrieval_config():
    config = create_default_retrieval_config()
    assert config.task == "retrieval"
    assert config.model == "clip"
    assert config.device == "auto"
    assert config.top_k > 0


def test_config_roundtrip_dict():
    config = BenchmarkConfig(
        task="retrieval",
        prompts=["tumor", "normal"],
        top_k=3,
    )
    data = benchmark_config_to_dict(config)
    loaded = benchmark_config_from_dict(data)
    assert loaded == config


def test_config_roundtrip_json(tmp_path: Path):
    config = BenchmarkConfig(
        task="retrieval",
        prompts=["tumor", "normal"],
        top_k=3,
    )
    path = tmp_path / "config.json"
    save_benchmark_config(config, path)
    loaded = load_benchmark_config(path)
    assert loaded == config


def test_invalid_task():
    with pytest.raises(ValueError):
        BenchmarkConfig(task="invalid")


def test_invalid_device():
    with pytest.raises(ValueError):
        BenchmarkConfig(task="retrieval", device="tpu")


def test_invalid_top_k():
    with pytest.raises(ValueError):
        BenchmarkConfig(task="retrieval", top_k=0)


def test_zero_shot_prompt_length_mismatch():
    with pytest.raises(ValueError):
        BenchmarkConfig(
            task="zero_shot",
            class_names=["tumor", "normal"],
            class_prompts=["tumor prompt"],
        )


def test_prompt_sensitivity_config_roundtrip_json(tmp_path: Path):
    config = BenchmarkConfig(
        task="prompt_sensitivity",
        image_dir="dataset/MHIST/images",
        model="clip",
        device="auto",
        concepts=["tumor", "normal", "necrosis"],
        top_k=5,
        use_pathology_prompts=True,
        save_report=True,
        report_dir="outputs/prompt_sensitivity_demo",
    )
    path = tmp_path / "prompt_sensitivity_config.json"
    save_benchmark_config(config, path)
    loaded = load_benchmark_config(path)
    assert loaded == config
