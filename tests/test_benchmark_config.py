from pathlib import Path

import pytest

from pathvlm_litebench.config import (
    BenchmarkConfig,
    PatchCoordinateHeatmapConfig,
    benchmark_config_from_dict,
    benchmark_config_to_dict,
    create_default_retrieval_config,
    load_benchmark_config,
    load_patch_coordinate_heatmap_config,
    patch_coordinate_heatmap_config_from_dict,
    patch_coordinate_heatmap_config_to_dict,
    save_benchmark_config,
    save_patch_coordinate_heatmap_config,
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


def test_zero_shot_config_roundtrip_json(tmp_path: Path):
    config = BenchmarkConfig(
        task="zero_shot",
        manifest="dataset/MHIST/manifest.csv",
        image_root="dataset/MHIST/images",
        split="test",
        class_names=["HP", "SSA"],
        class_prompts=[
            "a histopathology image of hyperplastic polyp",
            "a histopathology image of sessile serrated adenoma",
        ],
        top_k=2,
        save_report=True,
        report_dir="outputs/zero_shot_demo",
    )
    path = tmp_path / "zero_shot_config.json"
    save_benchmark_config(config, path)
    loaded = load_benchmark_config(path)
    assert loaded == config


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


def test_patch_coordinate_heatmap_config_roundtrip_json(tmp_path: Path):
    config = PatchCoordinateHeatmapConfig(
        manifest="dataset/patch_coordinates/coordinate_manifest.csv",
        score_csv="outputs/patch_coordinate_heatmap_demo/scores.csv",
        output="outputs/patch_coordinate_heatmap_demo/heatmap.png",
        title="Patch score heatmap",
    )

    data = patch_coordinate_heatmap_config_to_dict(config)
    assert data["task"] == "patch_coordinate_heatmap"

    loaded_from_dict = patch_coordinate_heatmap_config_from_dict(data)
    assert loaded_from_dict == config

    path = tmp_path / "patch_coordinate_heatmap.json"
    save_patch_coordinate_heatmap_config(config, path)
    loaded = load_patch_coordinate_heatmap_config(path)
    assert loaded == config


def test_patch_coordinate_heatmap_config_rejects_bad_align_by():
    with pytest.raises(ValueError, match="align_by"):
        PatchCoordinateHeatmapConfig(
            manifest="manifest.csv",
            score_csv="scores.csv",
            output="heatmap.png",
            align_by="case_id",
        )


def test_patch_coordinate_heatmap_config_rejects_unknown_field():
    with pytest.raises(ValueError, match="Unknown"):
        patch_coordinate_heatmap_config_from_dict(
            {
                "task": "patch_coordinate_heatmap",
                "manifest": "manifest.csv",
                "score_csv": "scores.csv",
                "output": "heatmap.png",
                "unexpected": True,
            }
        )
