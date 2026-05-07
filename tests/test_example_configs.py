from pathlib import Path

from pathvlm_litebench.config import load_benchmark_config
from pathvlm_litebench.evaluation.zero_shot_grid import (
    expand_zero_shot_grid_runs,
    load_zero_shot_grid_config,
    run_zero_shot_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def test_standard_example_configs_load():
    config_names = [
        "retrieval_demo_config.json",
        "zero_shot_demo_config.json",
        "prompt_sensitivity_demo_config.json",
        "zero_shot_mhist_clip_sample.json",
        "zero_shot_mhist_plip_sample.json",
    ]

    configs = [
        load_benchmark_config(CONFIG_DIR / config_name)
        for config_name in config_names
    ]

    assert [config.task for config in configs] == [
        "retrieval",
        "zero_shot",
        "prompt_sensitivity",
        "zero_shot",
        "zero_shot",
    ]
    assert [config.model for config in configs] == [
        "clip",
        "clip",
        "clip",
        "clip",
        "plip",
    ]
    assert all(config.device == "auto" for config in configs)


def test_mhist_zero_shot_baseline_configs_are_sampled_report_configs():
    clip_config = load_benchmark_config(
        CONFIG_DIR / "zero_shot_mhist_clip_sample.json"
    )
    plip_config = load_benchmark_config(
        CONFIG_DIR / "zero_shot_mhist_plip_sample.json"
    )

    assert clip_config.task == "zero_shot"
    assert plip_config.task == "zero_shot"
    assert clip_config.model == "clip"
    assert plip_config.model == "plip"
    assert clip_config.manifest == plip_config.manifest
    assert clip_config.manifest == "dataset/MHIST/manifest_test_50_per_class.csv"
    assert clip_config.image_root == "dataset/MHIST/images"
    assert clip_config.class_names == ["HP", "SSA"]
    assert clip_config.class_prompts == plip_config.class_prompts
    assert clip_config.top_k == 2
    assert clip_config.save_report is True
    assert plip_config.save_report is True
    assert clip_config.report_dir == "outputs/zero_shot_clip_mhist_sample"
    assert plip_config.report_dir == "outputs/zero_shot_plip_mhist_sample"


def test_zero_shot_prompt_grid_example_config_expands_without_model_loading():
    config = load_zero_shot_grid_config(
        CONFIG_DIR / "zero_shot_prompt_grid_mhist_sample.json"
    )

    runs = expand_zero_shot_grid_runs(config)
    run_names = [run.run_name for run in runs]

    assert len(runs) == 9
    assert run_names == [
        "clip_default",
        "clip_diagnosis",
        "clip_patch",
        "plip_default",
        "plip_diagnosis",
        "plip_patch",
        "conch_default",
        "conch_diagnosis",
        "conch_patch",
    ]
    assert runs[0].report_dir.as_posix().endswith(
        "outputs/zero_shot_prompt_grid_mhist_sample/clip/default"
    )
    assert runs[-1].log_path is not None
    assert runs[-1].log_path.name == "run.log"


def test_zero_shot_prompt_grid_example_dry_run_does_not_write_outputs():
    config = load_zero_shot_grid_config(
        CONFIG_DIR / "zero_shot_prompt_grid_mhist_sample.json"
    )

    result = run_zero_shot_grid(config, dry_run=True)

    assert len(result["runs"]) == 9
    assert result["comparison_path"] is None
