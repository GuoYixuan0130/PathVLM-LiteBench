import json
from pathlib import Path

import pytest

from pathvlm_litebench.evaluation.zero_shot_grid import (
    build_zero_shot_grid_command,
    expand_zero_shot_grid_runs,
    load_zero_shot_grid_config,
    run_zero_shot_grid,
)


def _write_grid_config(path: Path, output_root: Path) -> None:
    payload = {
        "task": "zero_shot_grid",
        "models": ["clip", "plip"],
        "device": "cpu",
        "manifest": "dataset/MHIST/manifest_test_50_per_class.csv",
        "image_root": "dataset/MHIST/images",
        "split": "test",
        "class_names": ["HP", "SSA"],
        "prompt_pairs": [
            {
                "key": "default",
                "class_prompts": [
                    "a histopathology image of hyperplastic polyp",
                    "a histopathology image of sessile serrated adenoma",
                ],
            },
            {
                "key": "patch",
                "class_prompts": [
                    "a pathology patch showing hyperplastic polyp",
                    "a pathology patch showing sessile serrated adenoma",
                ],
            },
        ],
        "top_k": 2,
        "output_root": str(output_root),
        "save_comparison": True,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_and_expand_zero_shot_grid_config(tmp_path: Path):
    config_path = tmp_path / "grid.json"
    _write_grid_config(config_path, tmp_path / "outputs")

    config = load_zero_shot_grid_config(config_path)
    runs = expand_zero_shot_grid_runs(config)

    assert len(runs) == 4
    assert [run.run_name for run in runs] == [
        "clip_default",
        "clip_patch",
        "plip_default",
        "plip_patch",
    ]
    assert runs[0].report_dir == tmp_path / "outputs" / "clip" / "default"
    assert runs[0].class_prompts[0].startswith("a histopathology")


def test_build_zero_shot_grid_command_keeps_prompt_strings(tmp_path: Path):
    config_path = tmp_path / "grid.json"
    _write_grid_config(config_path, tmp_path / "outputs")
    run = expand_zero_shot_grid_runs(load_zero_shot_grid_config(config_path))[0]

    command = build_zero_shot_grid_command(run)

    assert "--class_prompts" in command
    prompt_index = command.index("--class_prompts") + 1
    assert command[prompt_index] == "a histopathology image of hyperplastic polyp"
    assert command[prompt_index + 1] == "a histopathology image of sessile serrated adenoma"


def test_zero_shot_grid_rejects_prompt_length_mismatch(tmp_path: Path):
    config_path = tmp_path / "grid.json"
    config_path.write_text(
        json.dumps(
            {
                "task": "zero_shot_grid",
                "models": ["clip"],
                "manifest": "manifest.csv",
                "class_names": ["HP", "SSA"],
                "prompt_pairs": [
                    {
                        "key": "bad",
                        "class_prompts": ["one prompt"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="one prompt per class name"):
        load_zero_shot_grid_config(config_path)


def test_run_zero_shot_grid_with_fake_runner_writes_comparison(tmp_path: Path):
    config_path = tmp_path / "grid.json"
    _write_grid_config(config_path, tmp_path / "outputs")
    config = load_zero_shot_grid_config(config_path)

    def fake_runner(run):
        run.report_dir.mkdir(parents=True, exist_ok=True)
        (run.report_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "model": run.model,
                        "split": run.split,
                        "num_images": 2,
                    },
                    "metrics": {
                        "classification_report": {
                            "accuracy": 0.5,
                            "balanced_accuracy": 0.5,
                            "macro_f1": 0.5,
                        },
                        "error_summary": {
                            "num_errors": 1,
                            "error_rate": 0.5,
                            "predicted_label_distribution": {"HP": 1, "SSA": 1},
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (run.report_dir / "predictions.csv").write_text("x\n", encoding="utf-8")
        (run.report_dir / "errors.csv").write_text("x\n", encoding="utf-8")

    result = run_zero_shot_grid(config, runner=fake_runner)

    comparison_path = Path(result["comparison_path"])
    assert comparison_path.exists()
    assert "Zero-Shot Comparison Summary" in comparison_path.read_text(encoding="utf-8")
