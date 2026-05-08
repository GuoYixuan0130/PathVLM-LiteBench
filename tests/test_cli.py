import subprocess
import sys
from pathlib import Path

from PIL import Image
import torch

from pathvlm_litebench.cli import _apply_zero_shot_grid_overrides, main
from pathvlm_litebench.evaluation.zero_shot_grid import PromptPair, ZeroShotGridConfig


def test_cli_version(capsys):
    exit_code = main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PathVLM-LiteBench" in captured.out
    assert "0.7.0" in captured.out


def test_cli_models(capsys):
    exit_code = main(["models"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "clip" in captured.out
    assert "plip" in captured.out
    assert "conch" in captured.out


def test_cli_demos(capsys):
    exit_code = main(["demos"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "examples/01_patch_text_retrieval_demo.py" in captured.out
    assert "examples/05_patch_coordinate_heatmap_demo.py" in captured.out


def test_cli_demos_lists_zero_shot_grid_command(capsys):
    exit_code = main(["demos"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run-zero-shot-grid" in captured.out
    assert "validate-config" in captured.out
    assert "render-coordinate-heatmap" in captured.out
    assert "score-coordinate-heatmap" in captured.out


def test_cli_no_subcommand_shows_help(capsys):
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "usage:" in captured.out
    assert "pathvlm-litebench" in captured.out


def test_cli_import_does_not_load_model_dependencies():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import pathvlm_litebench.cli; "
                "print('torch' in sys.modules); "
                "print('transformers' in sys.modules)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["False", "False"]


def test_cli_validate_standard_config(tmp_path: Path, capsys):
    config_path = tmp_path / "retrieval.json"
    config_path.write_text(
        (
            '{"task": "retrieval", '
            '"model": "clip", '
            '"device": "auto", '
            '"prompts": ["tumor", "normal"], '
            '"top_k": 2}'
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Config valid: retrieval" in captured.out
    assert "Prompts: 2" in captured.out


def test_cli_validate_zero_shot_grid_config(tmp_path: Path, capsys):
    config_path = tmp_path / "grid.json"
    config_path.write_text(
        (
            '{"task": "zero_shot_grid", '
            '"models": ["clip", "plip"], '
            '"manifest": "dataset/MHIST/manifest_test_50_per_class.csv", '
            '"class_names": ["HP", "SSA"], '
            '"prompt_pairs": ['
            '{"key": "default", "class_prompts": ['
            '"a histopathology image of hyperplastic polyp", '
            '"a histopathology image of sessile serrated adenoma"]}'
            ']}'
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Config valid: zero_shot_grid" in captured.out
    assert "Models: clip, plip" in captured.out
    assert "Runs: 2" in captured.out


def test_cli_validate_patch_coordinate_heatmap_config(tmp_path: Path, capsys):
    config_path = tmp_path / "heatmap.json"
    config_path.write_text(
        (
            '{"task": "patch_coordinate_heatmap", '
            '"manifest": "dataset/patch_coordinates/coordinate_manifest.csv", '
            '"score_csv": "outputs/patch_coordinate_heatmap_demo/scores.csv", '
            '"output": "outputs/patch_coordinate_heatmap_demo/heatmap.png"}'
        ),
        encoding="utf-8",
    )

    exit_code = main(["validate-config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Config valid: patch_coordinate_heatmap" in captured.out
    assert "Align by: image_path" in captured.out


def test_cli_validate_config_rejects_bad_task(tmp_path: Path, capsys):
    config_path = tmp_path / "bad.json"
    config_path.write_text('{"task": "bad"}', encoding="utf-8")

    exit_code = main(["validate-config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.out


def test_cli_convert_manifest_mhist(tmp_path: Path, capsys):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "MHIST_aaa.png").write_text("x", encoding="utf-8")

    input_csv = tmp_path / "annotations.csv"
    input_csv.write_text(
        "Image Name,Majority Vote Label,Number of Annotators who Selected SSA (Out of 7),Partition\n"
        "MHIST_aaa.png,SSA,6,train\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "manifest.csv"
    exit_code = main(
        [
            "convert-manifest",
            "--preset",
            "mhist",
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--image_root",
            str(images_dir),
            "--require_exists",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_csv.exists()
    assert "Saved converted manifest" in captured.out


def test_cli_convert_manifest_requires_path_column_without_preset(tmp_path: Path, capsys):
    input_csv = tmp_path / "annotations.csv"
    input_csv.write_text(
        "Image Name,Majority Vote Label\n"
        "MHIST_aaa.png,SSA\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "manifest.csv"

    exit_code = main(
        [
            "convert-manifest",
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "path_column" in captured.out


def test_cli_sample_manifest(tmp_path: Path, capsys):
    input_csv = tmp_path / "manifest.csv"
    input_csv.write_text(
        "image_path,label,split\n"
        "a.png,HP,test\n"
        "b.png,HP,test\n"
        "c.png,SSA,test\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "sampled_manifest.csv"

    exit_code = main(
        [
            "sample-manifest",
            "--input",
            str(input_csv),
            "--output",
            str(output_csv),
            "--split",
            "test",
            "--samples_per_label",
            "1",
            "--seed",
            "42",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_csv.exists()
    assert "Saved sampled manifest" in captured.out
    assert "Label distribution" in captured.out


def test_cli_summarize_zero_shot_report(tmp_path: Path, capsys):
    report_dir = tmp_path / "zero_shot_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        (
            '{"metadata": {"model": "clip", "device": "cpu", "num_images": 1}, '
            '"metrics": {"error_summary": {"num_samples": 1, '
            '"predicted_label_distribution": {"HP": 1}}}}'
        ),
        encoding="utf-8",
    )
    (report_dir / "predictions.csv").write_text(
        "image_index,image_path,true_label,predicted_label,predicted_index,confidence,correct,top_predictions_json\n"
        "0,a.png,,HP,0,0.8,,[]\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.md"

    exit_code = main(
        [
            "summarize-report",
            "--task",
            "zero-shot",
            "--report_dir",
            str(report_dir),
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved experiment summary" in captured.out


def test_cli_summarize_retrieval_report(tmp_path: Path, capsys):
    report_dir = tmp_path / "retrieval_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "retrieval_metrics.json").write_text(
        (
            '{"metadata": {"model": "clip", "device": "cpu", "num_images": 2}, '
            '"metrics": {"recall_at_k": {"R@1": 0.5}, "mean_recall": 0.5}}'
        ),
        encoding="utf-8",
    )
    (report_dir / "retrieval_results.csv").write_text(
        "prompt_index,prompt,target_label,rank,image_index,image_path,score,label,is_positive\n"
        "0,prompt HP,HP,1,0,a.png,0.8,HP,True\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "retrieval_summary.md"

    exit_code = main(
        [
            "summarize-report",
            "--task",
            "retrieval",
            "--report_dir",
            str(report_dir),
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved experiment summary" in captured.out


def test_cli_summarize_prompt_sensitivity_report(tmp_path: Path, capsys):
    report_dir = tmp_path / "prompt_sensitivity_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "prompt_sensitivity_metrics.json").write_text(
        (
            '{"metadata": {"model": "clip", "device": "cpu", "num_images": 2}, '
            '"results": [{"concept_name": "tumor", "num_prompts": 2, '
            '"mean_topk_overlap": 0.5, "mean_similarity_std": 0.03}]}'
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "prompt_sensitivity_summary.md"

    exit_code = main(
        [
            "summarize-report",
            "--task",
            "prompt-sensitivity",
            "--report_dir",
            str(report_dir),
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved experiment summary" in captured.out


def test_cli_compare_zero_shot_reports(tmp_path: Path, capsys):
    first_report = tmp_path / "clip_zero_shot"
    second_report = tmp_path / "plip_zero_shot"
    first_report.mkdir(parents=True, exist_ok=True)
    second_report.mkdir(parents=True, exist_ok=True)
    metrics_text = (
        '{"metadata": {"model": "clip", "split": "test", "num_images": 1}, '
        '"metrics": {"classification_report": {"accuracy": 0.5, '
        '"balanced_accuracy": 0.5, "macro_f1": 0.5}, '
        '"error_summary": {"num_errors": 0, "error_rate": 0.0, '
        '"predicted_label_distribution": {"HP": 1}}}}'
    )
    (first_report / "metrics.json").write_text(metrics_text, encoding="utf-8")
    (second_report / "metrics.json").write_text(
        metrics_text.replace('"clip"', '"plip"'),
        encoding="utf-8",
    )
    output_path = tmp_path / "comparison_summary.md"

    exit_code = main(
        [
            "compare-reports",
            "--task",
            "zero-shot",
            "--report_dirs",
            str(first_report),
            str(second_report),
            "--run_names",
            "clip",
            "plip",
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved comparison summary" in captured.out
    assert "Zero-Shot Comparison Summary" in output_path.read_text(encoding="utf-8")


def test_cli_compare_reports_rejects_mismatched_run_names(tmp_path: Path, capsys):
    report_dir = tmp_path / "zero_shot"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(
        '{"metadata": {"model": "clip"}, "metrics": {}}',
        encoding="utf-8",
    )

    exit_code = main(
        [
            "compare-reports",
            "--task",
            "zero-shot",
            "--report_dirs",
            str(report_dir),
            "--run_names",
            "a",
            "b",
            "--output",
            str(tmp_path / "comparison.md"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "run_names" in captured.out


def test_cli_zero_shot_grid_dry_run(tmp_path: Path, capsys):
    config_path = tmp_path / "grid.json"
    config_path.write_text(
        (
            '{"task": "zero_shot_grid", '
            '"models": ["clip"], '
            '"manifest": "dataset/MHIST/manifest_test_50_per_class.csv", '
            '"image_root": "dataset/MHIST/images", '
            '"class_names": ["HP", "SSA"], '
            '"prompt_pairs": ['
            '{"key": "default", "class_prompts": ['
            '"a histopathology image of hyperplastic polyp", '
            '"a histopathology image of sessile serrated adenoma"]}'
            '], '
            '"output_root": "outputs/test_grid"}'
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run-zero-shot-grid",
            "--config",
            str(config_path),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Zero-shot grid runs: 1" in captured.out
    assert "Dry run only" in captured.out


def test_cli_zero_shot_grid_dry_run_allows_output_root_override(tmp_path: Path, capsys):
    config_path = tmp_path / "grid.json"
    override_root = tmp_path / "override_grid"
    config_path.write_text(
        (
            '{"task": "zero_shot_grid", '
            '"models": ["clip"], '
            '"manifest": "dataset/MHIST/manifest_test_50_per_class.csv", '
            '"image_root": "dataset/MHIST/images", '
            '"class_names": ["HP", "SSA"], '
            '"prompt_pairs": ['
            '{"key": "default", "class_prompts": ['
            '"a histopathology image of hyperplastic polyp", '
            '"a histopathology image of sessile serrated adenoma"]}'
            '], '
            '"output_root": "outputs/original_grid", '
            '"comparison_output": "outputs/original_grid/comparison.md"}'
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run-zero-shot-grid",
            "--config",
            str(config_path),
            "--output-root",
            str(override_root),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert str(override_root / "clip" / "default") in captured.out
    assert "outputs/original_grid" not in captured.out


def test_zero_shot_grid_output_root_override_resets_comparison_output():
    config = ZeroShotGridConfig(
        models=["clip"],
        class_names=["HP", "SSA"],
        prompt_pairs=[
            PromptPair(
                key="default",
                class_prompts=[
                    "a histopathology image of hyperplastic polyp",
                    "a histopathology image of sessile serrated adenoma",
                ],
            )
        ],
        manifest="dataset/MHIST/manifest_test_50_per_class.csv",
        output_root="outputs/original_grid",
        comparison_output="outputs/original_grid/comparison.md",
    )

    updated = _apply_zero_shot_grid_overrides(
        config,
        output_root="outputs/new_grid",
    )

    assert updated.output_root == "outputs/new_grid"
    assert updated.comparison_output is None


def test_zero_shot_grid_comparison_output_override_keeps_config_output_root():
    config = ZeroShotGridConfig(
        models=["clip"],
        class_names=["HP", "SSA"],
        prompt_pairs=[
            PromptPair(
                key="default",
                class_prompts=[
                    "a histopathology image of hyperplastic polyp",
                    "a histopathology image of sessile serrated adenoma",
                ],
            )
        ],
        manifest="dataset/MHIST/manifest_test_50_per_class.csv",
        output_root="outputs/original_grid",
        comparison_output="outputs/original_grid/comparison.md",
    )

    updated = _apply_zero_shot_grid_overrides(
        config,
        comparison_output="outputs/new_comparison.md",
    )

    assert updated.output_root == "outputs/original_grid"
    assert updated.comparison_output == "outputs/new_comparison.md"


def test_cli_render_coordinate_heatmap_aligns_by_image_path(tmp_path: Path, capsys):
    manifest_path = tmp_path / "coordinate_manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/a.png,0,0\n"
        "patches/b.png,224,0\n",
        encoding="utf-8",
    )

    first_path = (tmp_path / "patches" / "a.png").resolve()
    second_path = (tmp_path / "patches" / "b.png").resolve()
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text(
        "image_path,score\n"
        f"{second_path},0.8\n"
        f"{first_path},0.2\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "heatmap.png"

    exit_code = main(
        [
            "render-coordinate-heatmap",
            "--manifest",
            str(manifest_path),
            "--score-csv",
            str(score_csv),
            "--output",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Saved patch-coordinate heatmap" in captured.out
    assert "Patches: 2" in captured.out
    assert "Grid shape: 1 rows x 2 columns" in captured.out


def test_cli_render_coordinate_heatmap_aligns_by_order(tmp_path: Path, capsys):
    manifest_path = tmp_path / "coordinate_manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/a.png,0,0\n"
        "patches/b.png,0,224\n",
        encoding="utf-8",
    )
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text(
        "score\n"
        "0.2\n"
        "0.8\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "heatmap.png"

    exit_code = main(
        [
            "render-coordinate-heatmap",
            "--manifest",
            str(manifest_path),
            "--score-csv",
            str(score_csv),
            "--output",
            str(output_path),
            "--align-by",
            "order",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Grid shape: 2 rows x 1 columns" in captured.out


def test_cli_render_coordinate_heatmap_rejects_missing_score_column(
    tmp_path: Path,
    capsys,
):
    manifest_path = tmp_path / "coordinate_manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/a.png,0,0\n",
        encoding="utf-8",
    )
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text("image_path,value\npatches/a.png,0.2\n", encoding="utf-8")

    exit_code = main(
        [
            "render-coordinate-heatmap",
            "--manifest",
            str(manifest_path),
            "--score-csv",
            str(score_csv),
            "--output",
            str(tmp_path / "heatmap.png"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Score column 'score' not found" in captured.out


def test_cli_render_coordinate_heatmap_requires_inputs_without_config(capsys):
    exit_code = main(["render-coordinate-heatmap"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--manifest is required" in captured.out


def test_cli_render_coordinate_heatmap_uses_config(tmp_path: Path, capsys):
    manifest_path = tmp_path / "coordinate_manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/a.png,0,0\n",
        encoding="utf-8",
    )
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text(
        "score\n"
        "0.4\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "heatmap.png"
    config_path = tmp_path / "heatmap_config.json"
    config_path.write_text(
        (
            '{"task": "patch_coordinate_heatmap", '
            f'"manifest": "{manifest_path.as_posix()}", '
            f'"score_csv": "{score_csv.as_posix()}", '
            f'"output": "{output_path.as_posix()}", '
            '"align_by": "order"}'
        ),
        encoding="utf-8",
    )

    exit_code = main(["render-coordinate-heatmap", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Patches: 1" in captured.out


def test_cli_render_coordinate_heatmap_config_allows_output_override(
    tmp_path: Path,
    capsys,
):
    manifest_path = tmp_path / "coordinate_manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/a.png,0,0\n",
        encoding="utf-8",
    )
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text("score\n0.4\n", encoding="utf-8")
    config_output = tmp_path / "config_heatmap.png"
    override_output = tmp_path / "override_heatmap.png"
    config_path = tmp_path / "heatmap_config.json"
    config_path.write_text(
        (
            '{"task": "patch_coordinate_heatmap", '
            f'"manifest": "{manifest_path.as_posix()}", '
            f'"score_csv": "{score_csv.as_posix()}", '
            f'"output": "{config_output.as_posix()}", '
            '"align_by": "order"}'
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "render-coordinate-heatmap",
            "--config",
            str(config_path),
            "--output",
            str(override_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert override_output.exists()
    assert not config_output.exists()
    assert str(override_output) in captured.out


def test_cli_score_coordinate_heatmap_uses_fake_model(
    tmp_path: Path,
    capsys,
    monkeypatch,
):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color="red").save(patches_dir / "a.png")
    Image.new("RGB", (16, 16), color="blue").save(patches_dir / "b.png")

    manifest_path = tmp_path / "coordinate_manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/a.png,0,0\n"
        "patches/b.png,224,0\n",
        encoding="utf-8",
    )

    class FakeModel:
        def encode_images(self, images):
            assert len(images) == 2
            return torch.tensor([[1.0, 0.0], [0.25, 0.75]])

        def encode_text(self, texts):
            assert texts == ["synthetic red score"]
            return torch.tensor([[1.0, 0.0]])

    def fake_create_model(model_key_or_name, device=None):
        assert model_key_or_name == "clip"
        assert device == "cpu"
        return FakeModel()

    import pathvlm_litebench.models

    monkeypatch.setattr(pathvlm_litebench.models, "create_model", fake_create_model)

    output_dir = tmp_path / "scored"
    exit_code = main(
        [
            "score-coordinate-heatmap",
            "--manifest",
            str(manifest_path),
            "--prompt",
            "synthetic red score",
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (output_dir / "scores.csv").exists()
    assert (output_dir / "heatmap.png").exists()
    scores_text = (output_dir / "scores.csv").read_text(encoding="utf-8")
    assert "synthetic red score" in scores_text
    assert "1.0" in scores_text
    assert "0.25" in scores_text
    assert "Saved patch-coordinate scores" in captured.out
