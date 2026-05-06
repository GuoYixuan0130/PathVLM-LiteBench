import subprocess
import sys
from pathlib import Path

from pathvlm_litebench.cli import main


def test_cli_version(capsys):
    exit_code = main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PathVLM-LiteBench" in captured.out
    assert "0.3.0" in captured.out


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
