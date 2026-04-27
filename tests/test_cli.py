from pathlib import Path

from pathvlm_litebench.cli import main


def test_cli_version(capsys):
    exit_code = main(["version"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PathVLM-LiteBench" in captured.out
    assert "0.1.0" in captured.out


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
