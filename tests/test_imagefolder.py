import csv
from pathlib import Path

import pytest
from PIL import Image

from pathvlm_litebench.cli import main
from pathvlm_litebench.data import build_imagefolder_manifest, load_patch_manifest


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def _build_flat_tree(root: Path) -> None:
    _make_image(root / "tumor" / "a.png")
    _make_image(root / "tumor" / "b.png")
    _make_image(root / "normal" / "c.png")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def test_build_flat_layout(tmp_path: Path):
    image_dir = tmp_path / "imagefolder"
    _build_flat_tree(image_dir)
    output_csv = tmp_path / "manifest.csv"

    summary = build_imagefolder_manifest(image_dir, output_csv)

    assert summary["num_records"] == 3
    assert summary["num_classes"] == 2
    assert summary["label_distribution"] == {"normal": 1, "tumor": 2}
    assert summary["split_distribution"] == {}

    fieldnames, rows = _read_csv(output_csv)
    assert fieldnames == ["image_path", "label", "split"]
    assert {row["label"] for row in rows} == {"tumor", "normal"}
    assert all(row["split"] == "" for row in rows)
    assert all(Path(row["image_path"]).is_absolute() for row in rows)


def test_build_split_layout(tmp_path: Path):
    image_dir = tmp_path / "imagefolder"
    _make_image(image_dir / "train" / "tumor" / "a.png")
    _make_image(image_dir / "train" / "normal" / "b.png")
    _make_image(image_dir / "test" / "tumor" / "c.png")
    output_csv = tmp_path / "manifest.csv"

    summary = build_imagefolder_manifest(image_dir, output_csv, has_split=True)

    assert summary["num_records"] == 3
    assert summary["num_classes"] == 2
    assert summary["split_distribution"] == {"test": 1, "train": 2}

    _, rows = _read_csv(output_csv)
    splits = {row["split"] for row in rows}
    assert splits == {"train", "test"}


def test_build_relative_paths(tmp_path: Path):
    image_dir = tmp_path / "imagefolder"
    _build_flat_tree(image_dir)
    output_csv = tmp_path / "out" / "manifest.csv"

    build_imagefolder_manifest(image_dir, output_csv, relative=True)

    _, rows = _read_csv(output_csv)
    assert all(not Path(row["image_path"]).is_absolute() for row in rows)

    records = load_patch_manifest(output_csv, require_exists=True)
    assert len(records) == 3


def test_build_filters_extensions(tmp_path: Path):
    image_dir = tmp_path / "imagefolder"
    _make_image(image_dir / "tumor" / "a.png")
    _make_image(image_dir / "tumor" / "b.jpg")
    output_csv = tmp_path / "manifest.csv"

    summary = build_imagefolder_manifest(image_dir, output_csv, extensions=["png"])

    assert summary["num_records"] == 1
    assert summary["label_distribution"] == {"tumor": 1}


def test_build_missing_directory(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_imagefolder_manifest(tmp_path / "missing", tmp_path / "manifest.csv")


def test_build_no_class_dirs(tmp_path: Path):
    image_dir = tmp_path / "imagefolder"
    image_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No class subdirectories"):
        build_imagefolder_manifest(image_dir, tmp_path / "manifest.csv")


def test_build_no_images(tmp_path: Path):
    image_dir = tmp_path / "imagefolder"
    (image_dir / "tumor").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No images"):
        build_imagefolder_manifest(image_dir, tmp_path / "manifest.csv")


def test_build_split_missing_split_dirs(tmp_path: Path):
    image_dir = tmp_path / "imagefolder"
    image_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="No split directories"):
        build_imagefolder_manifest(image_dir, tmp_path / "manifest.csv", has_split=True)


def test_cli_build_imagefolder_manifest(tmp_path: Path, capsys):
    image_dir = tmp_path / "imagefolder"
    _build_flat_tree(image_dir)
    output_csv = tmp_path / "manifest.csv"

    exit_code = main(
        [
            "build-imagefolder-manifest",
            "--image-dir",
            str(image_dir),
            "--output",
            str(output_csv),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_csv.exists()
    assert "Saved imagefolder manifest to:" in captured.out
    assert "Number of records: 3" in captured.out

    records = load_patch_manifest(output_csv, require_exists=True)
    assert len(records) == 3


def test_cli_build_imagefolder_manifest_missing_dir(tmp_path: Path, capsys):
    exit_code = main(
        [
            "build-imagefolder-manifest",
            "--image-dir",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "manifest.csv"),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.out
