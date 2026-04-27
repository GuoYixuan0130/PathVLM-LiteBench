import csv
from pathlib import Path

import pytest

from pathvlm_litebench.data import convert_manifest, convert_mhist_manifest


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def test_convert_manifest_basic(tmp_path: Path):
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / "patch_001.png").write_text("x", encoding="utf-8")
    (images_dir / "patch_002.png").write_text("x", encoding="utf-8")

    input_csv = tmp_path / "annotations.csv"
    input_csv.write_text(
        "Image Name,Majority Vote Label,Partition\n"
        "patch_001.png,tumor,train\n"
        "patch_002.png,normal,test\n",
        encoding="utf-8",
    )

    output_csv = tmp_path / "manifest.csv"
    convert_manifest(
        input_csv=input_csv,
        output_csv=output_csv,
        path_column="Image Name",
        label_column="Majority Vote Label",
        split_column="Partition",
        image_root=images_dir,
        require_exists=True,
    )

    fieldnames, rows = _read_csv(output_csv)
    assert fieldnames[:5] == ["image_path", "label", "split", "case_id", "slide_id"]
    assert len(rows) == 2
    assert rows[0]["image_path"] == "patch_001.png"
    assert rows[0]["label"] == "tumor"
    assert rows[0]["split"] == "train"
    assert rows[0]["case_id"] == "patch_001"
    assert rows[0]["slide_id"] == ""


def test_convert_mhist_manifest(tmp_path: Path):
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
    convert_mhist_manifest(
        annotations_csv=input_csv,
        output_csv=output_csv,
        image_root=images_dir,
        require_exists=True,
    )

    fieldnames, rows = _read_csv(output_csv)
    assert len(rows) == 1
    assert rows[0]["image_path"] == "MHIST_aaa.png"
    assert rows[0]["label"] == "SSA"
    assert rows[0]["split"] == "train"
    assert rows[0]["case_id"] == "MHIST_aaa"
    assert "Number of Annotators who Selected SSA (Out of 7)" in fieldnames
    assert rows[0]["Number of Annotators who Selected SSA (Out of 7)"] == "6"


def test_missing_input_csv(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        convert_manifest(
            input_csv=tmp_path / "missing.csv",
            output_csv=tmp_path / "manifest.csv",
            path_column="Image Name",
        )


def test_missing_path_column(tmp_path: Path):
    input_csv = tmp_path / "annotations.csv"
    input_csv.write_text(
        "Label,Partition\n"
        "tumor,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        convert_manifest(
            input_csv=input_csv,
            output_csv=tmp_path / "manifest.csv",
            path_column="Image Name",
        )


def test_require_exists_missing_image(tmp_path: Path):
    input_csv = tmp_path / "annotations.csv"
    input_csv.write_text(
        "Image Name,Majority Vote Label,Partition\n"
        "patch_001.png,tumor,train\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        convert_manifest(
            input_csv=input_csv,
            output_csv=tmp_path / "manifest.csv",
            path_column="Image Name",
            label_column="Majority Vote Label",
            split_column="Partition",
            image_root=tmp_path / "images",
            require_exists=True,
        )


def test_empty_csv(tmp_path: Path):
    input_csv = tmp_path / "annotations.csv"
    input_csv.write_text("Image Name,Majority Vote Label,Partition\n", encoding="utf-8")

    with pytest.raises(ValueError):
        convert_manifest(
            input_csv=input_csv,
            output_csv=tmp_path / "manifest.csv",
            path_column="Image Name",
        )
