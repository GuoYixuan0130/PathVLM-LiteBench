from pathlib import Path

import pytest

from pathvlm_litebench.data import (
    coordinate_records_to_image_paths,
    load_coordinate_patch_manifest,
)


def test_load_coordinate_patch_manifest_basic(tmp_path: Path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    (patches_dir / "patch_001.png").write_text("x", encoding="utf-8")
    (patches_dir / "patch_002.png").write_text("x", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,x,y,width,height,label,split,case_id,slide_id,stain\n"
        "patches/patch_001.png,0,256,224,224,tumor,test,case_001,slide_a,H&E\n"
        "patches/patch_002.png,224,256,224,224,normal,test,case_001,slide_a,H&E\n",
        encoding="utf-8",
    )

    records = load_coordinate_patch_manifest(manifest_path)

    assert len(records) == 2
    assert records[0].x == 0
    assert records[0].y == 256
    assert records[0].width == 224
    assert records[0].height == 224
    assert records[0].label == "tumor"
    assert records[0].split == "test"
    assert records[0].case_id == "case_001"
    assert records[0].slide_id == "slide_a"
    assert records[0].metadata == {"stain": "H&E"}
    assert Path(records[0].image_path).exists()

    image_paths = coordinate_records_to_image_paths(records)
    assert image_paths == [record.image_path for record in records]


def test_load_coordinate_patch_manifest_with_image_root(tmp_path: Path):
    image_root = tmp_path / "dataset"
    patches_dir = image_root / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    (patches_dir / "patch_001.png").write_text("x", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/patch_001.png,10,20\n",
        encoding="utf-8",
    )

    records = load_coordinate_patch_manifest(manifest_path, image_root=image_root)

    assert len(records) == 1
    assert Path(records[0].image_path) == (patches_dir / "patch_001.png").resolve()


def test_load_coordinate_patch_manifest_missing_coordinate_column(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,x\n"
        "patches/patch_001.png,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Required column 'y'"):
        load_coordinate_patch_manifest(manifest_path, require_exists=False)


def test_load_coordinate_patch_manifest_invalid_coordinate(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/patch_001.png,left,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid numeric coordinate 'x'"):
        load_coordinate_patch_manifest(manifest_path, require_exists=False)


def test_load_coordinate_patch_manifest_rejects_non_positive_size(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,x,y,width\n"
        "patches/patch_001.png,0,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive finite"):
        load_coordinate_patch_manifest(manifest_path, require_exists=False)


def test_load_coordinate_patch_manifest_missing_image_respects_require_exists(
    tmp_path: Path,
):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,x,y\n"
        "patches/missing.png,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_coordinate_patch_manifest(manifest_path, require_exists=True)

    records = load_coordinate_patch_manifest(manifest_path, require_exists=False)
    assert len(records) == 1
    assert not Path(records[0].image_path).exists()
