from pathlib import Path

import pytest

from pathvlm_litebench.data import (
    filter_records_by_label,
    filter_records_by_split,
    get_unique_labels,
    load_patch_manifest,
    records_to_image_paths,
    records_to_labels,
)


def test_load_patch_manifest_basic(tmp_path: Path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    (patches_dir / "patch_001.png").write_text("x", encoding="utf-8")
    (patches_dir / "patch_002.png").write_text("x", encoding="utf-8")
    (patches_dir / "patch_003.png").write_text("x", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,label,split,case_id\n"
        "patches/patch_001.png,tumor,train,case_001\n"
        "patches/patch_002.png,normal,train,case_001\n"
        "patches/patch_003.png,tumor,test,case_002\n",
        encoding="utf-8",
    )

    records = load_patch_manifest(manifest_path)

    assert len(records) == 3
    assert records[0].label == "tumor"
    assert records[0].split == "train"
    assert records[0].case_id == "case_001"
    assert records[0].slide_id is None
    assert Path(records[0].image_path).exists()

    image_paths = records_to_image_paths(records)
    labels = records_to_labels(records)

    assert len(image_paths) == 3
    assert labels == ["tumor", "normal", "tumor"]


def test_get_unique_labels_and_filters(tmp_path: Path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    for file_name in ("patch_001.png", "patch_002.png", "patch_003.png"):
        (patches_dir / file_name).write_text("x", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,label,split,case_id\n"
        "patches/patch_001.png,tumor,train,case_001\n"
        "patches/patch_002.png,normal,train,case_001\n"
        "patches/patch_003.png,tumor,test,case_002\n",
        encoding="utf-8",
    )

    records = load_patch_manifest(manifest_path)

    assert get_unique_labels(records) == ["normal", "tumor"]
    assert len(filter_records_by_split(records, "train")) == 2
    assert len(filter_records_by_split(records, "test")) == 1
    assert len(filter_records_by_label(records, "tumor")) == 2
    assert len(filter_records_by_label(records, "normal")) == 1


def test_missing_manifest_raises_file_not_found(tmp_path: Path):
    missing_manifest = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_patch_manifest(missing_manifest)


def test_missing_path_column_raises_value_error(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "path,label\n"
        "patches/patch_001.png,tumor\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_patch_manifest(manifest_path)


def test_empty_manifest_rows_raise_value_error(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("image_path,label\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_patch_manifest(manifest_path)


def test_missing_image_with_require_exists_true_raises(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,label\n"
        "patches/missing.png,tumor\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_patch_manifest(manifest_path, require_exists=True)


def test_missing_image_with_require_exists_false(tmp_path: Path):
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,label\n"
        "patches/missing.png,tumor\n",
        encoding="utf-8",
    )

    records = load_patch_manifest(manifest_path, require_exists=False)
    assert len(records) == 1
    assert records[0].label == "tumor"
    assert not Path(records[0].image_path).exists()


def test_optional_columns_missing(tmp_path: Path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    (patches_dir / "patch_001.png").write_text("x", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path\n"
        "patches/patch_001.png\n",
        encoding="utf-8",
    )

    records = load_patch_manifest(manifest_path)
    assert len(records) == 1
    assert records[0].label is None
    assert records[0].split is None
    assert records[0].case_id is None
    assert records[0].slide_id is None


def test_metadata_columns_preserved(tmp_path: Path):
    patches_dir = tmp_path / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    (patches_dir / "patch_001.png").write_text("x", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,label,magnification,stain\n"
        "patches/patch_001.png,tumor,20x,H&E\n",
        encoding="utf-8",
    )

    records = load_patch_manifest(manifest_path)
    assert len(records) == 1
    assert records[0].metadata is not None
    assert records[0].metadata["magnification"] == "20x"
    assert records[0].metadata["stain"] == "H&E"


def test_load_manifest_with_image_root(tmp_path: Path):
    image_root = tmp_path / "dataset_root"
    patches_dir = image_root / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    (patches_dir / "patch_001.png").write_text("x", encoding="utf-8")

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "image_path,label\n"
        "patches/patch_001.png,tumor\n",
        encoding="utf-8",
    )

    records = load_patch_manifest(manifest_path, image_root=image_root)
    assert len(records) == 1
    assert Path(records[0].image_path) == (patches_dir / "patch_001.png").resolve()
