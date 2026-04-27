from pathlib import Path

import pytest
from PIL import Image

from pathvlm_litebench.data import load_patch_images_from_paths


def test_load_patch_images_from_paths_basic(tmp_path: Path):
    path1 = tmp_path / "patch_001.png"
    path2 = tmp_path / "patch_002.jpg"

    Image.new("RGB", (32, 32), color="red").save(path1)
    Image.new("RGB", (32, 32), color="blue").save(path2)

    images, loaded_paths = load_patch_images_from_paths([path1, path2])

    assert len(images) == 2
    assert len(loaded_paths) == 2
    assert images[0].mode == "RGB"
    assert Path(loaded_paths[0]).exists()


def test_load_patch_images_from_paths_max_images(tmp_path: Path):
    path1 = tmp_path / "patch_001.png"
    path2 = tmp_path / "patch_002.jpg"

    Image.new("RGB", (32, 32), color="red").save(path1)
    Image.new("RGB", (32, 32), color="blue").save(path2)

    images, loaded_paths = load_patch_images_from_paths([path1, path2], max_images=1)

    assert len(images) == 1
    assert len(loaded_paths) == 1


def test_load_patch_images_from_paths_missing_file(tmp_path: Path):
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        load_patch_images_from_paths([missing_path])


def test_load_patch_images_from_paths_unsupported_extension(tmp_path: Path):
    txt_path = tmp_path / "patch_001.txt"
    txt_path.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError):
        load_patch_images_from_paths([txt_path])
