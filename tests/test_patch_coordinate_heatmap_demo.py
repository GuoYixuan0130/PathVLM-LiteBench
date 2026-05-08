import csv
import importlib.util
from pathlib import Path

from PIL import Image


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "05_patch_coordinate_heatmap_demo.py"
)


def _load_demo_module():
    spec = importlib.util.spec_from_file_location(
        "patch_coordinate_heatmap_demo",
        EXAMPLE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthetic_patch_coordinate_heatmap_demo_writes_artifacts(tmp_path: Path):
    module = _load_demo_module()

    saved_paths = module.create_synthetic_coordinate_demo(tmp_path / "demo")

    manifest_path = Path(saved_paths["manifest"])
    scores_path = Path(saved_paths["scores"])
    heatmap_path = Path(saved_paths["heatmap"])

    assert manifest_path.exists()
    assert scores_path.exists()
    assert heatmap_path.exists()

    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_file:
        manifest_rows = list(csv.DictReader(manifest_file))
    with scores_path.open("r", encoding="utf-8", newline="") as scores_file:
        score_rows = list(csv.DictReader(scores_file))

    assert len(manifest_rows) == 6
    assert len(score_rows) == 6
    assert manifest_rows[0]["image_path"] == "patches/patch_00.png"
    assert score_rows[0]["score"] == "0.1"

    with Image.open(heatmap_path) as image:
        assert image.format == "PNG"
        assert image.size[0] > 0
        assert image.size[1] > 0
