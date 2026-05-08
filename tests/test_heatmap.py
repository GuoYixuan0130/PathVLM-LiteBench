import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pathvlm_litebench.data.coordinate_manifest import CoordinatePatchRecord
from pathvlm_litebench.visualization import (
    aggregate_patch_scores_to_grid,
    save_patch_scores_csv,
    save_score_heatmap,
)


def _record(x: float, y: float, image_path: str = "patch.png") -> CoordinatePatchRecord:
    return CoordinatePatchRecord(image_path=image_path, x=x, y=y, slide_id="slide_a")


def test_aggregate_patch_scores_to_grid_basic():
    records = [
        _record(0, 0, "a.png"),
        _record(224, 0, "b.png"),
        _record(0, 224, "c.png"),
    ]
    scores = [0.1, 0.9, 0.5]

    grid = aggregate_patch_scores_to_grid(records, scores)

    assert grid.x_values == [0, 224]
    assert grid.y_values == [0, 224]
    np.testing.assert_allclose(grid.values[0, 0], 0.1)
    np.testing.assert_allclose(grid.values[0, 1], 0.9)
    np.testing.assert_allclose(grid.values[1, 0], 0.5)
    assert np.isnan(grid.values[1, 1])
    assert grid.counts.tolist() == [[1, 1], [1, 0]]


def test_aggregate_patch_scores_to_grid_averages_duplicate_coordinates():
    records = [
        _record(0, 0, "a.png"),
        _record(0, 0, "b.png"),
        _record(224, 0, "c.png"),
    ]
    scores = [0.2, 0.6, 1.0]

    grid = aggregate_patch_scores_to_grid(records, scores)

    np.testing.assert_allclose(grid.values[0, 0], 0.4)
    np.testing.assert_allclose(grid.values[0, 1], 1.0)
    assert grid.counts.tolist() == [[2, 1]]


def test_aggregate_patch_scores_to_grid_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        aggregate_patch_scores_to_grid([_record(0, 0)], [])


def test_aggregate_patch_scores_to_grid_rejects_empty_records():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_patch_scores_to_grid([], [])


def test_aggregate_patch_scores_to_grid_rejects_non_finite_score():
    with pytest.raises(ValueError, match="not finite"):
        aggregate_patch_scores_to_grid([_record(0, 0)], [float("nan")])


def test_save_score_heatmap_creates_png(tmp_path: Path):
    grid = aggregate_patch_scores_to_grid(
        [_record(0, 0), _record(224, 0), _record(0, 224)],
        [0.1, 0.9, 0.5],
    )

    output_path = tmp_path / "reports" / "heatmap.png"
    saved_path = save_score_heatmap(grid, output_path, title="Tumor prompt")

    assert Path(saved_path).exists()
    with Image.open(saved_path) as image:
        assert image.format == "PNG"
        assert image.size[0] > 0
        assert image.size[1] > 0


def test_save_patch_scores_csv(tmp_path: Path):
    records = [
        CoordinatePatchRecord(
            image_path="a.png",
            x=0,
            y=0,
            width=224,
            height=224,
            label="tumor",
            slide_id="slide_a",
        ),
        CoordinatePatchRecord(
            image_path="b.png",
            x=224,
            y=0,
        ),
    ]

    output_path = tmp_path / "reports" / "scores.csv"
    saved_path = save_patch_scores_csv(
        records,
        [0.25, 0.75],
        output_path,
        prompt="a histopathology image of tumor tissue",
    )

    with Path(saved_path).open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 2
    assert rows[0]["image_path"] == "a.png"
    assert rows[0]["slide_id"] == "slide_a"
    assert rows[0]["x"] == "0"
    assert rows[0]["width"] == "224"
    assert rows[0]["label"] == "tumor"
    assert rows[0]["score"] == "0.25"
    assert rows[0]["prompt"] == "a histopathology image of tumor tissue"
    assert rows[1]["width"] == ""
