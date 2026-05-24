import csv
import json
from pathlib import Path

import pytest

from pathvlm_litebench.visualization import (
    build_patch_score_comparison_summary,
    compare_patch_score_csvs,
    save_patch_score_comparison_csv,
    save_patch_score_comparison_summary,
    summarize_patch_score_csv,
)


def _write_score_artifacts(
    run_dir: Path,
    *,
    scores: list[float],
    prompt: str,
    model: str = "clip",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    score_csv = run_dir / "scores.csv"
    score_csv.write_text(
        "image_path,x,y,score,prompt\n"
        + "\n".join(
            f"patch_{idx}.png,{idx * 224},0,{score},{prompt}"
            for idx, score in enumerate(scores)
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "task": "patch_coordinate_heatmap_scoring",
        "version": "0.9.0.dev0",
        "created_at_utc": "2026-05-24T00:00:00Z",
        "manifest": "dataset/patch_coordinates/coordinate_manifest.csv",
        "prompt": prompt,
        "model": model,
        "device": "cpu",
        "heatmap_output": str(run_dir / "heatmap.png"),
        "patch_count": len(scores),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    return score_csv


def test_summarize_patch_score_csv_loads_sibling_metadata(tmp_path: Path):
    score_csv = _write_score_artifacts(
        tmp_path / "tumor_run",
        scores=[0.2, 0.6],
        prompt="tumor prompt",
    )

    summary = summarize_patch_score_csv(score_csv)

    assert summary.run_name == "tumor_run"
    assert summary.prompt == "tumor prompt"
    assert summary.model == "clip"
    assert summary.device == "cpu"
    assert summary.row_count == 2
    assert summary.metadata_patch_count == 2
    assert summary.score_mean == pytest.approx(0.4)
    assert summary.score_std == pytest.approx(0.2)
    assert summary.score_min == pytest.approx(0.2)
    assert summary.score_max == pytest.approx(0.6)


def test_compare_patch_score_csvs_saves_csv_and_markdown(tmp_path: Path):
    tumor_csv = _write_score_artifacts(
        tmp_path / "tumor_run",
        scores=[0.2, 0.6],
        prompt="tumor prompt",
    )
    lymphocyte_csv = _write_score_artifacts(
        tmp_path / "lymphocyte_run",
        scores=[0.4, 0.8],
        prompt="lymphocyte prompt",
    )

    summaries = compare_patch_score_csvs(
        [tumor_csv, lymphocyte_csv],
        run_names=["tumor", "lymphocyte"],
    )
    output_csv = tmp_path / "comparison" / "score_summary.csv"
    output_md = tmp_path / "comparison" / "score_summary.md"

    saved_csv = save_patch_score_comparison_csv(summaries, output_csv)
    saved_md = save_patch_score_comparison_summary(summaries, output_md)

    with Path(saved_csv).open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert Path(saved_md).exists()
    assert [row["run_name"] for row in rows] == ["tumor", "lymphocyte"]
    assert rows[0]["score_mean"] == "0.4"
    assert rows[1]["score_max"] == "0.8"
    markdown = output_md.read_text(encoding="utf-8")
    assert "# Patch-Coordinate Score Comparison" in markdown
    assert "tumor prompt" in markdown
    assert "artifact-only" in markdown


def test_build_patch_score_comparison_summary_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one"):
        build_patch_score_comparison_summary([])


def test_compare_patch_score_csvs_rejects_missing_score_column(tmp_path: Path):
    score_csv = tmp_path / "scores.csv"
    score_csv.write_text("image_path,value\npatch.png,0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Score column 'score' not found"):
        summarize_patch_score_csv(score_csv)


def test_compare_patch_score_csvs_rejects_row_count_mismatch(tmp_path: Path):
    first_csv = _write_score_artifacts(
        tmp_path / "first_run",
        scores=[0.2, 0.6],
        prompt="first prompt",
    )
    second_csv = _write_score_artifacts(
        tmp_path / "second_run",
        scores=[0.2],
        prompt="second prompt",
    )

    with pytest.raises(ValueError, match="same row count"):
        compare_patch_score_csvs([first_csv, second_csv])


def test_compare_patch_score_csvs_allows_row_count_mismatch(tmp_path: Path):
    first_csv = _write_score_artifacts(
        tmp_path / "first_run",
        scores=[0.2, 0.6],
        prompt="first prompt",
    )
    second_csv = _write_score_artifacts(
        tmp_path / "second_run",
        scores=[0.2],
        prompt="second prompt",
    )

    summaries = compare_patch_score_csvs(
        [first_csv, second_csv],
        require_equal_row_count=False,
    )

    assert [summary.row_count for summary in summaries] == [2, 1]


def test_compare_patch_score_csvs_rejects_metadata_patch_count_mismatch(
    tmp_path: Path,
):
    score_csv = _write_score_artifacts(
        tmp_path / "bad_metadata",
        scores=[0.2, 0.6],
        prompt="bad metadata prompt",
    )
    metadata_path = score_csv.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["patch_count"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="patch_count"):
        summarize_patch_score_csv(score_csv)
