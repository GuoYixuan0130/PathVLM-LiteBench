import csv
from pathlib import Path

import pytest

from pathvlm_litebench.data import sample_manifest, summarize_manifest


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_sample_manifest_balanced(tmp_path: Path):
    input_csv = tmp_path / "manifest.csv"
    input_csv.write_text(
        "image_path,label,split\n"
        "a.png,HP,test\n"
        "b.png,HP,test\n"
        "c.png,SSA,test\n"
        "d.png,SSA,test\n"
        "e.png,SSA,test\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "sampled.csv"

    sample_manifest(
        input_csv=input_csv,
        output_csv=output_csv,
        split="test",
        samples_per_label=1,
        seed=42,
    )

    rows = _read_rows(output_csv)
    assert len(rows) == 2
    labels = [row["label"] for row in rows]
    assert labels.count("HP") == 1
    assert labels.count("SSA") == 1


def test_sample_manifest_split_filter(tmp_path: Path):
    input_csv = tmp_path / "manifest.csv"
    input_csv.write_text(
        "image_path,label,split\n"
        "a.png,HP,train\n"
        "b.png,HP,test\n"
        "c.png,SSA,test\n",
        encoding="utf-8",
    )
    output_csv = tmp_path / "sampled.csv"

    sample_manifest(
        input_csv=input_csv,
        output_csv=output_csv,
        split="test",
        seed=42,
    )

    rows = _read_rows(output_csv)
    assert len(rows) == 2
    assert all(row["split"] == "test" for row in rows)


def test_sample_manifest_reproducible(tmp_path: Path):
    input_csv = tmp_path / "manifest.csv"
    input_csv.write_text(
        "image_path,label,split\n"
        "a.png,HP,test\n"
        "b.png,HP,test\n"
        "c.png,SSA,test\n"
        "d.png,SSA,test\n",
        encoding="utf-8",
    )
    output_csv_1 = tmp_path / "sampled_1.csv"
    output_csv_2 = tmp_path / "sampled_2.csv"

    sample_manifest(
        input_csv=input_csv,
        output_csv=output_csv_1,
        split="test",
        samples_per_label=1,
        seed=42,
    )
    sample_manifest(
        input_csv=input_csv,
        output_csv=output_csv_2,
        split="test",
        samples_per_label=1,
        seed=42,
    )

    assert output_csv_1.read_text(encoding="utf-8") == output_csv_2.read_text(encoding="utf-8")


def test_sample_manifest_missing_label_column(tmp_path: Path):
    input_csv = tmp_path / "manifest.csv"
    input_csv.write_text(
        "image_path,split\n"
        "a.png,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        sample_manifest(
            input_csv=input_csv,
            output_csv=tmp_path / "sampled.csv",
        )


def test_sample_manifest_empty_after_split(tmp_path: Path):
    input_csv = tmp_path / "manifest.csv"
    input_csv.write_text(
        "image_path,label,split\n"
        "a.png,HP,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        sample_manifest(
            input_csv=input_csv,
            output_csv=tmp_path / "sampled.csv",
            split="test",
        )


def test_summarize_manifest(tmp_path: Path):
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "image_path,label,split\n"
        "a.png,HP,test\n"
        "b.png,HP,test\n"
        "c.png,SSA,train\n",
        encoding="utf-8",
    )

    summary = summarize_manifest(manifest_csv)
    assert summary["num_records"] == 3
    assert summary["label_distribution"]["HP"] == 2
    assert summary["label_distribution"]["SSA"] == 1
    assert summary["split_distribution"]["test"] == 2
    assert summary["split_distribution"]["train"] == 1
