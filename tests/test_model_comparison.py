import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import pathvlm_litebench.models as models_module
from pathvlm_litebench.cli import main
from pathvlm_litebench.evaluation.model_comparison import (
    ModelZeroShotResult,
    evaluate_models_zero_shot,
    resolve_true_indices,
)
from pathvlm_litebench.visualization.model_comparison_report import (
    compute_model_accuracy_cis,
    save_model_comparison_chart,
    save_model_comparison_csv,
    save_model_comparison_per_class_csv,
)

CLASS_NAMES = ["adipose", "muscle", "tumor"]


class _PerfectModel:
    """Encodes each image (an int index) to a one-hot matching the true class."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def encode_images(self, images, batch_size=32, show_progress=False):
        return torch.nn.functional.one_hot(
            torch.tensor(list(images)), num_classes=self.num_classes
        ).float()

    def encode_text(self, prompts):
        return torch.eye(len(prompts))


class _ShiftedModel(_PerfectModel):
    """Always predicts the next class index, so accuracy is zero."""

    def encode_images(self, images, batch_size=32, show_progress=False):
        shifted = [(int(i) + 1) % self.num_classes for i in images]
        return torch.nn.functional.one_hot(
            torch.tensor(shifted), num_classes=self.num_classes
        ).float()


def test_resolve_true_indices_from_integer_labels():
    assert resolve_true_indices(["0", "2", "1"], CLASS_NAMES) == [0, 2, 1]


def test_resolve_true_indices_from_class_names_case_insensitive():
    assert resolve_true_indices(["Tumor", "ADIPOSE"], CLASS_NAMES) == [2, 0]


def test_resolve_true_indices_rejects_unknown_label():
    with pytest.raises(ValueError, match="Could not map label"):
        resolve_true_indices(["spleen"], CLASS_NAMES)


def test_resolve_true_indices_rejects_missing_label():
    with pytest.raises(ValueError, match="Missing label"):
        resolve_true_indices(["0", None], CLASS_NAMES)


def test_resolve_true_indices_rejects_empty_class_names():
    with pytest.raises(ValueError, match="class_names must not be empty"):
        resolve_true_indices(["0"], [])


def test_evaluate_models_zero_shot_with_injected_factory():
    images = [0, 1, 2, 0, 1]
    true_indices = [0, 1, 2, 0, 1]
    prompts = ["a", "b", "c"]

    def factory(model_key, device):
        if model_key == "good":
            return _PerfectModel(len(prompts))
        return _ShiftedModel(len(prompts))

    results = evaluate_models_zero_shot(
        images,
        true_indices,
        prompts,
        ["good", "bad"],
        model_factory=factory,
    )

    assert [r.model for r in results] == ["good", "bad"]
    assert results[0].accuracy == 1.0
    assert results[0].correct == 5
    assert results[0].total == 5
    assert results[0].per_class_total == [2, 2, 1]
    assert results[0].per_class_correct == [2, 2, 1]
    assert results[1].accuracy == 0.0
    assert results[1].correct == 0
    assert results[1].per_class_total == [2, 2, 1]
    assert results[1].per_class_correct == [0, 0, 0]
    assert results[0].correct_flags == [1, 1, 1, 1, 1]
    assert results[1].correct_flags == [0, 0, 0, 0, 0]


def test_compute_model_accuracy_cis_skips_results_without_flags():
    results = [
        ModelZeroShotResult("with_flags", 0.5, 1, 2, correct_flags=[1, 0]),
        ModelZeroShotResult("no_flags", 0.5, 1, 2),
    ]
    cis = compute_model_accuracy_cis(results, num_resamples=200, seed=0)
    assert cis[0] is not None
    assert cis[0]["estimate"] == 0.5
    assert cis[0]["ci_low"] <= cis[0]["estimate"] <= cis[0]["ci_high"]
    assert cis[1] is None


def test_save_model_comparison_csv_includes_ci_columns(tmp_path: Path):
    results = [ModelZeroShotResult("clip", 0.5, 1, 2, correct_flags=[1, 0])]
    cis = compute_model_accuracy_cis(results, num_resamples=200)
    out = tmp_path / "model_comparison.csv"
    save_model_comparison_csv(results, out, cis=cis)
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["ci_low"] != ""
    assert rows[0]["ci_high"] != ""


def test_save_model_comparison_chart_with_cis(tmp_path: Path):
    results = [ModelZeroShotResult("clip", 0.5, 1, 2, correct_flags=[1, 0])]
    cis = compute_model_accuracy_cis(results, num_resamples=200)
    out = tmp_path / "chart.png"
    save_model_comparison_chart(results, out, random_baseline=0.5, cis=cis)
    assert out.exists() and out.stat().st_size > 0


def test_evaluate_models_zero_shot_rejects_length_mismatch():
    def factory(model_key, device):
        return _PerfectModel(2)

    with pytest.raises(ValueError, match="same length"):
        evaluate_models_zero_shot(
            [0, 1],
            [0],
            ["a", "b"],
            ["good"],
            model_factory=factory,
        )


def test_save_model_comparison_csv(tmp_path: Path):
    results = [
        ModelZeroShotResult("clip", 0.25, 5, 20),
        ModelZeroShotResult("plip", 0.6, 12, 20),
    ]
    out = tmp_path / "model_comparison.csv"
    save_model_comparison_csv(results, out)

    assert out.exists()
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["clip", "plip"]
    assert rows[1]["correct"] == "12"


def test_save_model_comparison_per_class_csv(tmp_path: Path):
    results = [
        ModelZeroShotResult("clip", 0.5, 2, 4, [1, 1, 0], [2, 1, 1]),
        ModelZeroShotResult("plip", 0.75, 3, 4, [2, 1, 0], [2, 1, 1]),
    ]
    out = tmp_path / "per_class.csv"
    save_model_comparison_per_class_csv(results, CLASS_NAMES, out)

    assert out.exists()
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(results) * len(CLASS_NAMES)
    clip_tumor = next(
        row for row in rows if row["model"] == "clip" and row["class_name"] == "tumor"
    )
    assert clip_tumor["total"] == "1"
    assert clip_tumor["correct"] == "0"
    assert clip_tumor["accuracy"] == "0.0"


def test_save_model_comparison_per_class_csv_blank_for_empty_class(tmp_path: Path):
    results = [ModelZeroShotResult("clip", 0.5, 1, 2, [1, 0, 0], [2, 0, 0])]
    out = tmp_path / "per_class.csv"
    save_model_comparison_per_class_csv(results, CLASS_NAMES, out)
    with out.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    muscle = next(row for row in rows if row["class_name"] == "muscle")
    assert muscle["total"] == "0"
    assert muscle["accuracy"] == ""


def test_save_model_comparison_per_class_csv_rejects_length_mismatch(tmp_path: Path):
    results = [ModelZeroShotResult("clip", 0.5, 1, 2, [1], [2])]
    with pytest.raises(ValueError, match="per-class entries"):
        save_model_comparison_per_class_csv(results, CLASS_NAMES, tmp_path / "x.csv")


def test_save_model_comparison_chart(tmp_path: Path):
    results = [ModelZeroShotResult("clip", 0.25, 5, 20)]
    out = tmp_path / "chart.png"
    save_model_comparison_chart(
        results,
        out,
        title="Test",
        subtitle="sub",
        random_baseline=1 / 3,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_save_model_comparison_csv_rejects_empty(tmp_path: Path):
    with pytest.raises(ValueError, match="must not be empty"):
        save_model_comparison_csv([], tmp_path / "x.csv")


def _write_manifest_with_images(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, label in enumerate([0, 1, 2, 0, 1, 2]):
        image_path = images_dir / f"patch_{idx}.png"
        Image.new("RGB", (8, 8), color=(label * 40, 0, 0)).save(image_path)
        rows.append((str(image_path), str(label)))

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label"])
        writer.writerows(rows)
    return manifest


def test_cli_compare_models_dry_run(tmp_path: Path, capsys):
    manifest = _write_manifest_with_images(tmp_path)
    exit_code = main(
        [
            "compare-models",
            "--manifest",
            str(manifest),
            "--models",
            "clip",
            "plip",
            "--class-names",
            "adipose",
            "muscle",
            "tumor",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Dry run only" in captured.out
    assert "an H&E image of adipose." in captured.out
    assert not (tmp_path / "out").exists()


def test_cli_compare_models_requires_class_names_for_integer_labels(
    tmp_path: Path, capsys
):
    manifest = _write_manifest_with_images(tmp_path)
    exit_code = main(
        [
            "compare-models",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "--class-names" in captured.out


def test_cli_compare_models_full_run(tmp_path: Path, capsys, monkeypatch):
    manifest = _write_manifest_with_images(tmp_path)

    class _FakeModel:
        def encode_images(self, images, batch_size=32, show_progress=False):
            return torch.ones(len(images), 4)

        def encode_text(self, prompts):
            return torch.ones(len(prompts), 4)

    monkeypatch.setattr(models_module, "create_model", lambda key, device: _FakeModel())

    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "compare-models",
            "--manifest",
            str(manifest),
            "--models",
            "clip",
            "plip",
            "--class-names",
            "adipose",
            "muscle",
            "tumor",
            "--output-dir",
            str(output_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (output_dir / "model_comparison.csv").exists()
    assert (output_dir / "model_comparison_per_class.csv").exists()
    assert (output_dir / "model_comparison.png").exists()
    assert (output_dir / "metadata.json").exists()

    with (output_dir / "model_comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["clip", "plip"]
    for row in rows:
        assert 0.0 <= float(row["accuracy"]) <= 1.0

    with (output_dir / "model_comparison_per_class.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        per_class_rows = list(csv.DictReader(handle))
    assert len(per_class_rows) == 2 * 3

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["results"][0]["per_class"]) == 3

    first_result = metadata["results"][0]
    assert first_result["accuracy_ci"] is not None
    assert (
        first_result["accuracy_ci"]["ci_low"]
        <= first_result["accuracy"]
        <= first_result["accuracy_ci"]["ci_high"]
    )
    assert metadata["bootstrap"]["confidence"] == 0.95
    assert metadata["environment"]["packages"]["torch"] is not None
    assert metadata["environment"]["pathvlm_litebench"]
