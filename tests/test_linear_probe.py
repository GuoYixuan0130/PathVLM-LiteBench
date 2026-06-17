import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import pathvlm_litebench.models as models_module
from pathvlm_litebench.cli import main
from pathvlm_litebench.evaluation import run_linear_probe


def _separable_dataset():
    # Two well-separated clusters: class "a" near +x, class "b" near +y.
    train_embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=float
    )
    train_labels = ["a", "a", "b", "b"]
    test_embeddings = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=float)
    return train_embeddings, train_labels, test_embeddings


def test_run_linear_probe_separates_classes():
    train_embeddings, train_labels, test_embeddings = _separable_dataset()
    probe = run_linear_probe(train_embeddings, train_labels, test_embeddings, seed=0)

    assert probe["predicted_labels"] == ["a", "b"]
    assert probe["class_names"] == ["a", "b"]
    assert probe["predicted_indices"] == [0, 1]
    assert probe["num_train"] == 4
    assert probe["num_test"] == 2
    assert probe["embedding_dim"] == 2
    assert all(0.0 <= c <= 1.0 for c in probe["confidences"])


def test_run_linear_probe_accepts_torch_tensors():
    train_embeddings, train_labels, test_embeddings = _separable_dataset()
    probe = run_linear_probe(
        torch.tensor(train_embeddings),
        train_labels,
        torch.tensor(test_embeddings),
    )
    assert probe["predicted_labels"] == ["a", "b"]


def test_run_linear_probe_uses_provided_class_order():
    train_embeddings, train_labels, test_embeddings = _separable_dataset()
    probe = run_linear_probe(
        train_embeddings,
        train_labels,
        test_embeddings,
        class_names=["b", "a"],
    )
    # "a" is index 1 and "b" is index 0 under the provided order.
    assert probe["class_names"] == ["b", "a"]
    assert probe["predicted_indices"] == [1, 0]


def test_run_linear_probe_rejects_dim_mismatch():
    with pytest.raises(ValueError, match="embedding dims differ"):
        run_linear_probe(
            np.zeros((2, 3)), ["a", "b"], np.zeros((1, 4))
        )


def test_run_linear_probe_rejects_label_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        run_linear_probe(np.zeros((2, 3)), ["a"], np.zeros((1, 3)))


def test_run_linear_probe_rejects_single_class():
    with pytest.raises(ValueError, match="at least two distinct"):
        run_linear_probe(np.zeros((2, 3)), ["a", "a"], np.zeros((1, 3)))


def test_run_linear_probe_rejects_missing_train_label():
    with pytest.raises(ValueError, match="must be present"):
        run_linear_probe(np.zeros((2, 3)), ["a", None], np.zeros((1, 3)))


def test_run_linear_probe_rejects_incomplete_class_names():
    train_embeddings, train_labels, test_embeddings = _separable_dataset()
    with pytest.raises(ValueError, match="missing train labels"):
        run_linear_probe(
            train_embeddings,
            train_labels,
            test_embeddings,
            class_names=["a"],
        )


class _ColorModel:
    """Encodes each patch into a one-hot of its red-channel class index."""

    def encode_images(self, images, batch_size=32, show_progress=False):
        features = []
        for image in images:
            class_index = image.getpixel((0, 0))[0] // 40
            onehot = [0.0, 0.0, 0.0]
            onehot[class_index] = 1.0
            features.append(onehot)
        return torch.tensor(features)

    def encode_text(self, prompts):
        return torch.eye(len(prompts))


def _write_split_manifest(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    plan = [("train", label) for label in [0, 1, 2, 0, 1, 2]] + [
        ("test", label) for label in [0, 1, 2]
    ]
    for idx, (split, label) in enumerate(plan):
        image_path = images_dir / f"patch_{idx}.png"
        Image.new("RGB", (8, 8), color=(label * 40, 0, 0)).save(image_path)
        rows.append((str(image_path), str(label), split))

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "label", "split"])
        writer.writerows(rows)
    return manifest


def test_cli_linear_probe_dry_run(tmp_path: Path, capsys):
    manifest = _write_split_manifest(tmp_path)
    exit_code = main(
        [
            "linear-probe",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dry run only" in captured.out
    assert "Train split 'train': 6 patches" in captured.out
    assert "Test split 'test': 3 patches" in captured.out
    assert not (tmp_path / "out").exists()


def test_cli_linear_probe_full_run(tmp_path: Path, capsys, monkeypatch):
    manifest = _write_split_manifest(tmp_path)
    monkeypatch.setattr(models_module, "create_model", lambda key, device: _ColorModel())

    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "linear-probe",
            "--manifest",
            str(manifest),
            "--model",
            "clip",
            "--output-dir",
            str(output_dir),
            "--bootstrap-resamples",
            "200",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "errors.csv").exists()
    assert (output_dir / "metrics.json").exists()

    payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["accuracy"] == 1.0
    assert payload["metrics"]["accuracy_ci"]["estimate"] == 1.0
    assert payload["metadata"]["task"] == "linear-probe"
    assert payload["metadata"]["num_train"] == 6
    assert payload["metadata"]["num_test"] == 3
    assert payload["metadata"]["probe"]["classifier"] == "logistic_regression"
    assert payload["metadata"]["environment"]["packages"]["scikit-learn"] is not None
    assert "Accuracy: 100.0%" in captured.out
