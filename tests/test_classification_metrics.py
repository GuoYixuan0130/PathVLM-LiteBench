import pytest

from pathvlm_litebench.evaluation import (
    compute_classification_report,
    compute_confusion_matrix,
)


def test_confusion_matrix_basic():
    true_labels = ["HP", "HP", "SSA", "SSA"]
    predicted_labels = ["HP", "SSA", "SSA", "HP"]
    class_names = ["HP", "SSA"]

    confusion = compute_confusion_matrix(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    assert confusion["class_names"] == class_names
    assert confusion["matrix"] == [
        [1, 1],
        [1, 1],
    ]


def test_classification_report_basic():
    true_labels = ["HP", "HP", "SSA", "SSA"]
    predicted_labels = ["HP", "SSA", "SSA", "HP"]
    class_names = ["HP", "SSA"]

    report = compute_classification_report(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    assert report["accuracy"] == pytest.approx(0.5)
    assert "HP" in report["per_class"]
    assert "SSA" in report["per_class"]
    assert "macro_f1" in report
    assert "balanced_accuracy" in report
    assert "confusion_matrix" in report


def test_perfect_classification():
    true_labels = ["HP", "SSA"]
    predicted_labels = ["HP", "SSA"]

    report = compute_classification_report(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=["HP", "SSA"],
    )

    assert report["accuracy"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert report["balanced_accuracy"] == pytest.approx(1.0)


def test_length_mismatch():
    true_labels = ["HP", "SSA"]
    predicted_labels = ["HP"]

    with pytest.raises(ValueError):
        compute_confusion_matrix(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            class_names=["HP", "SSA"],
        )


def test_empty_labels():
    with pytest.raises(ValueError):
        compute_confusion_matrix(
            true_labels=[],
            predicted_labels=[],
            class_names=["HP", "SSA"],
        )


def test_unknown_label_with_given_class_names():
    true_labels = ["HP", "OTHER"]
    predicted_labels = ["HP", "SSA"]

    with pytest.raises(ValueError):
        compute_confusion_matrix(
            true_labels=true_labels,
            predicted_labels=predicted_labels,
            class_names=["HP", "SSA"],
        )
