import pytest

from pathvlm_litebench.evaluation import (
    accuracy_ci_from_labels,
    bootstrap_proportion_ci,
)


def test_bootstrap_all_correct_is_degenerate():
    result = bootstrap_proportion_ci([1, 1, 1, 1])
    assert result["estimate"] == 1.0
    assert result["ci_low"] == 1.0
    assert result["ci_high"] == 1.0
    assert result["n"] == 4
    assert result["method"] == "percentile_bootstrap"


def test_bootstrap_all_wrong_is_degenerate():
    result = bootstrap_proportion_ci([0, 0, 0])
    assert result["estimate"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_bootstrap_estimate_within_interval_and_ordered():
    flags = [1, 0] * 50
    result = bootstrap_proportion_ci(flags, seed=123)
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]
    assert result["estimate"] == pytest.approx(0.5)


def test_bootstrap_is_reproducible_with_seed():
    flags = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    first = bootstrap_proportion_ci(flags, seed=7)
    second = bootstrap_proportion_ci(flags, seed=7)
    assert first == second


def test_bootstrap_accepts_booleans():
    result = bootstrap_proportion_ci([True, False, True, True])
    assert result["estimate"] == pytest.approx(0.75)


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        bootstrap_proportion_ci([])


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5])
def test_bootstrap_rejects_bad_confidence(confidence):
    with pytest.raises(ValueError, match="confidence must be in"):
        bootstrap_proportion_ci([1, 0], confidence=confidence)


def test_bootstrap_rejects_nonpositive_resamples():
    with pytest.raises(ValueError, match="num_resamples must be positive"):
        bootstrap_proportion_ci([1, 0], num_resamples=0)


def test_accuracy_ci_from_labels_matches_proportion():
    true = ["a", "b", "a", "b"]
    pred = ["a", "b", "b", "b"]
    result = accuracy_ci_from_labels(true, pred, seed=1)
    assert result["estimate"] == pytest.approx(0.75)
    assert result["n"] == 4


def test_accuracy_ci_from_labels_ignores_unlabeled():
    true = ["a", None, "b", None]
    pred = ["a", "x", "b", "y"]
    result = accuracy_ci_from_labels(true, pred)
    assert result["n"] == 2
    assert result["estimate"] == 1.0


def test_accuracy_ci_from_labels_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        accuracy_ci_from_labels(["a"], ["a", "b"])


def test_accuracy_ci_from_labels_rejects_all_unlabeled():
    with pytest.raises(ValueError, match="No labeled samples"):
        accuracy_ci_from_labels([None, None], ["a", "b"])
