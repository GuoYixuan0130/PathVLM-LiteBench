from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_proportion_ci(
    success_flags: Sequence[bool | int | float],
    *,
    num_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Percentile bootstrap confidence interval for a proportion.

    A proportion such as classification accuracy or Recall@K is the mean of a
    per-unit success indicator (1 = correct/recalled, 0 = not). This resamples
    those indicators with replacement to approximate the sampling distribution
    of the mean and returns a two-sided percentile interval. It is the cheapest
    way to attach uncertainty to a point estimate on a small evaluation set.

    Args:
        success_flags: Per-unit 0/1 (or boolean) success indicators.
        num_resamples: Number of bootstrap resamples.
        confidence: Two-sided confidence level in (0, 1), e.g. 0.95.
        seed: RNG seed so the interval is reproducible.

    Returns:
        Dict with keys: estimate, ci_low, ci_high, confidence, num_resamples,
        n, method.

    Raises:
        ValueError: if success_flags is empty, confidence is out of range, or
            num_resamples is not positive.
    """
    flags = np.asarray(list(success_flags), dtype=float)
    n = int(flags.shape[0])
    if n == 0:
        raise ValueError("success_flags must not be empty.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    if num_resamples <= 0:
        raise ValueError(f"num_resamples must be positive, got {num_resamples}.")

    estimate = float(flags.mean())
    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(num_resamples, n))
    resample_means = flags[resample_indices].mean(axis=1)

    alpha = 1.0 - confidence
    ci_low = float(np.quantile(resample_means, alpha / 2.0))
    ci_high = float(np.quantile(resample_means, 1.0 - alpha / 2.0))

    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "confidence": confidence,
        "num_resamples": num_resamples,
        "n": n,
        "method": "percentile_bootstrap",
    }


def accuracy_ci_from_labels(
    true_labels: Sequence[str | None],
    predicted_labels: Sequence[str],
    *,
    num_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    """
    Bootstrap confidence interval for classification accuracy.

    Unlabeled samples (``true_label is None``) are dropped before scoring, so
    the interval reflects only the labeled patches that accuracy is computed on.

    Args:
        true_labels: Ground-truth labels; ``None`` entries are ignored.
        predicted_labels: Predicted labels, aligned with ``true_labels``.
        num_resamples: Number of bootstrap resamples.
        confidence: Two-sided confidence level in (0, 1).
        seed: RNG seed.

    Returns:
        The same dict as :func:`bootstrap_proportion_ci`.

    Raises:
        ValueError: if lengths mismatch or there are no labeled samples.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"true_labels and predicted_labels must have the same length: "
            f"{len(true_labels)} vs {len(predicted_labels)}"
        )

    flags = [
        1 if predicted == true else 0
        for true, predicted in zip(true_labels, predicted_labels)
        if true is not None
    ]
    if not flags:
        raise ValueError("No labeled samples to compute an accuracy interval.")

    return bootstrap_proportion_ci(
        flags,
        num_resamples=num_resamples,
        confidence=confidence,
        seed=seed,
    )
