from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _to_2d_float_array(embeddings: Any, name: str) -> np.ndarray:
    """Coerce a torch tensor / array-like of embeddings to a 2D float ndarray."""
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    array = np.asarray(embeddings, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D [num_samples, embedding_dim].")
    return array


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def run_linear_probe(
    train_embeddings: Any,
    train_labels: Sequence[str],
    test_embeddings: Any,
    *,
    class_names: Sequence[str] | None = None,
    C: float = 1.0,
    max_iter: int = 1000,
    seed: int = 0,
    normalize: bool = True,
) -> dict:
    """
    Fit a logistic-regression linear probe on frozen embeddings and predict.

    A linear probe trains only a linear classifier on top of frozen image
    embeddings, leaving the encoder untouched. It is the standard low-compute
    step beyond zero-shot: it measures how linearly separable a model's frozen
    features already are, without any fine-tuning.

    Args:
        train_embeddings: Frozen train embeddings, shape [num_train, dim].
        train_labels: One label per train embedding; must all be non-empty.
        test_embeddings: Frozen test embeddings, shape [num_test, dim].
        class_names: Optional explicit class order. If given, it must cover all
            train labels; otherwise the sorted unique train labels are used.
        C: Inverse L2 regularization strength for logistic regression.
        max_iter: Maximum solver iterations.
        seed: Random seed for a reproducible fit.
        normalize: L2-normalize embeddings before fitting (standard for
            CLIP-style features).

    Returns:
        Dict with predicted_labels, predicted_indices (into class_names),
        confidences (max class probability), class_names, and the probe config.

    Raises:
        ValueError: on shape/length mismatch, missing train labels, fewer than
            two train classes, or class_names that omit a train label.
    """
    from sklearn.linear_model import LogisticRegression

    train_array = _to_2d_float_array(train_embeddings, "train_embeddings")
    test_array = _to_2d_float_array(test_embeddings, "test_embeddings")

    if len(train_labels) != train_array.shape[0]:
        raise ValueError(
            f"train_labels and train_embeddings must have the same length: "
            f"{len(train_labels)} vs {train_array.shape[0]}"
        )
    if train_array.shape[1] != test_array.shape[1]:
        raise ValueError(
            f"train and test embedding dims differ: "
            f"{train_array.shape[1]} vs {test_array.shape[1]}"
        )
    if any(label is None or not str(label).strip() for label in train_labels):
        raise ValueError("Every train label must be present to fit a linear probe.")

    train_label_list = [str(label) for label in train_labels]
    unique_train_labels = sorted(set(train_label_list))
    if len(unique_train_labels) < 2:
        raise ValueError(
            "A linear probe needs at least two distinct train classes; "
            f"found {len(unique_train_labels)}."
        )

    if class_names is None:
        resolved_class_names = unique_train_labels
    else:
        resolved_class_names = [str(name) for name in class_names]
        missing = set(unique_train_labels) - set(resolved_class_names)
        if missing:
            raise ValueError(
                f"class_names is missing train labels: {sorted(missing)}."
            )

    if normalize:
        train_array = _l2_normalize(train_array)
        test_array = _l2_normalize(test_array)

    classifier = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    classifier.fit(train_array, train_label_list)

    probabilities = classifier.predict_proba(test_array)
    best_positions = probabilities.argmax(axis=1)
    fitted_classes = list(classifier.classes_)

    predicted_labels = [fitted_classes[pos] for pos in best_positions]
    confidences = [float(probabilities[row, pos]) for row, pos in enumerate(best_positions)]
    index_by_class = {name: idx for idx, name in enumerate(resolved_class_names)}
    predicted_indices = [index_by_class[label] for label in predicted_labels]

    return {
        "predicted_labels": predicted_labels,
        "predicted_indices": predicted_indices,
        "confidences": confidences,
        "class_names": resolved_class_names,
        "num_train": train_array.shape[0],
        "num_test": test_array.shape[0],
        "embedding_dim": int(train_array.shape[1]),
        "C": C,
        "max_iter": max_iter,
        "seed": seed,
        "normalize": normalize,
    }
