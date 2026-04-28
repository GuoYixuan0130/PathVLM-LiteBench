from __future__ import annotations


def get_class_names_from_labels(
    true_labels: list[str],
    predicted_labels: list[str],
    class_names: list[str] | None = None,
) -> list[str]:
    """
    Resolve class names used for classification metric computation.

    Args:
        true_labels:
            Ground-truth labels.
        predicted_labels:
            Predicted labels.
        class_names:
            Optional explicit class-name list. If provided, this list is returned.

    Returns:
        A list of class names.

    Raises:
        ValueError:
            If true_labels or predicted_labels is empty.
    """
    if len(true_labels) == 0:
        raise ValueError("true_labels must not be empty.")

    if len(predicted_labels) == 0:
        raise ValueError("predicted_labels must not be empty.")

    if class_names is not None:
        return class_names

    return sorted(set(true_labels) | set(predicted_labels))


def compute_confusion_matrix(
    true_labels: list[str],
    predicted_labels: list[str],
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute confusion matrix for classification results.

    Args:
        true_labels:
            Ground-truth labels.
        predicted_labels:
            Predicted labels.
        class_names:
            Optional explicit class-name list. If omitted, derived from labels.

    Returns:
        A dictionary with:
        - class_names: list[str]
        - matrix: list[list[int]]
          where matrix[i][j] is count of true class i predicted as class j.

    Raises:
        ValueError:
            If labels are empty, lengths mismatch, or labels not found in class_names.
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            f"true_labels and predicted_labels must have the same length: "
            f"{len(true_labels)} vs {len(predicted_labels)}"
        )

    resolved_class_names = get_class_names_from_labels(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )
    if len(resolved_class_names) == 0:
        raise ValueError("class_names must not be empty.")

    index_by_class = {name: idx for idx, name in enumerate(resolved_class_names)}
    matrix = [
        [0 for _ in range(len(resolved_class_names))]
        for _ in range(len(resolved_class_names))
    ]

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label not in index_by_class:
            raise ValueError(f"Unknown true label '{true_label}' not found in class_names.")
        if predicted_label not in index_by_class:
            raise ValueError(
                f"Unknown predicted label '{predicted_label}' not found in class_names."
            )

        true_idx = index_by_class[true_label]
        predicted_idx = index_by_class[predicted_label]
        matrix[true_idx][predicted_idx] += 1

    return {
        "class_names": resolved_class_names,
        "matrix": matrix,
    }


def compute_classification_report(
    true_labels: list[str],
    predicted_labels: list[str],
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute aggregate and per-class classification metrics.

    Metrics:
    - accuracy
    - balanced_accuracy (macro recall)
    - macro_precision
    - macro_recall
    - macro_f1
    - per_class precision/recall/f1/support
    - confusion_matrix

    Args:
        true_labels:
            Ground-truth labels.
        predicted_labels:
            Predicted labels.
        class_names:
            Optional explicit class-name list.

    Returns:
        A dictionary containing aggregate metrics, per-class metrics, and confusion matrix.
    """
    confusion = compute_confusion_matrix(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )
    resolved_class_names = confusion["class_names"]
    matrix = confusion["matrix"]

    total = len(true_labels)
    correct = sum(matrix[i][i] for i in range(len(resolved_class_names)))
    accuracy = correct / total

    per_class: dict[str, dict[str, float | int]] = {}
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []

    for class_idx, class_name in enumerate(resolved_class_names):
        tp = matrix[class_idx][class_idx]
        support = sum(matrix[class_idx])
        fn = support - tp
        fp = sum(matrix[row_idx][class_idx] for row_idx in range(len(resolved_class_names))) - tp

        precision_den = tp + fp
        recall_den = tp + fn

        precision = tp / precision_den if precision_den > 0 else 0.0
        recall = tp / recall_den if recall_den > 0 else 0.0
        f1_den = precision + recall
        f1 = (2.0 * precision * recall / f1_den) if f1_den > 0 else 0.0

        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

    macro_precision = sum(precision_values) / len(precision_values)
    macro_recall = sum(recall_values) / len(recall_values)
    macro_f1 = sum(f1_values) / len(f1_values)
    balanced_accuracy = macro_recall

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
