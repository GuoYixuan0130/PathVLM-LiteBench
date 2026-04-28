from __future__ import annotations

import csv
import random
from pathlib import Path


def sample_manifest(
    input_csv: str | Path,
    output_csv: str | Path,
    label_column: str = "label",
    split_column: str | None = "split",
    split: str | None = None,
    samples_per_label: int | None = None,
    max_total: int | None = None,
    seed: int = 42,
) -> str:
    """
    Sample a smaller manifest CSV subset with optional split filter and label balancing.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"Input CSV has no data rows: {input_csv}")

    if label_column not in fieldnames:
        raise ValueError(
            f"label_column '{label_column}' is not present in input CSV. "
            f"Available columns: {', '.join(fieldnames)}"
        )

    if split is not None:
        if split_column is None:
            raise ValueError("split_column must not be None when split is provided.")
        if split_column not in fieldnames:
            raise ValueError(
                f"split_column '{split_column}' is not present in input CSV. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        rows = [row for row in rows if row.get(split_column) == split]

    if len(rows) == 0:
        raise ValueError("No records available after split filtering.")

    rng = random.Random(seed)

    if samples_per_label is not None:
        if samples_per_label <= 0:
            raise ValueError("samples_per_label must be > 0.")

        grouped_rows: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            label = row.get(label_column, "")
            grouped_rows.setdefault(label, []).append(row)

        sampled_rows: list[dict[str, str]] = []
        for label in sorted(grouped_rows):
            label_rows = grouped_rows[label]
            take_n = min(samples_per_label, len(label_rows))
            sampled_rows.extend(rng.sample(label_rows, k=take_n))
        rows = sampled_rows

    if max_total is not None:
        if max_total <= 0:
            raise ValueError("max_total must be > 0.")
        if len(rows) > max_total:
            rows = rng.sample(rows, k=max_total)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return str(output_csv)


def summarize_manifest(
    manifest_csv: str | Path,
    label_column: str = "label",
    split_column: str | None = "split",
) -> dict:
    """
    Summarize manifest record count plus label/split distributions.
    """
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")

    with manifest_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV is empty: {manifest_csv}")

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if label_column not in fieldnames:
        raise ValueError(
            f"label_column '{label_column}' is not present in manifest CSV. "
            f"Available columns: {', '.join(fieldnames)}"
        )

    label_distribution: dict[str, int] = {}
    split_distribution: dict[str, int] = {}

    for row in rows:
        label = row.get(label_column, "")
        label_distribution[label] = label_distribution.get(label, 0) + 1

        if split_column is not None and split_column in fieldnames:
            split_value = row.get(split_column, "")
            split_distribution[split_value] = split_distribution.get(split_value, 0) + 1

    return {
        "num_records": len(rows),
        "label_distribution": label_distribution,
        "split_distribution": split_distribution,
    }
