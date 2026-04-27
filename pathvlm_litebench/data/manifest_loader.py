from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchRecord:
    image_path: str
    label: str | None = None
    split: str | None = None
    case_id: str | None = None
    slide_id: str | None = None
    metadata: dict[str, str] | None = None


def _read_optional_value(
    row: dict[str, str],
    column_name: str | None,
    fieldnames: list[str],
) -> str | None:
    if column_name is None or column_name not in fieldnames:
        return None

    value = row.get(column_name)
    if value is None:
        return None

    value = value.strip()
    return value if value else None


def load_patch_manifest(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
    path_column: str = "image_path",
    label_column: str | None = "label",
    split_column: str | None = "split",
    case_id_column: str | None = "case_id",
    slide_id_column: str | None = "slide_id",
    require_exists: bool = True,
) -> list[PatchRecord]:
    """
    Load patch records from a CSV manifest file.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    base_dir = Path(image_root) if image_root is not None else manifest_path.parent

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Manifest CSV is empty: {manifest_path}")

        fieldnames = list(reader.fieldnames)

        if path_column not in fieldnames:
            raise ValueError(
                f"Required path column '{path_column}' not found in manifest. "
                f"Available columns: {', '.join(fieldnames)}"
            )

        excluded_columns = {
            path_column,
            label_column,
            split_column,
            case_id_column,
            slide_id_column,
        }

        records: list[PatchRecord] = []

        for row_idx, row in enumerate(reader, start=2):
            raw_image_path = row.get(path_column)
            if raw_image_path is None or not raw_image_path.strip():
                raise ValueError(
                    f"Empty image path at row {row_idx} in manifest: {manifest_path}"
                )

            row_path = Path(raw_image_path.strip())
            if not row_path.is_absolute():
                row_path = base_dir / row_path

            resolved_path = row_path.resolve()

            if require_exists and not resolved_path.exists():
                raise FileNotFoundError(
                    f"Image file not found at row {row_idx}: {resolved_path}"
                )

            metadata = {
                key: value
                for key, value in row.items()
                if key is not None
                and key not in excluded_columns
                and value is not None
                and value.strip() != ""
            }

            records.append(
                PatchRecord(
                    image_path=str(resolved_path),
                    label=_read_optional_value(row, label_column, fieldnames),
                    split=_read_optional_value(row, split_column, fieldnames),
                    case_id=_read_optional_value(row, case_id_column, fieldnames),
                    slide_id=_read_optional_value(row, slide_id_column, fieldnames),
                    metadata=metadata or None,
                )
            )

    if len(records) == 0:
        raise ValueError(f"Manifest CSV has no rows: {manifest_path}")

    return records


def records_to_image_paths(records: list[PatchRecord]) -> list[str]:
    """
    Convert PatchRecord list to image path list.
    """
    return [record.image_path for record in records]


def records_to_labels(records: list[PatchRecord]) -> list[str | None]:
    """
    Convert PatchRecord list to label list.
    """
    return [record.label for record in records]


def get_unique_labels(records: list[PatchRecord]) -> list[str]:
    """
    Get sorted unique non-empty labels from records.
    """
    return sorted({record.label for record in records if record.label is not None})


def filter_records_by_split(records: list[PatchRecord], split: str) -> list[PatchRecord]:
    """
    Filter records by split value.
    """
    return [record for record in records if record.split == split]


def filter_records_by_label(records: list[PatchRecord], label: str) -> list[PatchRecord]:
    """
    Filter records by label value.
    """
    return [record for record in records if record.label == label]
