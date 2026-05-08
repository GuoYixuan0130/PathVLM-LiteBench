from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CoordinatePatchRecord:
    image_path: str
    x: float
    y: float
    width: float | None = None
    height: float | None = None
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


def _read_required_float(
    row: dict[str, str],
    column_name: str,
    row_idx: int,
    manifest_path: Path,
) -> float:
    value = row.get(column_name)
    if value is None or not value.strip():
        raise ValueError(
            f"Empty required coordinate column '{column_name}' at row {row_idx} "
            f"in manifest: {manifest_path}"
        )

    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric coordinate '{column_name}' at row {row_idx}: {value!r}"
        ) from exc

    if not math.isfinite(parsed):
        raise ValueError(
            f"Invalid non-finite coordinate '{column_name}' at row {row_idx}: {value!r}"
        )

    return parsed


def _read_optional_positive_float(
    row: dict[str, str],
    column_name: str | None,
    fieldnames: list[str],
    row_idx: int,
) -> float | None:
    if column_name is None or column_name not in fieldnames:
        return None

    value = row.get(column_name)
    if value is None or not value.strip():
        return None

    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric size column '{column_name}' at row {row_idx}: {value!r}"
        ) from exc

    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(
            f"Size column '{column_name}' must be a positive finite value "
            f"at row {row_idx}: {value!r}"
        )

    return parsed


def load_coordinate_patch_manifest(
    manifest_path: str | Path,
    image_root: str | Path | None = None,
    path_column: str = "image_path",
    x_column: str = "x",
    y_column: str = "y",
    width_column: str | None = "width",
    height_column: str | None = "height",
    label_column: str | None = "label",
    split_column: str | None = "split",
    case_id_column: str | None = "case_id",
    slide_id_column: str | None = "slide_id",
    require_exists: bool = True,
) -> list[CoordinatePatchRecord]:
    """
    Load slide-derived patch records with required x/y coordinates.

    The loader expects pre-extracted patch image paths. It does not read WSI files
    or perform slide tiling.
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
        required_columns = [path_column, x_column, y_column]
        for column_name in required_columns:
            if column_name not in fieldnames:
                raise ValueError(
                    f"Required column '{column_name}' not found in manifest. "
                    f"Available columns: {', '.join(fieldnames)}"
                )

        excluded_columns = {
            path_column,
            x_column,
            y_column,
            width_column,
            height_column,
            label_column,
            split_column,
            case_id_column,
            slide_id_column,
        }

        records: list[CoordinatePatchRecord] = []

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
                CoordinatePatchRecord(
                    image_path=str(resolved_path),
                    x=_read_required_float(row, x_column, row_idx, manifest_path),
                    y=_read_required_float(row, y_column, row_idx, manifest_path),
                    width=_read_optional_positive_float(
                        row, width_column, fieldnames, row_idx
                    ),
                    height=_read_optional_positive_float(
                        row, height_column, fieldnames, row_idx
                    ),
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


def coordinate_records_to_image_paths(records: list[CoordinatePatchRecord]) -> list[str]:
    """
    Convert CoordinatePatchRecord list to image path list.
    """
    return [record.image_path for record in records]
