from __future__ import annotations

import csv
from pathlib import Path


def _read_cell(row: dict[str, str], column: str | None, fieldnames: list[str]) -> str:
    if column is None or column not in fieldnames:
        return ""

    value = row.get(column, "")
    return value.strip() if value is not None else ""


def convert_manifest(
    input_csv: str | Path,
    output_csv: str | Path,
    path_column: str,
    label_column: str | None = None,
    split_column: str | None = None,
    case_id_column: str | None = None,
    slide_id_column: str | None = None,
    image_root: str | Path | None = None,
    require_exists: bool = False,
    case_id_from_filename: bool = True,
    copy_extra_columns: bool = True,
) -> str:
    """
    Convert a dataset-specific annotation CSV into the standard patch manifest format.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")

        fieldnames = list(reader.fieldnames)
        if path_column not in fieldnames:
            raise ValueError(
                f"path_column '{path_column}' is not present in input CSV. "
                f"Available columns: {', '.join(fieldnames)}"
            )

        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"Input CSV has no data rows: {input_csv}")

    mapped_columns = {
        path_column,
        label_column,
        split_column,
        case_id_column,
        slide_id_column,
    }
    extra_columns = [
        name for name in fieldnames if name not in mapped_columns
    ] if copy_extra_columns else []

    output_fieldnames = ["image_path", "label", "split", "case_id", "slide_id"] + extra_columns
    output_rows: list[dict[str, str]] = []

    exists_root = Path(image_root) if image_root is not None else input_csv.parent

    for row_idx, row in enumerate(rows, start=2):
        image_path_value = _read_cell(row, path_column, fieldnames)
        if image_path_value == "":
            raise ValueError(f"Empty image path in row {row_idx} of {input_csv}")

        label_value = _read_cell(row, label_column, fieldnames)
        split_value = _read_cell(row, split_column, fieldnames)
        case_id_value = _read_cell(row, case_id_column, fieldnames)
        slide_id_value = _read_cell(row, slide_id_column, fieldnames)

        if case_id_value == "" and case_id_from_filename:
            case_id_value = Path(image_path_value).stem

        if require_exists:
            image_path_obj = Path(image_path_value)
            if not image_path_obj.is_absolute():
                image_path_obj = exists_root / image_path_obj
            if not image_path_obj.exists():
                raise FileNotFoundError(
                    f"Image path from manifest does not exist at row {row_idx}: {image_path_obj}"
                )

        converted_row: dict[str, str] = {
            "image_path": image_path_value,
            "label": label_value,
            "split": split_value,
            "case_id": case_id_value,
            "slide_id": slide_id_value,
        }

        for extra_column in extra_columns:
            converted_row[extra_column] = _read_cell(row, extra_column, fieldnames)

        output_rows.append(converted_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return str(output_csv)


def convert_mhist_manifest(
    annotations_csv: str | Path,
    output_csv: str | Path,
    image_root: str | Path | None = None,
    require_exists: bool = False,
) -> str:
    """
    Convert MHIST annotations.csv to the standard patch manifest format.
    """
    return convert_manifest(
        input_csv=annotations_csv,
        output_csv=output_csv,
        path_column="Image Name",
        label_column="Majority Vote Label",
        split_column="Partition",
        case_id_column=None,
        slide_id_column=None,
        image_root=image_root,
        require_exists=require_exists,
        case_id_from_filename=True,
        copy_extra_columns=True,
    )
