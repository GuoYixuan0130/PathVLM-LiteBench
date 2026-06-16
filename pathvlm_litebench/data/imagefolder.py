from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable

DEFAULT_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
)


def _normalize_extensions(extensions: Iterable[str] | None) -> set[str]:
    if extensions is None:
        return set(DEFAULT_EXTENSIONS)

    normalized: set[str] = set()
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        normalized.add(ext)

    if not normalized:
        raise ValueError("extensions must contain at least one entry.")

    return normalized


def _list_class_dirs(parent: Path) -> list[Path]:
    return sorted(
        (child for child in parent.iterdir() if child.is_dir() and not child.name.startswith(".")),
        key=lambda path: path.name,
    )


def _list_images(class_dir: Path, valid_extensions: set[str]) -> list[Path]:
    return sorted(
        (
            child
            for child in class_dir.iterdir()
            if child.is_file()
            and not child.name.startswith(".")
            and child.suffix.lower() in valid_extensions
        ),
        key=lambda path: path.name,
    )


def _format_path(image_path: Path, output_dir: Path, relative: bool) -> str:
    resolved = image_path.resolve()
    if not relative:
        return str(resolved)

    try:
        return os.path.relpath(resolved, output_dir.resolve())
    except ValueError as exc:
        raise ValueError(
            "Cannot build a relative path because the images and the output "
            "manifest are on different drives; omit --relative to write "
            "absolute paths."
        ) from exc


def build_imagefolder_manifest(
    image_dir: str | Path,
    output_csv: str | Path,
    *,
    has_split: bool = False,
    extensions: Iterable[str] | None = None,
    relative: bool = False,
) -> dict[str, Any]:
    """
    Build a standard patch manifest from an ImageFolder-style directory tree.

    Flat layout (``has_split=False``)::

        image_dir/<class>/<image>

    Split layout (``has_split=True``)::

        image_dir/<split>/<class>/<image>

    Each leaf class directory becomes a label and each image file in it becomes
    a manifest row. The output CSV has ``image_path,label,split`` columns, where
    ``split`` is blank in the flat layout.

    Args:
        image_dir: Root of the ImageFolder tree.
        output_csv: Output manifest CSV path.
        has_split: Whether the tree has a leading split level.
        extensions: Image extensions to include (defaults to common formats).
        relative: Write image paths relative to the output CSV directory instead
            of absolute paths.

    Returns:
        A summary dict with ``output_csv``, ``num_records``, ``num_classes``,
        ``label_distribution``, and ``split_distribution``.

    Raises:
        FileNotFoundError: If ``image_dir`` does not exist.
        ValueError: If no class directories or no images are found.
    """
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"Image directory is not a directory: {image_dir}")

    valid_extensions = _normalize_extensions(extensions)
    output_dir = output_csv.parent

    if has_split:
        scopes = [(split_dir.name, split_dir) for split_dir in _list_class_dirs(image_dir)]
        if not scopes:
            raise ValueError(
                f"No split directories found under {image_dir}. Expected "
                f"<split>/<class>/<image> layout for --has-split."
            )
    else:
        scopes = [("", image_dir)]

    rows: list[dict[str, str]] = []
    label_distribution: dict[str, int] = {}
    split_distribution: dict[str, int] = {}

    for split_name, scope_dir in scopes:
        class_dirs = _list_class_dirs(scope_dir)
        if not class_dirs:
            hint = (
                " Did you mean to pass --has-split for a <split>/<class>/<image> tree?"
                if not has_split
                else ""
            )
            raise ValueError(
                f"No class subdirectories found under {scope_dir}.{hint}"
            )

        for class_dir in class_dirs:
            label = class_dir.name
            for image_path in _list_images(class_dir, valid_extensions):
                rows.append(
                    {
                        "image_path": _format_path(image_path, output_dir, relative),
                        "label": label,
                        "split": split_name,
                    }
                )
                label_distribution[label] = label_distribution.get(label, 0) + 1
                if split_name:
                    split_distribution[split_name] = (
                        split_distribution.get(split_name, 0) + 1
                    )

    if not rows:
        raise ValueError(
            f"No images with extensions {sorted(valid_extensions)} were found "
            f"under {image_dir}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "output_csv": str(output_csv),
        "num_records": len(rows),
        "num_classes": len(label_distribution),
        "label_distribution": dict(sorted(label_distribution.items())),
        "split_distribution": dict(sorted(split_distribution.items())),
    }
