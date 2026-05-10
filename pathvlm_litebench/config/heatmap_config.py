from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


VALID_ALIGN_BY = {"image_path", "order"}
VALID_SCORING_DEVICES = {"auto", "cpu", "cuda"}


@dataclass(frozen=True)
class PatchCoordinateHeatmapConfig:
    manifest: str
    score_csv: str
    output: str
    score_column: str = "score"
    score_path_column: str = "image_path"
    align_by: str = "image_path"
    image_root: str | None = None
    score_image_root: str | None = None
    path_column: str = "image_path"
    x_column: str = "x"
    y_column: str = "y"
    require_exists: bool = False
    title: str | None = None
    cmap: str = "viridis"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_non_empty_string(self.manifest, "manifest")
        _require_non_empty_string(self.score_csv, "score_csv")
        _require_non_empty_string(self.output, "output")
        _require_non_empty_string(self.score_column, "score_column")
        _require_non_empty_string(self.score_path_column, "score_path_column")
        _require_non_empty_string(self.path_column, "path_column")
        _require_non_empty_string(self.x_column, "x_column")
        _require_non_empty_string(self.y_column, "y_column")
        _require_non_empty_string(self.cmap, "cmap")

        if self.align_by not in VALID_ALIGN_BY:
            valid = ", ".join(sorted(VALID_ALIGN_BY))
            raise ValueError(f"align_by must be one of: {valid}. Got: {self.align_by}")

        if not isinstance(self.require_exists, bool):
            raise ValueError("require_exists must be a bool.")


@dataclass(frozen=True)
class PatchCoordinateHeatmapScoringConfig:
    manifest: str
    prompt: str
    output_dir: str = "outputs/patch_coordinate_heatmap_scored"
    score_csv: str | None = None
    heatmap_output: str | None = None
    model: str = "clip"
    device: str = "auto"
    image_root: str | None = None
    path_column: str = "image_path"
    x_column: str = "x"
    y_column: str = "y"
    max_images: int | None = None
    title: str | None = None
    cmap: str = "viridis"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_non_empty_string(self.manifest, "manifest")
        _require_non_empty_string(self.prompt, "prompt")
        _require_non_empty_string(self.output_dir, "output_dir")
        _require_optional_non_empty_string(self.score_csv, "score_csv")
        _require_optional_non_empty_string(self.heatmap_output, "heatmap_output")
        _require_non_empty_string(self.model, "model")
        _require_non_empty_string(self.device, "device")
        _require_optional_non_empty_string(self.image_root, "image_root")
        _require_non_empty_string(self.path_column, "path_column")
        _require_non_empty_string(self.x_column, "x_column")
        _require_non_empty_string(self.y_column, "y_column")
        _require_optional_non_empty_string(self.title, "title")
        _require_non_empty_string(self.cmap, "cmap")

        if self.device not in VALID_SCORING_DEVICES:
            valid = ", ".join(sorted(VALID_SCORING_DEVICES))
            raise ValueError(f"device must be one of: {valid}. Got: {self.device}")

        if self.max_images is not None and self.max_images <= 0:
            raise ValueError(
                f"max_images must be > 0 when provided. Got: {self.max_images}"
            )


def _require_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_optional_non_empty_string(value: object, field_name: str) -> None:
    if value is None:
        return
    _require_non_empty_string(value, field_name)


def patch_coordinate_heatmap_config_to_dict(
    config: PatchCoordinateHeatmapConfig,
) -> dict[str, Any]:
    data = asdict(config)
    data["task"] = "patch_coordinate_heatmap"
    return data


def patch_coordinate_heatmap_config_from_dict(
    data: dict[str, Any],
) -> PatchCoordinateHeatmapConfig:
    task = data.get("task", "patch_coordinate_heatmap")
    if task != "patch_coordinate_heatmap":
        raise ValueError(
            f"Config task must be 'patch_coordinate_heatmap'. Got: {task}"
        )

    allowed_fields = set(PatchCoordinateHeatmapConfig.__dataclass_fields__)
    unknown_fields = sorted(set(data) - allowed_fields - {"task"})
    if unknown_fields:
        raise ValueError(
            "Unknown patch-coordinate heatmap config field(s): "
            + ", ".join(unknown_fields)
        )
    config_data = {key: value for key, value in data.items() if key in allowed_fields}
    try:
        return PatchCoordinateHeatmapConfig(**config_data)
    except TypeError as exc:
        raise ValueError(f"Invalid patch-coordinate heatmap config: {exc}") from exc


def patch_coordinate_heatmap_scoring_config_to_dict(
    config: PatchCoordinateHeatmapScoringConfig,
) -> dict[str, Any]:
    data = asdict(config)
    data["task"] = "patch_coordinate_heatmap_scoring"
    return data


def patch_coordinate_heatmap_scoring_config_from_dict(
    data: dict[str, Any],
) -> PatchCoordinateHeatmapScoringConfig:
    task = data.get("task", "patch_coordinate_heatmap_scoring")
    if task != "patch_coordinate_heatmap_scoring":
        raise ValueError(
            "Config task must be 'patch_coordinate_heatmap_scoring'. "
            f"Got: {task}"
        )

    allowed_fields = set(PatchCoordinateHeatmapScoringConfig.__dataclass_fields__)
    unknown_fields = sorted(set(data) - allowed_fields - {"task"})
    if unknown_fields:
        raise ValueError(
            "Unknown patch-coordinate heatmap scoring config field(s): "
            + ", ".join(unknown_fields)
        )
    config_data = {key: value for key, value in data.items() if key in allowed_fields}
    try:
        return PatchCoordinateHeatmapScoringConfig(**config_data)
    except TypeError as exc:
        raise ValueError(
            f"Invalid patch-coordinate heatmap scoring config: {exc}"
        ) from exc


def load_patch_coordinate_heatmap_config(
    path: str | Path,
) -> PatchCoordinateHeatmapConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Patch-coordinate heatmap config must be a JSON object.")
    return patch_coordinate_heatmap_config_from_dict(data)


def load_patch_coordinate_heatmap_scoring_config(
    path: str | Path,
) -> PatchCoordinateHeatmapScoringConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            "Patch-coordinate heatmap scoring config must be a JSON object."
        )
    return patch_coordinate_heatmap_scoring_config_from_dict(data)


def save_patch_coordinate_heatmap_config(
    config: PatchCoordinateHeatmapConfig,
    path: str | Path,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = patch_coordinate_heatmap_config_to_dict(config)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def save_patch_coordinate_heatmap_scoring_config(
    config: PatchCoordinateHeatmapScoringConfig,
    path: str | Path,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = patch_coordinate_heatmap_scoring_config_to_dict(config)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
