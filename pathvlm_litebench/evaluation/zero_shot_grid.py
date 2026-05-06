from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable

from pathvlm_litebench.visualization.report_summary import (
    save_experiment_comparison_summary,
)


VALID_DEVICES = {"auto", "cpu", "cuda"}


@dataclass(frozen=True)
class PromptPair:
    key: str
    class_prompts: list[str]


@dataclass(frozen=True)
class ZeroShotGridConfig:
    models: list[str]
    class_names: list[str]
    prompt_pairs: list[PromptPair]
    device: str = "auto"
    manifest: str | None = None
    image_root: str | None = None
    image_dir: str | None = None
    split: str | None = None
    max_images: int | None = None
    top_k: int = 2
    output_root: str = "outputs/zero_shot_prompt_grid"
    save_comparison: bool = True
    comparison_output: str | None = None
    write_logs: bool = True


@dataclass(frozen=True)
class ZeroShotGridRun:
    model: str
    prompt_key: str
    class_names: list[str]
    class_prompts: list[str]
    device: str
    report_dir: Path
    manifest: str | None = None
    image_root: str | None = None
    image_dir: str | None = None
    split: str | None = None
    max_images: int | None = None
    top_k: int = 2
    log_path: Path | None = None

    @property
    def run_name(self) -> str:
        return f"{_slugify(self.model)}_{self.prompt_key}"


Runner = Callable[[ZeroShotGridRun], None]


def _slugify(value: str) -> str:
    text = value.strip().replace("\\", "/").split("/")[-1].lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = text.strip("._-")
    return text or "run"


def _require_string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list of strings.")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{field_name} must be a non-empty list of strings.")
    return [item.strip() for item in value]


def _load_prompt_pair(item: object, class_count: int) -> PromptPair:
    if not isinstance(item, dict):
        raise ValueError("Each prompt pair must be an object.")

    key = item.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Each prompt pair must include a non-empty key.")

    class_prompts = _require_string_list(item.get("class_prompts"), "class_prompts")
    if len(class_prompts) != class_count:
        raise ValueError(
            "Each prompt pair must contain exactly one prompt per class name."
        )

    return PromptPair(key=_slugify(key), class_prompts=class_prompts)


def load_zero_shot_grid_config(path: str | Path) -> ZeroShotGridConfig:
    """
    Load a zero-shot prompt-grid configuration from JSON.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Zero-shot grid config must be a JSON object.")

    task = data.get("task", "zero_shot_grid")
    if task != "zero_shot_grid":
        raise ValueError(f"Config task must be 'zero_shot_grid'. Got: {task}")

    models = _require_string_list(data.get("models"), "models")
    class_names = _require_string_list(data.get("class_names"), "class_names")

    prompt_pair_items = data.get("prompt_pairs")
    if not isinstance(prompt_pair_items, list) or not prompt_pair_items:
        raise ValueError("prompt_pairs must be a non-empty list.")
    prompt_pairs = [
        _load_prompt_pair(item, class_count=len(class_names))
        for item in prompt_pair_items
    ]

    device = data.get("device", "auto")
    if device not in VALID_DEVICES:
        valid = ", ".join(sorted(VALID_DEVICES))
        raise ValueError(f"device must be one of: {valid}. Got: {device}")

    max_images = data.get("max_images")
    if max_images is not None and (not isinstance(max_images, int) or max_images <= 0):
        raise ValueError("max_images must be a positive integer when provided.")

    top_k = data.get("top_k", 2)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    manifest = data.get("manifest")
    image_dir = data.get("image_dir")
    if not manifest and not image_dir:
        raise ValueError("Either manifest or image_dir must be provided.")

    return ZeroShotGridConfig(
        models=models,
        class_names=class_names,
        prompt_pairs=prompt_pairs,
        device=device,
        manifest=manifest,
        image_root=data.get("image_root"),
        image_dir=image_dir,
        split=data.get("split"),
        max_images=max_images,
        top_k=top_k,
        output_root=data.get("output_root", "outputs/zero_shot_prompt_grid"),
        save_comparison=bool(data.get("save_comparison", True)),
        comparison_output=data.get("comparison_output"),
        write_logs=bool(data.get("write_logs", True)),
    )


def expand_zero_shot_grid_runs(config: ZeroShotGridConfig) -> list[ZeroShotGridRun]:
    """
    Expand a zero-shot grid config into concrete model/prompt runs.
    """
    output_root = Path(config.output_root)
    runs: list[ZeroShotGridRun] = []
    for model in config.models:
        model_dir = output_root / _slugify(model)
        for prompt_pair in config.prompt_pairs:
            report_dir = model_dir / prompt_pair.key
            log_path = report_dir / "run.log" if config.write_logs else None
            runs.append(
                ZeroShotGridRun(
                    model=model,
                    prompt_key=prompt_pair.key,
                    class_names=config.class_names,
                    class_prompts=prompt_pair.class_prompts,
                    device=config.device,
                    report_dir=report_dir,
                    manifest=config.manifest,
                    image_root=config.image_root,
                    image_dir=config.image_dir,
                    split=config.split,
                    max_images=config.max_images,
                    top_k=config.top_k,
                    log_path=log_path,
                )
            )
    return runs


def build_zero_shot_grid_command(run: ZeroShotGridRun) -> list[str]:
    """
    Build the subprocess command for one zero-shot prompt-grid run.
    """
    demo_script = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "02_zero_shot_classification_demo.py"
    )
    command = [
        sys.executable,
        str(demo_script),
    ]
    if run.manifest is not None:
        command.extend(["--manifest", run.manifest])
    if run.image_root is not None:
        command.extend(["--image_root", run.image_root])
    if run.image_dir is not None:
        command.extend(["--image_dir", run.image_dir])
    if run.split is not None:
        command.extend(["--split", run.split])
    if run.max_images is not None:
        command.extend(["--max_images", str(run.max_images)])

    command.extend(
        [
            "--model",
            run.model,
            "--device",
            run.device,
            "--class_names",
            *run.class_names,
            "--class_prompts",
            *run.class_prompts,
            "--top_k",
            str(run.top_k),
            "--save_report",
            "--report_dir",
            str(run.report_dir),
        ]
    )
    return command


def run_zero_shot_grid_subprocess(run: ZeroShotGridRun) -> None:
    """
    Run one zero-shot grid item through the existing demo script.
    """
    command = build_zero_shot_grid_command(run)
    if run.log_path is not None:
        run.log_path.parent.mkdir(parents=True, exist_ok=True)
        with run.log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
    else:
        result = subprocess.run(command, check=False)

    if result.returncode != 0:
        log_hint = f" See log: {run.log_path}" if run.log_path is not None else ""
        raise RuntimeError(
            f"Zero-shot grid run failed: {run.run_name}.{log_hint}"
        )

    metrics_path = run.report_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Zero-shot grid run did not write metrics: {metrics_path}")


def run_zero_shot_grid(
    config: ZeroShotGridConfig,
    *,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> dict[str, object]:
    """
    Run or preview a zero-shot prompt grid.
    """
    runs = expand_zero_shot_grid_runs(config)
    if dry_run:
        return {
            "runs": runs,
            "comparison_path": None,
        }

    selected_runner = runner or run_zero_shot_grid_subprocess
    for run in runs:
        selected_runner(run)

    comparison_path: str | None = None
    if config.save_comparison:
        output_root = Path(config.output_root)
        comparison_output = (
            Path(config.comparison_output)
            if config.comparison_output is not None
            else output_root / "zero_shot_grid_comparison.md"
        )
        comparison_path = save_experiment_comparison_summary(
            task="zero-shot",
            report_dirs=[run.report_dir for run in runs],
            run_names=[run.run_name for run in runs],
            output_path=comparison_output,
        )

    return {
        "runs": runs,
        "comparison_path": comparison_path,
    }
