from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


VALID_TASKS = {"retrieval", "zero_shot", "prompt_sensitivity"}
VALID_DEVICES = {"auto", "cpu", "cuda"}


@dataclass(eq=True)
class BenchmarkConfig:
    task: str
    model: str = "clip"
    device: str = "auto"
    image_dir: str | None = None
    prompts: list[str] | None = None
    class_names: list[str] | None = None
    class_prompts: list[str] | None = None
    concepts: list[str] | None = None
    top_k: int = 5
    use_cache: bool = False
    cache_dir: str = "outputs/cache"
    save_visualization: bool = False
    save_html_report: bool = False
    output_dir: str = "outputs/retrieval_demo"
    html_report_path: str = "outputs/retrieval_demo/retrieval_report.html"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.task not in VALID_TASKS:
            valid = ", ".join(sorted(VALID_TASKS))
            raise ValueError(f"task must be one of: {valid}. Got: {self.task}")

        if not self.model:
            raise ValueError("model must not be empty.")

        if self.device not in VALID_DEVICES:
            valid = ", ".join(sorted(VALID_DEVICES))
            raise ValueError(f"device must be one of: {valid}. Got: {self.device}")

        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0. Got: {self.top_k}")

        if self.task == "retrieval":
            _validate_optional_string_list(self.prompts, "prompts")

        if self.task == "zero_shot":
            _validate_optional_string_list(self.class_names, "class_names")
            _validate_optional_string_list(self.class_prompts, "class_prompts")

            if self.class_names is not None and self.class_prompts is not None:
                if len(self.class_names) != len(self.class_prompts):
                    raise ValueError(
                        "class_names and class_prompts must have the same length when both are provided."
                    )

        if self.task == "prompt_sensitivity":
            _validate_optional_string_list(self.concepts, "concepts")


def _validate_optional_string_list(value: list[str] | None, field_name: str) -> None:
    if value is None:
        return

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings if provided.")

    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings if provided.")


def benchmark_config_to_dict(config: BenchmarkConfig) -> dict[str, Any]:
    """
    Convert BenchmarkConfig to a plain dictionary.
    """
    return asdict(config)


def benchmark_config_from_dict(data: dict[str, Any]) -> BenchmarkConfig:
    """
    Build BenchmarkConfig from a plain dictionary.
    """
    return BenchmarkConfig(**data)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """
    Load benchmark configuration from a JSON file.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return benchmark_config_from_dict(data)


def save_benchmark_config(config: BenchmarkConfig, path: str | Path) -> str:
    """
    Save benchmark configuration to a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = benchmark_config_to_dict(config)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def create_default_retrieval_config() -> BenchmarkConfig:
    """
    Create a default retrieval benchmark configuration.
    """
    return BenchmarkConfig(
        task="retrieval",
        model="clip",
        device="auto",
        image_dir=None,
        prompts=[
            "a histopathology image of tumor tissue",
            "a histopathology image of normal tissue",
            "a histopathology image showing necrosis",
        ],
        top_k=5,
        use_cache=True,
        cache_dir="outputs/cache",
        save_visualization=True,
        save_html_report=True,
        output_dir="outputs/retrieval_demo",
        html_report_path="outputs/retrieval_demo/retrieval_report.html",
    )
