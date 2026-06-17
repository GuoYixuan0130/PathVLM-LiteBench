from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ... import version
from ..constants import DEMO_SCRIPTS
from ...models.registry import list_available_models


def _handle_version() -> int:
    print(f"PathVLM-LiteBench version {version}")
    return 0


def _handle_models() -> int:
    print("Registered models:")
    for item in list_available_models():
        print(f"- key: {item['key']}")
        print(f"  model_name: {item['model_name']}")
        print(f"  implemented: {item['implemented']}")
        print(f"  description: {item['description']}")
    return 0


def _handle_demos() -> int:
    print("Tip: run a bundled demo directly with: pathvlm-litebench demo <name>")
    print("     (names: " + ", ".join(DEMO_SCRIPTS) + ")")
    print()
    print("Available demo commands:")
    print("python examples/01_patch_text_retrieval_demo.py --model clip")
    print("python examples/02_zero_shot_classification_demo.py --model clip")
    print("python examples/03_prompt_sensitivity_demo.py --model clip")
    print("python examples/04_retrieval_metrics_demo.py")
    print("python examples/05_patch_coordinate_heatmap_demo.py")
    print("python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_clip_sample.json")
    print("python examples/02_zero_shot_classification_demo.py --config configs/zero_shot_mhist_plip_sample.json")
    print("pathvlm-litebench validate-config configs/zero_shot_mhist_clip_sample.json")
    print("pathvlm-litebench validate-config configs/zero_shot_mhist_plip_sample.json")
    print("pathvlm-litebench validate-config configs/zero_shot_prompt_grid_mhist_sample.json")
    print("pathvlm-litebench run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --dry-run")
    print("pathvlm-litebench run-zero-shot-grid --config configs/zero_shot_prompt_grid_mhist_sample.json --output-root outputs/zero_shot_prompt_grid_mhist_sample_run")
    print("pathvlm-litebench validate-config configs/patch_coordinate_heatmap_demo_config.json")
    print("pathvlm-litebench validate-config configs/patch_coordinate_heatmap_scoring_demo_config.json")
    print("pathvlm-litebench validate-config configs/patch_coordinate_heatmap_prompt_set_demo_config.json")
    print("pathvlm-litebench render-coordinate-heatmap --config configs/patch_coordinate_heatmap_demo_config.json")
    print("pathvlm-litebench score-coordinate-heatmap --config configs/patch_coordinate_heatmap_scoring_demo_config.json --dry-run")
    print("pathvlm-litebench score-coordinate-heatmap --config configs/patch_coordinate_heatmap_scoring_demo_config.json")
    print("pathvlm-litebench score-coordinate-heatmap-prompt-set --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json --dry-run")
    print("pathvlm-litebench score-coordinate-heatmap-prompt-set --config configs/patch_coordinate_heatmap_prompt_set_demo_config.json")
    print("pathvlm-litebench compare-coordinate-heatmap-scores --score-csvs outputs/patch_coordinate_heatmap_scored_tumor/scores.csv outputs/patch_coordinate_heatmap_scored_lymphocyte/scores.csv --run-names tumor lymphocyte --output-csv outputs/patch_coordinate_heatmap_comparison/score_summary.csv --output-md outputs/patch_coordinate_heatmap_comparison/score_summary.md")
    print("pathvlm-litebench render-coordinate-heatmap --manifest dataset/patch_coordinates/coordinate_manifest.csv --score-csv outputs/patch_coordinate_heatmap_demo/scores.csv --output outputs/patch_coordinate_heatmap_demo/heatmap.png")
    print("pathvlm-litebench score-coordinate-heatmap --manifest dataset/patch_coordinates/coordinate_manifest.csv --prompt \"a histopathology image of tumor tissue\" --output-dir outputs/patch_coordinate_heatmap_scored --model clip")
    print("pathvlm-litebench compare-models --manifest dataset/CRC_VAL_HE_100_sample_manifest.csv --models clip plip conch --class-names \"adipose tissue\" background debris lymphocytes mucus \"smooth muscle\" \"normal colon mucosa\" \"cancer-associated stroma\" \"colorectal adenocarcinoma epithelium\" --output-dir outputs/model_comparison")
    print("pathvlm-litebench build-imagefolder-manifest --image-dir path/to/imagefolder --output dataset/imagefolder_manifest.csv")
    return 0


def _examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / "examples"


def _print_demo_list() -> None:
    print("Available demos:")
    for name, script in DEMO_SCRIPTS.items():
        print(f"- {name}  (examples/{script})")
    print()
    print("Run one with: pathvlm-litebench demo <name> [demo args...]")
    print("Example: pathvlm-litebench demo retrieval --model clip --device auto")
    if not _examples_dir().exists():
        print()
        print(
            "Note: the bundled demo scripts ship with a source checkout, not the "
            "PyPI package."
        )
        print(
            "Clone https://github.com/GuoYixuan0130/PathVLM-LiteBench to run them."
        )


def _handle_demo(args: argparse.Namespace) -> int:
    if args.name is None:
        _print_demo_list()
        return 0

    script = DEMO_SCRIPTS.get(args.name)
    if script is None:
        print(f"Error: unknown demo '{args.name}'.")
        print(f"Available demos: {', '.join(DEMO_SCRIPTS)}")
        return 1

    script_path = _examples_dir() / script
    if not script_path.exists():
        print(f"Error: demo script not found: {script_path}")
        print("The 'demo' command runs from a source checkout of the repository.")
        return 1

    command = [sys.executable, str(script_path), *args.demo_args]
    return subprocess.call(command)
