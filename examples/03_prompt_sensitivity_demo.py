from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathvlm_litebench.config import load_benchmark_config


def create_demo_images(output_dir: str | Path) -> Path:
    """
    Create a small demo image folder for smoke testing.

    These are not pathology images. They are only used to verify that
    the prompt sensitivity pipeline works end-to-end on a laptop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    demo_images = {
        "patch_red.png": "red",
        "patch_darkred.png": (120, 0, 0),
        "patch_blue.png": "blue",
        "patch_darkblue.png": (0, 0, 120),
        "patch_white.png": "white",
        "patch_black.png": "black",
        "patch_gray.png": "gray",
    }

    for filename, color in demo_images.items():
        image_path = output_dir / filename
        if not image_path.exists():
            Image.new("RGB", (224, 224), color=color).save(image_path)

    return output_dir


def get_default_prompt_groups() -> tuple[list[str], list[list[str]]]:
    """
    Return default concept names and prompt variants for smoke testing.
    """
    concept_names = ["red_like", "blue_like", "dark_like"]

    prompt_texts_by_concept = [
        [
            "a red image",
            "a bright red patch",
            "an image with red color",
        ],
        [
            "a blue image",
            "a bright blue patch",
            "an image with blue color",
        ],
        [
            "a black image",
            "a dark patch",
            "an image with black color",
        ],
    ]

    return concept_names, prompt_texts_by_concept


def run_prompt_sensitivity_demo(
    image_dir: str | Path | None = None,
    top_k: int = 3,
    model: str = "clip",
    device: str = "auto",
    use_pathology_prompts: bool = False,
    concepts: list[str] | None = None,
    save_report: bool = False,
    report_dir: str | Path = "outputs/prompt_sensitivity_demo",
) -> None:
    """
    Run a minimal patch-level prompt sensitivity analysis demo.
    """
    from pathvlm_litebench.data import load_patch_images
    from pathvlm_litebench.evaluation import analyze_prompt_sensitivity
    from pathvlm_litebench.models import create_model
    from pathvlm_litebench.prompts import build_prompt_groups
    from pathvlm_litebench.visualization import (
        save_prompt_sensitivity_summary_csv,
        save_prompt_sensitivity_details_csv,
        save_prompt_sensitivity_metrics_json,
    )

    using_demo_images = image_dir is None

    if using_demo_images:
        image_dir = create_demo_images(Path("examples") / "demo_patches")
        print(f"[INFO] No image_dir provided. Created demo images at: {image_dir}")
    else:
        image_dir = Path(image_dir)

    if use_pathology_prompts:
        concept_names, prompt_texts_by_concept = build_prompt_groups(concepts)
        print("[INFO] Using built-in pathology prompt templates.")
        if using_demo_images:
            print(
                "[INFO] Built-in demo images are smoke tests and are not pathology images. "
                "For meaningful pathology prompt analysis, pass --image_dir path/to/your_patch_folder."
            )
    else:
        concept_names, prompt_texts_by_concept = get_default_prompt_groups()
        print("[INFO] Using default color prompt groups for smoke testing.")

    print("[INFO] Loading patch images...")
    images, image_paths = load_patch_images(image_dir)
    print(f"[INFO] Loaded {len(images)} images from {image_dir}")

    print("[INFO] Concepts and prompt variants:")
    for concept_name, prompt_texts in zip(concept_names, prompt_texts_by_concept):
        print(f"  - {concept_name}")
        for prompt in prompt_texts:
            print(f"      * {prompt}")

    print(f"[INFO] Loading model: {model}")
    print(f"[INFO] Requested device: {device}")
    vlm = create_model(model, device=device)
    if hasattr(vlm, "device"):
        print(f"[INFO] Using device: {vlm.device}")

    print("[INFO] Encoding images...")
    image_embeddings = vlm.encode_images(images)

    print("[INFO] Encoding prompt variants...")
    prompt_embeddings_by_concept = []

    for prompt_texts in prompt_texts_by_concept:
        prompt_embeddings = vlm.encode_text(prompt_texts)
        prompt_embeddings_by_concept.append(prompt_embeddings)

    print("[INFO] Analyzing prompt sensitivity...")
    results = analyze_prompt_sensitivity(
        image_embeddings=image_embeddings,
        prompt_embeddings_by_concept=prompt_embeddings_by_concept,
        concept_names=concept_names,
        prompt_texts_by_concept=prompt_texts_by_concept,
        top_k=top_k,
    )

    print("\n========== Prompt Sensitivity Results ==========")

    for result in results:
        print(f"\nConcept: {result['concept_name']}")
        print(f"Number of prompts: {result['num_prompts']}")
        print(f"Mean top-k overlap: {result['mean_topk_overlap']:.4f}")
        print(f"Mean similarity std: {result['mean_similarity_std']:.4f}")

        print("Prompt-level top-k results:")
        for prompt_result in result["prompt_results"]:
            prompt_text = prompt_result["prompt_text"]
            top_indices = prompt_result["top_indices"]
            top_scores = prompt_result["top_scores"]

            print(f"  Prompt: {prompt_text}")
            for rank, (index, score) in enumerate(zip(top_indices, top_scores), start=1):
                image_path = image_paths[index]
                print(
                    f"    Top {rank}: "
                    f"index={index}, "
                    f"score={score:.4f}, "
                    f"path={image_path}"
                )

    if save_report:
        report_dir = Path(report_dir)
        summary_path = report_dir / "prompt_sensitivity_summary.csv"
        details_path = report_dir / "prompt_sensitivity_details.csv"
        metrics_path = report_dir / "prompt_sensitivity_metrics.json"

        saved_summary_path = save_prompt_sensitivity_summary_csv(
            results=results,
            output_csv_path=summary_path,
        )
        saved_details_path = save_prompt_sensitivity_details_csv(
            results=results,
            output_csv_path=details_path,
        )

        metadata = {
            "model": model,
            "device": device,
            "image_dir": str(image_dir) if image_dir is not None else None,
            "top_k": top_k,
            "use_pathology_prompts": use_pathology_prompts,
            "concepts": concept_names,
            "num_images": len(image_paths),
            "num_concepts": len(concept_names),
        }
        saved_metrics_path = save_prompt_sensitivity_metrics_json(
            results=results,
            output_json_path=metrics_path,
            metadata=metadata,
        )

        print(f"\n[INFO] Saved prompt sensitivity summary: {saved_summary_path}")
        print(f"[INFO] Saved prompt sensitivity details: {saved_details_path}")
        print(f"[INFO] Saved prompt sensitivity metrics: {saved_metrics_path}")

    print("\n[INFO] Prompt sensitivity demo finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal patch-level prompt sensitivity analysis demo."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON benchmark config file. Command-line arguments override config values when explicitly provided.",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to a folder containing patch images. If omitted, demo images will be created.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of top retrieved images used to measure overlap.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Registered model key or Hugging Face model name. Example: 'clip' or 'openai/clip-vit-base-patch32'.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device for model inference. Use 'auto' to select CUDA if available, otherwise CPU.",
    )

    parser.add_argument(
        "--use_pathology_prompts",
        action="store_true",
        help="Use built-in pathology prompt templates instead of color smoke-test prompts.",
    )

    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        help="Pathology concepts to use when --use_pathology_prompts is enabled. Example: tumor normal necrosis.",
    )

    parser.add_argument(
        "--save_report",
        action="store_true",
        help="Save prompt sensitivity results as CSV/JSON reports.",
    )

    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Directory for saving prompt sensitivity report files.",
    )

    return parser.parse_args()


def merge_args_with_config(args: argparse.Namespace) -> dict:
    """
    Merge command-line arguments with optional benchmark config values.
    """
    default_values = {
        "model": "clip",
        "device": "auto",
        "top_k": 3,
        "use_pathology_prompts": False,
        "save_report": False,
        "report_dir": "outputs/prompt_sensitivity_demo",
    }

    if args.config is None:
        return {
            "image_dir": args.image_dir,
            "top_k": args.top_k if args.top_k is not None else default_values["top_k"],
            "model": args.model if args.model is not None else default_values["model"],
            "device": args.device if args.device is not None else default_values["device"],
            "use_pathology_prompts": args.use_pathology_prompts,
            "concepts": args.concepts,
            "save_report": args.save_report,
            "report_dir": (
                args.report_dir
                if args.report_dir is not None
                else default_values["report_dir"]
            ),
        }

    config = load_benchmark_config(args.config)
    if config.task != "prompt_sensitivity":
        raise ValueError(
            f"Config task must be 'prompt_sensitivity' for this demo, got '{config.task}'."
        )

    return {
        "image_dir": args.image_dir if args.image_dir is not None else config.image_dir,
        "top_k": args.top_k if args.top_k is not None else config.top_k,
        "model": args.model if args.model is not None else config.model,
        "device": args.device if args.device is not None else config.device,
        "use_pathology_prompts": (
            args.use_pathology_prompts
            if args.use_pathology_prompts
            else config.use_pathology_prompts
        ),
        "concepts": args.concepts if args.concepts is not None else config.concepts,
        "save_report": args.save_report if args.save_report else config.save_report,
        "report_dir": (
            args.report_dir
            if args.report_dir is not None
            else config.report_dir
        ),
    }


if __name__ == "__main__":
    args = parse_args()
    run_kwargs = merge_args_with_config(args)

    if args.config is not None:
        print(f"[INFO] Loaded benchmark config: {args.config}")

    print("[INFO] Final run configuration:")
    print("  task: prompt_sensitivity")
    print(f"  model: {run_kwargs['model']}")
    print(f"  device: {run_kwargs['device']}")
    print(f"  image_dir: {run_kwargs['image_dir']}")
    print(f"  top_k: {run_kwargs['top_k']}")
    print(f"  use_pathology_prompts: {run_kwargs['use_pathology_prompts']}")
    print(f"  concepts: {run_kwargs['concepts']}")
    print(f"  save_report: {run_kwargs['save_report']}")
    print(f"  report_dir: {run_kwargs['report_dir']}")

    run_prompt_sensitivity_demo(**run_kwargs)
