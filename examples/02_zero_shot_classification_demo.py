from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathvlm_litebench.data import load_patch_images
from pathvlm_litebench.evaluation import zero_shot_predict
from pathvlm_litebench.models import create_model


def create_demo_images(output_dir: str | Path) -> Path:
    """
    Create a small demo image folder for smoke testing.

    These are not pathology images. They are only used to verify that
    the zero-shot classification pipeline works end-to-end on a laptop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_images = {
        "patch_red.png": "red",
        "patch_blue.png": "blue",
        "patch_white.png": "white",
        "patch_green.png": "green",
        "patch_black.png": "black",
    }

    for filename, color in demo_images.items():
        image_path = output_dir / filename
        if not image_path.exists():
            Image.new("RGB", (224, 224), color=color).save(image_path)

    return output_dir


def build_class_prompts(
    class_names: list[str],
    class_prompts: list[str] | None = None,
) -> list[str]:
    """
    Build text prompts for zero-shot classification.
    """
    if class_prompts is not None:
        if len(class_prompts) != len(class_names):
            raise ValueError(
                f"class_prompts and class_names must have the same length: "
                f"{len(class_prompts)} vs {len(class_names)}"
            )
        return class_prompts

    return [f"a histopathology image of {class_name}" for class_name in class_names]


def run_zero_shot_classification_demo(
    image_dir: str | Path | None = None,
    class_names: list[str] | None = None,
    class_prompts: list[str] | None = None,
    top_k: int = 3,
    model: str = "clip",
) -> None:
    """
    Run a minimal patch-level zero-shot classification demo.
    """
    if image_dir is None:
        image_dir = create_demo_images(Path("examples") / "demo_patches")
        print(f"[INFO] No image_dir provided. Created demo images at: {image_dir}")
    else:
        image_dir = Path(image_dir)

    if class_names is None or len(class_names) == 0:
        class_names = ["red", "blue", "white", "black", "green"]

    if class_prompts is None and class_names == ["red", "blue", "white", "black", "green"]:
        class_prompts = [
            "a red image",
            "a blue image",
            "a white image",
            "a black image",
            "a green image",
        ]

    class_prompts = build_class_prompts(
        class_names=class_names,
        class_prompts=class_prompts,
    )

    print("[INFO] Loading patch images...")
    images, image_paths = load_patch_images(image_dir)
    print(f"[INFO] Loaded {len(images)} images from {image_dir}")

    print("[INFO] Class names:")
    for name, prompt in zip(class_names, class_prompts):
        print(f"  - {name}: {prompt}")

    print(f"[INFO] Loading model: {model}")
    vlm = create_model(model)

    print("[INFO] Encoding images...")
    image_embeddings = vlm.encode_images(images)

    print("[INFO] Encoding class prompts...")
    class_embeddings = vlm.encode_text(class_prompts)

    print("[INFO] Running zero-shot classification...")
    results = zero_shot_predict(
        image_embeddings=image_embeddings,
        class_embeddings=class_embeddings,
        class_names=class_names,
        top_k=top_k,
    )

    print("\n========== Zero-Shot Classification Results ==========")

    for image_path, result in zip(image_paths, results):
        print(f"\nImage: {image_path}")
        print(
            f"Predicted: {result['predicted_label']} "
            f"(confidence={result['confidence']:.4f})"
        )

        print("Top predictions:")
        for rank, item in enumerate(result["top_predictions"], start=1):
            print(
                f"  Top {rank}: "
                f"class={item['class_name']}, "
                f"probability={item['probability']:.4f}, "
                f"logit={item['logit']:.4f}"
            )

    print("\n[INFO] Zero-shot classification demo finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal patch-level zero-shot classification demo."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to a folder containing patch images. If omitted, demo images will be created.",
    )

    parser.add_argument(
        "--class_names",
        nargs="+",
        default=None,
        help="Class names for zero-shot classification.",
    )

    parser.add_argument(
        "--class_prompts",
        nargs="+",
        default=None,
        help="Text prompts for each class. Must match class_names length.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top class predictions to show for each image.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        help="Registered model key or Hugging Face model name. Example: 'clip' or 'openai/clip-vit-base-patch32'.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_zero_shot_classification_demo(
        image_dir=args.image_dir,
        class_names=args.class_names,
        class_prompts=args.class_prompts,
        top_k=args.top_k,
        model=args.model,
    )
