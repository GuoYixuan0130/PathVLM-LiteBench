from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from pathvlm_litebench.data import load_patch_images
from pathvlm_litebench.evaluation import analyze_prompt_sensitivity
from pathvlm_litebench.models import CLIPWrapper


def create_demo_images(output_dir: str | Path) -> Path:
    """
    Create a small demo image folder for smoke testing.

    These are not pathology images. They are only used to verify that
    the prompt sensitivity pipeline works end-to-end on a laptop.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    model_name: str = "openai/clip-vit-base-patch32",
) -> None:
    """
    Run a minimal patch-level prompt sensitivity analysis demo.
    """
    if image_dir is None:
        image_dir = create_demo_images(Path("examples") / "demo_patches")
        print(f"[INFO] No image_dir provided. Created demo images at: {image_dir}")
    else:
        image_dir = Path(image_dir)

    concept_names, prompt_texts_by_concept = get_default_prompt_groups()

    print("[INFO] Loading patch images...")
    images, image_paths = load_patch_images(image_dir)
    print(f"[INFO] Loaded {len(images)} images from {image_dir}")

    print("[INFO] Concepts and prompt variants:")
    for concept_name, prompt_texts in zip(concept_names, prompt_texts_by_concept):
        print(f"  - {concept_name}")
        for prompt in prompt_texts:
            print(f"      * {prompt}")

    print("[INFO] Loading CLIP model...")
    model = CLIPWrapper(model_name=model_name)

    print("[INFO] Encoding images...")
    image_embeddings = model.encode_images(images)

    print("[INFO] Encoding prompt variants...")
    prompt_embeddings_by_concept = []

    for prompt_texts in prompt_texts_by_concept:
        prompt_embeddings = model.encode_text(prompt_texts)
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

    print("\n[INFO] Prompt sensitivity demo finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal patch-level prompt sensitivity analysis demo."
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
        default=3,
        help="Number of top retrieved images used to measure overlap.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face model name for CLIP-style model.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_prompt_sensitivity_demo(
        image_dir=args.image_dir,
        top_k=args.top_k,
        model_name=args.model_name,
    )
