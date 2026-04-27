from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathvlm_litebench.data import (
    load_patch_images,
    save_embeddings,
    load_embeddings,
    save_metadata,
    load_metadata,
)
from pathvlm_litebench.models import CLIPWrapper
from pathvlm_litebench.retrieval import retrieve_topk_images
from pathvlm_litebench.visualization import (
    save_topk_image_grids,
    save_retrieval_html_report,
)


def create_demo_images(output_dir: str | Path) -> Path:
    """
    Create a small demo image folder for smoke testing.

    These are not pathology images. They are only used to verify that
    the image-text retrieval pipeline works end-to-end on a laptop.
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


def run_patch_text_retrieval_demo(
    image_dir: str | Path | None = None,
    prompts: list[str] | None = None,
    top_k: int = 3,
    model_name: str = "openai/clip-vit-base-patch32",
    save_visualization: bool = False,
    output_dir: str | Path = "outputs/retrieval_demo",
    use_cache: bool = False,
    cache_dir: str | Path = "outputs/cache",
    save_html_report: bool = False,
    html_report_path: str | Path = "outputs/retrieval_demo/retrieval_report.html",
) -> None:
    """
    Run a minimal patch-level image-text retrieval demo.
    """
    if image_dir is None:
        image_dir = create_demo_images(Path("examples") / "demo_patches")
        print(f"[INFO] No image_dir provided. Created demo images at: {image_dir}")
    else:
        image_dir = Path(image_dir)

    if prompts is None or len(prompts) == 0:
        prompts = [
            "a red image",
            "a blue image",
            "a white image",
        ]

    print("[INFO] Loading patch images...")
    images, image_paths = load_patch_images(image_dir)

    print(f"[INFO] Loaded {len(images)} images from {image_dir}")

    print("[INFO] Loading CLIP model...")
    model = CLIPWrapper(model_name=model_name)

    if use_cache:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        image_embedding_cache_path = cache_dir / "image_embeddings.pt"
        image_paths_cache_path = cache_dir / "image_paths.json"

        print(f"[INFO] Embedding cache enabled.")
        print(f"[INFO] Image embedding cache: {image_embedding_cache_path}")
        print(f"[INFO] Image paths cache: {image_paths_cache_path}")

        cache_exists = image_embedding_cache_path.exists() and image_paths_cache_path.exists()

        if cache_exists:
            cached_image_paths = load_metadata(image_paths_cache_path)

            if cached_image_paths == image_paths:
                print("[INFO] Cache hit: loading image embeddings from cache.")
                image_embeddings = load_embeddings(image_embedding_cache_path)
            else:
                print("[INFO] Cache mismatch: image paths changed. Re-encoding images.")
                image_embeddings = model.encode_images(images)
                save_embeddings(image_embeddings, image_embedding_cache_path)
                save_metadata(image_paths, image_paths_cache_path)
                print("[INFO] Updated image embedding cache.")
        else:
            print("[INFO] Cache miss: encoding images and saving cache.")
            image_embeddings = model.encode_images(images)
            save_embeddings(image_embeddings, image_embedding_cache_path)
            save_metadata(image_paths, image_paths_cache_path)
            print("[INFO] Saved image embedding cache.")
    else:
        print("[INFO] Embedding cache disabled.")
        print("[INFO] Encoding images...")
        image_embeddings = model.encode_images(images)

    print("[INFO] Encoding text prompts...")
    text_embeddings = model.encode_text(prompts)

    print("[INFO] Retrieving top-k images...")
    results = retrieve_topk_images(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_paths=image_paths,
        top_k=top_k,
    )

    print("\n========== Retrieval Results ==========")

    for prompt, prompt_results in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        for rank, item in enumerate(prompt_results, start=1):
            print(
                f"  Top {rank}: "
                f"index={item['index']}, "
                f"score={item['score']:.4f}, "
                f"path={item.get('path', 'N/A')}"
            )

    if save_visualization:
        print("\n[INFO] Saving top-k visualization grids...")
        saved_paths = save_topk_image_grids(
            prompts=prompts,
            retrieval_results=results,
            output_dir=output_dir,
        )

        for path in saved_paths:
            print(f"[INFO] Saved visualization: {path}")

    if save_html_report:
        print("\n[INFO] Saving HTML retrieval report...")
        saved_html_path = save_retrieval_html_report(
            prompts=prompts,
            retrieval_results=results,
            output_html_path=html_report_path,
        )
        print(f"[INFO] Saved HTML report: {saved_html_path}")

    print("\n[INFO] Demo finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal patch-level image-text retrieval demo."
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to a folder containing patch images. If omitted, demo images will be created.",
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Text prompts for image-text retrieval.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top images to retrieve for each prompt.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face model name for CLIP-style model.",
    )

    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Save top-k retrieval visualization grids.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/retrieval_demo",
        help="Directory for saving visualization outputs.",
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cached image embeddings if available.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="outputs/cache",
        help="Directory for saving/loading embedding cache.",
    )

    parser.add_argument(
        "--save_html_report",
        action="store_true",
        help="Save retrieval results as an HTML report.",
    )

    parser.add_argument(
        "--html_report_path",
        type=str,
        default="outputs/retrieval_demo/retrieval_report.html",
        help="Output path for the HTML retrieval report.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_patch_text_retrieval_demo(
        image_dir=args.image_dir,
        prompts=args.prompts,
        top_k=args.top_k,
        model_name=args.model_name,
        save_visualization=args.save_visualization,
        output_dir=args.output_dir,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        save_html_report=args.save_html_report,
        html_report_path=args.html_report_path,
    )
