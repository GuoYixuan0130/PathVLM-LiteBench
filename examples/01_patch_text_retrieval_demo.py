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
    load_patch_images_from_paths,
    load_patch_manifest,
    records_to_image_paths,
    records_to_labels,
    filter_records_by_split,
    save_embeddings,
    load_embeddings,
    save_metadata,
    load_metadata,
)
from pathvlm_litebench.config import load_benchmark_config
from pathvlm_litebench.evaluation import (
    compute_text_to_image_recall_at_k,
    compute_mean_recall,
)
from pathvlm_litebench.models import create_model
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


def build_text_to_image_positive_pairs(
    labels: list[str | None],
    label_prompts: list[str],
) -> dict[int, set[int]]:
    """
    Build text-to-image positive index mapping from image labels.
    """
    if len(labels) == 0:
        raise ValueError("labels must not be empty.")

    if len(label_prompts) == 0:
        raise ValueError("label_prompts must not be empty.")

    if any(label is None for label in labels):
        raise ValueError("labels contain None values. labels must be complete.")

    positive_pairs: dict[int, set[int]] = {}

    for prompt_idx, label_name in enumerate(label_prompts):
        positives = {
            image_idx
            for image_idx, image_label in enumerate(labels)
            if image_label == label_name
        }

        if len(positives) == 0:
            raise ValueError(
                f"No positive images found for label prompt '{label_name}'."
            )

        positive_pairs[prompt_idx] = positives

    return positive_pairs


def enrich_retrieval_results_with_labels(
    retrieval_results: list[list[dict]],
    labels: list[str | None] | None = None,
    label_prompts: list[str] | None = None,
) -> list[list[dict]]:
    """
    Enrich retrieval results with label and prompt-target match metadata.
    """
    if labels is None:
        return retrieval_results

    if label_prompts is not None and len(label_prompts) != len(retrieval_results):
        raise ValueError(
            "label_prompts and retrieval_results must have the same length when "
            "enriching retrieval results."
        )

    enriched_results: list[list[dict]] = []

    for prompt_idx, prompt_results in enumerate(retrieval_results):
        enriched_prompt_results: list[dict] = []

        for item in prompt_results:
            enriched_item = dict(item)
            index = enriched_item.get("index")
            if not isinstance(index, int):
                raise ValueError("Each retrieval result item must contain an integer 'index'.")

            if index < 0 or index >= len(labels):
                raise ValueError(
                    f"Retrieved index {index} is out of range for labels of length {len(labels)}."
                )

            label = labels[index]
            enriched_item["label"] = label

            if label_prompts is not None:
                target_label = label_prompts[prompt_idx]
                enriched_item["target_label"] = target_label
                enriched_item["is_positive"] = label is not None and label == target_label

            enriched_prompt_results.append(enriched_item)

        enriched_results.append(enriched_prompt_results)

    return enriched_results


def run_patch_text_retrieval_demo(
    image_dir: str | Path | None = None,
    prompts: list[str] | None = None,
    manifest: str | Path | None = None,
    image_root: str | Path | None = None,
    split: str | None = None,
    label_prompts: list[str] | None = None,
    recall_k: list[int] | None = None,
    max_images: int | None = None,
    top_k: int = 3,
    model: str = "clip",
    device: str = "auto",
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
    labels: list[str | None] | None = None
    records = None

    if manifest is not None:
        if image_dir is not None:
            print("[INFO] --manifest is provided. --image_dir will be ignored.")

        records = load_patch_manifest(manifest, image_root=image_root)
        if split is not None:
            records = filter_records_by_split(records, split)
            print(f"[INFO] Applied split filter: {split}")

        if len(records) == 0:
            raise ValueError("No records available after manifest loading/filtering.")

        if max_images is not None:
            records = records[:max_images]

        image_paths = records_to_image_paths(records)
        labels = records_to_labels(records)
        images, image_paths = load_patch_images_from_paths(image_paths)

        print(f"[INFO] Loaded patch records from manifest: {manifest}")
        print(f"[INFO] Number of records: {len(records)}")
        if labels and all(label is not None for label in labels):
            unique_labels = sorted({str(label) for label in labels})
            print(f"[INFO] Unique labels in manifest records: {unique_labels}")
    elif image_dir is not None:
        image_dir = Path(image_dir)
        print("[INFO] Loading patch images...")
        images, image_paths = load_patch_images(image_dir, max_images=max_images)
        print(f"[INFO] Loaded {len(images)} images from {image_dir}")
    else:
        image_dir = create_demo_images(Path("examples") / "demo_patches")
        print(f"[INFO] No image_dir provided. Created demo images at: {image_dir}")
        print("[INFO] Loading patch images...")
        images, image_paths = load_patch_images(image_dir, max_images=max_images)
        print(f"[INFO] Loaded {len(images)} images from {image_dir}")

    if prompts is None or len(prompts) == 0:
        prompts = [
            "a red image",
            "a blue image",
            "a white image",
        ]

    print(f"[INFO] Loading model: {model}")
    print(f"[INFO] Requested device: {device}")
    vlm = create_model(model, device=device)
    if hasattr(vlm, "device"):
        print(f"[INFO] Using device: {vlm.device}")

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
                image_embeddings = vlm.encode_images(images)
                save_embeddings(image_embeddings, image_embedding_cache_path)
                save_metadata(image_paths, image_paths_cache_path)
                print("[INFO] Updated image embedding cache.")
        else:
            print("[INFO] Cache miss: encoding images and saving cache.")
            image_embeddings = vlm.encode_images(images)
            save_embeddings(image_embeddings, image_embedding_cache_path)
            save_metadata(image_paths, image_paths_cache_path)
            print("[INFO] Saved image embedding cache.")
    else:
        print("[INFO] Embedding cache disabled.")
        print("[INFO] Encoding images...")
        image_embeddings = vlm.encode_images(images)

    print("[INFO] Encoding text prompts...")
    text_embeddings = vlm.encode_text(prompts)

    print("[INFO] Retrieving top-k images...")
    results = retrieve_topk_images(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_paths=image_paths,
        top_k=top_k,
    )
    results = enrich_retrieval_results_with_labels(
        retrieval_results=results,
        labels=labels,
        label_prompts=label_prompts,
    )

    print("\n========== Retrieval Results ==========")

    for prompt, prompt_results in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        for rank, item in enumerate(prompt_results, start=1):
            summary = (
                f"  Top {rank}: "
                f"index={item['index']}, "
                f"score={item['score']:.4f}, "
            )

            if all(field in item for field in ("label", "target_label", "is_positive")):
                summary += (
                    f"label={item['label']}, "
                    f"target={item['target_label']}, "
                    f"match={item['is_positive']}, "
                )

            summary += f"path={item.get('path', 'N/A')}"
            print(summary)

    if manifest is not None:
        if labels is None or len(labels) == 0:
            print("\n[INFO] Manifest labels are unavailable. Skipping Recall@K.")
        elif any(label is None for label in labels):
            print("\n[INFO] Manifest labels are incomplete. Skipping Recall@K.")
        elif label_prompts is None:
            print(
                "\n[INFO] Manifest labels detected, but --label_prompts was not provided. "
                "Skipping Recall@K."
            )
        else:
            if len(label_prompts) != len(prompts):
                raise ValueError(
                    "label_prompts and prompts must have the same length when "
                    "computing Recall@K."
                )

            recall_k_values = recall_k if recall_k is not None else [1, 5, 10]
            positive_pairs = build_text_to_image_positive_pairs(
                labels=labels,
                label_prompts=label_prompts,
            )
            recall_metrics = compute_text_to_image_recall_at_k(
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
                positive_pairs=positive_pairs,
                k_values=recall_k_values,
                normalize=True,
            )
            mean_recall = compute_mean_recall(recall_metrics)

            print("\n========== Retrieval Metrics ==========")
            print("Text-to-image Recall@K:")
            for metric_name, metric_value in recall_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
            print(f"Mean Recall: {mean_recall:.4f}")

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


def merge_args_with_config(args: argparse.Namespace) -> dict:
    """
    Merge command-line arguments with optional benchmark config values.
    """
    default_values = {
        "image_dir": None,
        "prompts": None,
        "top_k": 3,
        "model": "clip",
        "device": "auto",
        "save_visualization": False,
        "output_dir": "outputs/retrieval_demo",
        "use_cache": False,
        "cache_dir": "outputs/cache",
        "save_html_report": False,
        "html_report_path": "outputs/retrieval_demo/retrieval_report.html",
        "recall_k": [1, 5, 10],
    }

    if args.config is None:
        return {
            "image_dir": args.image_dir,
            "prompts": args.prompts,
            "manifest": args.manifest,
            "image_root": args.image_root,
            "split": args.split,
            "label_prompts": args.label_prompts,
            "recall_k": args.recall_k if args.recall_k is not None else default_values["recall_k"],
            "max_images": args.max_images,
            "top_k": args.top_k if args.top_k is not None else default_values["top_k"],
            "model": args.model if args.model is not None else default_values["model"],
            "device": args.device if args.device is not None else default_values["device"],
            "save_visualization": args.save_visualization,
            "output_dir": args.output_dir if args.output_dir is not None else default_values["output_dir"],
            "use_cache": args.use_cache,
            "cache_dir": args.cache_dir if args.cache_dir is not None else default_values["cache_dir"],
            "save_html_report": args.save_html_report,
            "html_report_path": (
                args.html_report_path
                if args.html_report_path is not None
                else default_values["html_report_path"]
            ),
        }

    config = load_benchmark_config(args.config)
    if config.task != "retrieval":
        raise ValueError(
            f"Config task must be 'retrieval' for this demo, got '{config.task}'."
        )

    return {
        "image_dir": args.image_dir if args.image_dir is not None else config.image_dir,
        "prompts": args.prompts if args.prompts is not None else config.prompts,
        "manifest": args.manifest,
        "image_root": args.image_root,
        "split": args.split,
        "label_prompts": args.label_prompts,
        "recall_k": args.recall_k if args.recall_k is not None else default_values["recall_k"],
        "max_images": args.max_images,
        "top_k": args.top_k if args.top_k is not None else config.top_k,
        "model": args.model if args.model is not None else config.model,
        "device": args.device if args.device is not None else config.device,
        "save_visualization": args.save_visualization or config.save_visualization,
        "output_dir": args.output_dir if args.output_dir is not None else config.output_dir,
        "use_cache": args.use_cache or config.use_cache,
        "cache_dir": args.cache_dir if args.cache_dir is not None else config.cache_dir,
        "save_html_report": args.save_html_report or config.save_html_report,
        "html_report_path": (
            args.html_report_path
            if args.html_report_path is not None
            else config.html_report_path
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal patch-level image-text retrieval demo."
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
        "--manifest",
        type=str,
        default=None,
        help="Path to a CSV patch manifest. If provided, images and optional labels are loaded from the manifest.",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root directory used to resolve relative image paths in the manifest.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split name to filter manifest records, such as train, val, or test.",
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Text prompts for image-text retrieval.",
    )

    parser.add_argument(
        "--label_prompts",
        nargs="+",
        default=None,
        help="Labels corresponding to each text prompt. Used to compute text-to-image Recall@K when manifest labels are available.",
    )

    parser.add_argument(
        "--recall_k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="K values for Recall@K when evaluating retrieval with manifest labels.",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional maximum number of images to load.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Number of top images to retrieve for each prompt.",
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
        "--save_visualization",
        action="store_true",
        help="Save top-k retrieval visualization grids.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
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
        default=None,
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
        default=None,
        help="Output path for the HTML retrieval report.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_kwargs = merge_args_with_config(args)

    if args.config is not None:
        print(f"[INFO] Loaded benchmark config: {args.config}")

    print("[INFO] Final run configuration:")
    print(f"  model: {run_kwargs['model']}")
    print(f"  device: {run_kwargs['device']}")
    print(f"  manifest: {run_kwargs['manifest']}")
    print(f"  image_root: {run_kwargs['image_root']}")
    print(f"  split: {run_kwargs['split']}")
    print(f"  image_dir: {run_kwargs['image_dir']}")
    print(f"  max_images: {run_kwargs['max_images']}")
    print(f"  top_k: {run_kwargs['top_k']}")
    print(f"  num_prompts: {len(run_kwargs['prompts']) if run_kwargs['prompts'] else 0}")
    print(
        f"  num_label_prompts: "
        f"{len(run_kwargs['label_prompts']) if run_kwargs['label_prompts'] else 0}"
    )
    print(f"  recall_k: {run_kwargs['recall_k']}")
    print(f"  use_cache: {run_kwargs['use_cache']}")
    print(f"  save_visualization: {run_kwargs['save_visualization']}")
    print(f"  save_html_report: {run_kwargs['save_html_report']}")

    run_patch_text_retrieval_demo(**run_kwargs)
