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
    get_unique_labels,
    filter_records_by_split,
)
from pathvlm_litebench.evaluation import (
    zero_shot_predict,
    compute_classification_report,
)
from pathvlm_litebench.models import create_model
from pathvlm_litebench.prompts import build_class_prompts as build_pathology_class_prompts


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


def run_zero_shot_classification_demo(
    image_dir: str | Path | None = None,
    class_names: list[str] | None = None,
    class_prompts: list[str] | None = None,
    manifest: str | Path | None = None,
    image_root: str | Path | None = None,
    split: str | None = None,
    max_images: int | None = None,
    top_k: int = 3,
    model: str = "clip",
    device: str = "auto",
) -> None:
    """
    Run a minimal patch-level zero-shot classification demo.
    """
    default_smoke_test_class_names = ["red", "blue", "white", "black", "green"]
    records = None
    true_labels: list[str | None] | None = None

    if manifest is not None:
        if image_dir is not None:
            print("[INFO] --manifest is provided. --image_dir will be ignored.")

        records = load_patch_manifest(manifest, image_root=image_root)
        if split is not None:
            records = filter_records_by_split(records, split)
            print(f"[INFO] Applied split filter: {split}")

        if len(records) == 0:
            raise ValueError(
                "No records found in manifest after applying filters."
            )

        if max_images is not None:
            records = records[:max_images]

        image_paths = records_to_image_paths(records)
        true_labels = records_to_labels(records)
        images, image_paths = load_patch_images_from_paths(image_paths)

        print(f"[INFO] Loaded patch records from manifest: {manifest}")
        print(f"[INFO] Number of records: {len(records)}")
    else:
        if image_dir is None:
            image_dir = create_demo_images(Path("examples") / "demo_patches")
            print(f"[INFO] No image_dir provided. Created demo images at: {image_dir}")
        else:
            image_dir = Path(image_dir)

        print("[INFO] Loading patch images...")
        images, image_paths = load_patch_images(image_dir, max_images=max_images)
        print(f"[INFO] Loaded {len(images)} images from {image_dir}")

    if class_names is None or len(class_names) == 0:
        if records is not None and len(get_unique_labels(records)) > 0:
            class_names = get_unique_labels(records)
            print(f"[INFO] Inferred class names from manifest labels: {class_names}")
        else:
            class_names = default_smoke_test_class_names

    if class_prompts is None and manifest is None and class_names == default_smoke_test_class_names:
        class_prompts = [
            "a red image",
            "a blue image",
            "a white image",
            "a black image",
            "a green image",
        ]

    if class_prompts is not None:
        if len(class_prompts) != len(class_names):
            raise ValueError(
                f"class_prompts and class_names must have the same length: "
                f"{len(class_prompts)} vs {len(class_names)}"
            )
    else:
        class_prompts = build_pathology_class_prompts(class_names)

    print("[INFO] Class names:")
    for name, prompt in zip(class_names, class_prompts):
        print(f"  - {name}: {prompt}")

    print(f"[INFO] Loading model: {model}")
    print(f"[INFO] Requested device: {device}")
    vlm = create_model(model, device=device)
    if hasattr(vlm, "device"):
        print(f"[INFO] Using device: {vlm.device}")

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
    predicted_labels = [item["predicted_label"] for item in results]

    print("\n========== Zero-Shot Classification Results ==========")

    for idx, (image_path, result) in enumerate(zip(image_paths, results)):
        print(f"\nImage: {image_path}")
        if true_labels is not None:
            true_label = true_labels[idx]
            print(f"True label: {true_label if true_label is not None else 'N/A'}")
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

    if true_labels is not None:
        if all(label is not None for label in true_labels):
            report = compute_classification_report(
                true_labels=[str(label) for label in true_labels],
                predicted_labels=predicted_labels,
                class_names=class_names,
            )

            print("\n========== Classification Metrics ==========")
            print(f"Accuracy: {report['accuracy']:.4f}")
            print(f"Balanced Accuracy: {report['balanced_accuracy']:.4f}")
            print(f"Macro Precision: {report['macro_precision']:.4f}")
            print(f"Macro Recall: {report['macro_recall']:.4f}")
            print(f"Macro F1: {report['macro_f1']:.4f}")

            print("\nPer-class metrics:")
            for class_name, class_metrics in report["per_class"].items():
                print(f"  {class_name}:")
                print(f"    precision={class_metrics['precision']:.4f}")
                print(f"    recall={class_metrics['recall']:.4f}")
                print(f"    f1={class_metrics['f1']:.4f}")
                print(f"    support={class_metrics['support']}")

            confusion = report["confusion_matrix"]
            confusion_class_names = confusion["class_names"]
            confusion_matrix = confusion["matrix"]
            print("\nConfusion matrix:")
            print(f"  classes: {', '.join(confusion_class_names)}")
            print("  matrix:")
            for row_class_name, row in zip(confusion_class_names, confusion_matrix):
                row_values = ", ".join(str(value) for value in row)
                print(f"    {row_class_name}: {row_values}")
        else:
            print("\n[INFO] Manifest labels are incomplete. Skipping classification metrics.")

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
        "--manifest",
        type=str,
        default=None,
        help="Path to a CSV patch manifest. If provided, images and labels are loaded from the manifest.",
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
        "--max_images",
        type=int,
        default=None,
        help="Optional maximum number of images to load.",
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

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for model inference. Use 'auto' to select CUDA if available, otherwise CPU.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_zero_shot_classification_demo(
        image_dir=args.image_dir,
        class_names=args.class_names,
        class_prompts=args.class_prompts,
        manifest=args.manifest,
        image_root=args.image_root,
        split=args.split,
        max_images=args.max_images,
        top_k=args.top_k,
        model=args.model,
        device=args.device,
    )
