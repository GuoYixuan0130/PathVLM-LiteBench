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
from pathvlm_litebench.config import load_benchmark_config
from pathvlm_litebench.evaluation import (
    zero_shot_predict,
    compute_classification_report,
)
from pathvlm_litebench.models import create_model
from pathvlm_litebench.prompts import build_class_prompts as build_pathology_class_prompts
from pathvlm_litebench.visualization import (
    save_zero_shot_predictions_csv,
    save_classification_metrics_json,
)
from pathvlm_litebench.visualization.zero_shot_report import (
    save_zero_shot_errors_csv,
    compute_zero_shot_error_summary,
)


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
    save_report: bool = False,
    report_dir: str | Path = "outputs/zero_shot_demo",
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

    classification_report: dict | None = None
    complete_true_labels: list[str] | None = None

    if true_labels is not None:
        if all(label is not None for label in true_labels):
            complete_true_labels = [str(label) for label in true_labels]
            classification_report = compute_classification_report(
                true_labels=complete_true_labels,
                predicted_labels=predicted_labels,
                class_names=class_names,
            )

            print("\n========== Classification Metrics ==========")
            print(f"Accuracy: {classification_report['accuracy']:.4f}")
            print(f"Balanced Accuracy: {classification_report['balanced_accuracy']:.4f}")
            print(f"Macro Precision: {classification_report['macro_precision']:.4f}")
            print(f"Macro Recall: {classification_report['macro_recall']:.4f}")
            print(f"Macro F1: {classification_report['macro_f1']:.4f}")

            print("\nPer-class metrics:")
            for class_name, class_metrics in classification_report["per_class"].items():
                print(f"  {class_name}:")
                print(f"    precision={class_metrics['precision']:.4f}")
                print(f"    recall={class_metrics['recall']:.4f}")
                print(f"    f1={class_metrics['f1']:.4f}")
                print(f"    support={class_metrics['support']}")

            confusion = classification_report["confusion_matrix"]
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

    if save_report:
        report_dir = Path(report_dir)
        predictions_path = report_dir / "predictions.csv"
        metrics_path = report_dir / "metrics.json"
        errors_path = report_dir / "errors.csv"

        saved_predictions_path = save_zero_shot_predictions_csv(
            image_paths=image_paths,
            results=results,
            output_csv_path=predictions_path,
            true_labels=true_labels,
        )
        print(f"\n[INFO] Saved zero-shot predictions: {saved_predictions_path}")

        metrics_payload: dict = {}
        if classification_report is not None:
            metrics_payload["classification_report"] = classification_report

        error_summary = compute_zero_shot_error_summary(
            results=results,
            true_labels=true_labels,
        )
        metrics_payload["error_summary"] = error_summary

        if true_labels is not None:
            saved_errors_path = save_zero_shot_errors_csv(
                image_paths=image_paths,
                results=results,
                output_csv_path=errors_path,
                true_labels=true_labels,
            )
            print(f"[INFO] Saved zero-shot errors: {saved_errors_path}")
        else:
            print(
                "[INFO] true_labels were not provided. Skipped errors.csv."
            )

        metadata = {
            "model": model,
            "device": device,
            "split": split,
            "manifest": str(manifest) if manifest is not None else None,
            "image_dir": str(image_dir) if image_dir is not None else None,
            "class_names": class_names,
            "class_prompts": class_prompts,
            "top_k": top_k,
            "num_images": len(image_paths),
        }
        saved_metrics_path = save_classification_metrics_json(
            metrics=metrics_payload,
            output_json_path=metrics_path,
            metadata=metadata,
        )
        print(f"[INFO] Saved zero-shot metrics: {saved_metrics_path}")

        warning = error_summary.get("warning")
        if isinstance(warning, str) and len(warning) > 0:
            print(f"[WARNING] {warning}")

    print("\n[INFO] Zero-shot classification demo finished successfully.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal patch-level zero-shot classification demo."
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
        default=None,
        help="Number of top class predictions to show for each image.",
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
        "--save_report",
        action="store_true",
        help="Save zero-shot predictions and metrics reports.",
    )

    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Directory for saving zero-shot report files.",
    )

    return parser.parse_args()


def merge_args_with_config(args: argparse.Namespace) -> dict:
    """
    Merge command-line arguments with optional benchmark config values.
    """
    default_values = {
        "image_dir": None,
        "manifest": None,
        "image_root": None,
        "split": None,
        "max_images": None,
        "class_names": None,
        "class_prompts": None,
        "top_k": 3,
        "model": "clip",
        "device": "auto",
        "save_report": False,
        "report_dir": "outputs/zero_shot_demo",
    }

    if args.config is None:
        return {
            "image_dir": args.image_dir,
            "class_names": args.class_names,
            "class_prompts": args.class_prompts,
            "manifest": args.manifest,
            "image_root": args.image_root,
            "split": args.split,
            "max_images": args.max_images,
            "top_k": args.top_k if args.top_k is not None else default_values["top_k"],
            "model": args.model if args.model is not None else default_values["model"],
            "device": args.device if args.device is not None else default_values["device"],
            "save_report": args.save_report,
            "report_dir": args.report_dir if args.report_dir is not None else default_values["report_dir"],
        }

    config = load_benchmark_config(args.config)
    if config.task != "zero_shot":
        raise ValueError(
            f"Config task must be 'zero_shot' for this demo, got '{config.task}'."
        )

    return {
        "image_dir": args.image_dir if args.image_dir is not None else config.image_dir,
        "class_names": args.class_names if args.class_names is not None else config.class_names,
        "class_prompts": args.class_prompts if args.class_prompts is not None else config.class_prompts,
        "manifest": args.manifest if args.manifest is not None else config.manifest,
        "image_root": args.image_root if args.image_root is not None else config.image_root,
        "split": args.split if args.split is not None else config.split,
        "max_images": args.max_images if args.max_images is not None else config.max_images,
        "top_k": args.top_k if args.top_k is not None else config.top_k,
        "model": args.model if args.model is not None else config.model,
        "device": args.device if args.device is not None else config.device,
        "save_report": args.save_report or config.save_report,
        "report_dir": args.report_dir if args.report_dir is not None else config.report_dir,
    }


if __name__ == "__main__":
    args = parse_args()
    run_kwargs = merge_args_with_config(args)

    if args.config is not None:
        print(f"[INFO] Loaded benchmark config: {args.config}")

    print("[INFO] Final run configuration:")
    print("  task: zero_shot")
    print(f"  model: {run_kwargs['model']}")
    print(f"  device: {run_kwargs['device']}")
    print(f"  manifest: {run_kwargs['manifest']}")
    print(f"  image_root: {run_kwargs['image_root']}")
    print(f"  image_dir: {run_kwargs['image_dir']}")
    print(f"  split: {run_kwargs['split']}")
    print(f"  top_k: {run_kwargs['top_k']}")
    print(f"  save_report: {run_kwargs['save_report']}")

    run_zero_shot_classification_demo(
        **run_kwargs,
    )
