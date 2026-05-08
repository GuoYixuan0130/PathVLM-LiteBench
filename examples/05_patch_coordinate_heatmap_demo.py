from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathvlm_litebench.data import load_coordinate_patch_manifest
from pathvlm_litebench.visualization import (
    aggregate_patch_scores_to_grid,
    save_patch_scores_csv,
    save_score_heatmap,
)


def create_synthetic_coordinate_demo(
    output_dir: str | Path = "outputs/patch_coordinate_heatmap_demo_synthetic",
) -> dict[str, str]:
    """
    Create a synthetic patch-coordinate heatmap demo.

    The generated images are simple colored tiles, not pathology images. This
    demo only verifies the artifact workflow: coordinate manifest -> scores ->
    heatmap.
    """
    output_dir = Path(output_dir)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "coordinate_manifest.csv"
    scores_path = output_dir / "scores.csv"
    heatmap_path = output_dir / "heatmap.png"

    rows = [
        ("patch_00.png", 0, 0, 0.10, (225, 245, 255)),
        ("patch_01.png", 224, 0, 0.35, (170, 215, 240)),
        ("patch_02.png", 448, 0, 0.65, (110, 170, 220)),
        ("patch_03.png", 0, 224, 0.20, (235, 230, 190)),
        ("patch_04.png", 224, 224, 0.55, (220, 185, 120)),
        ("patch_05.png", 448, 224, 0.90, (190, 115, 80)),
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["image_path", "x", "y", "width", "height", "label"],
        )
        writer.writeheader()

        for filename, x, y, score, color in rows:
            image_path = patches_dir / filename
            _save_synthetic_patch(image_path, color=color, text=f"{score:.2f}")
            writer.writerow(
                {
                    "image_path": f"patches/{filename}",
                    "x": x,
                    "y": y,
                    "width": 224,
                    "height": 224,
                    "label": "synthetic",
                }
            )

    records = load_coordinate_patch_manifest(
        manifest_path,
        image_root=output_dir,
        require_exists=True,
    )
    scores = [row[3] for row in rows]
    grid = aggregate_patch_scores_to_grid(records, scores)

    save_patch_scores_csv(
        records,
        scores,
        scores_path,
        prompt="synthetic high-score region",
    )
    save_score_heatmap(
        grid,
        heatmap_path,
        title="Synthetic patch-coordinate scores",
    )

    return {
        "manifest": str(manifest_path),
        "scores": str(scores_path),
        "heatmap": str(heatmap_path),
    }


def _save_synthetic_patch(
    image_path: Path,
    *,
    color: tuple[int, int, int],
    text: str,
) -> None:
    image = Image.new("RGB", (224, 224), color=color)
    draw = ImageDraw.Draw(image)
    draw.text((16, 16), text, fill=(0, 0, 0))
    image.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a synthetic patch-coordinate heatmap demo.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/patch_coordinate_heatmap_demo_synthetic",
        help="Output directory for synthetic patches, manifest, scores, and heatmap.",
    )
    args = parser.parse_args()

    saved_paths = create_synthetic_coordinate_demo(args.output_dir)
    print("[INFO] Synthetic patch-coordinate heatmap demo complete.")
    print(f"[INFO] Manifest: {saved_paths['manifest']}")
    print(f"[INFO] Scores: {saved_paths['scores']}")
    print(f"[INFO] Heatmap: {saved_paths['heatmap']}")


if __name__ == "__main__":
    main()
