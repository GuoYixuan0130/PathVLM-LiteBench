from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _safe_filename(text: str, max_length: int = 40) -> str:
    """
    Convert a prompt into a safe filename component.
    """
    safe = "".join(c.lower() if c.isalnum() else "_" for c in text)
    safe = "_".join(part for part in safe.split("_") if part)
    if not safe:
        safe = "prompt"
    return safe[:max_length]


def _truncate_text(text: str, max_length: int = 60) -> str:
    """
    Truncate long text for visualization.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def save_topk_image_grids(
    prompts: list[str],
    retrieval_results: list[list[dict[str, Any]]],
    output_dir: str | Path,
    image_size: tuple[int, int] = (160, 160),
    padding: int = 16,
) -> list[str]:
    """
    Save top-k retrieval results as image grids.

    Args:
        prompts:
            A list of text prompts.
        retrieval_results:
            A nested list returned by retrieve_topk_images.
            Each result dictionary must contain a "path" field.
        output_dir:
            Directory where visualization images will be saved.
        image_size:
            Size of each displayed image.
        padding:
            Padding between images and text blocks.

    Returns:
        A list of saved visualization file paths.

    Raises:
        ValueError:
            If prompts and retrieval_results have different lengths.
            If any retrieval result does not contain an image path.
        FileNotFoundError:
            If an image path does not exist.
    """
    if len(prompts) != len(retrieval_results):
        raise ValueError(
            f"prompts and retrieval_results must have the same length: "
            f"{len(prompts)} vs {len(retrieval_results)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for prompt_idx, (prompt, prompt_results) in enumerate(zip(prompts, retrieval_results)):
        if len(prompt_results) == 0:
            continue

        num_items = len(prompt_results)
        thumb_w, thumb_h = image_size

        title_height = 36
        caption_height = 46

        canvas_width = padding + num_items * (thumb_w + padding)
        canvas_height = padding + title_height + thumb_h + caption_height + padding

        canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
        draw = ImageDraw.Draw(canvas)

        title = f"Prompt: {_truncate_text(prompt)}"
        draw.text((padding, padding), title, fill="black", font=font)

        y_image = padding + title_height
        y_caption = y_image + thumb_h + 4

        for rank, item in enumerate(prompt_results, start=1):
            if "path" not in item:
                raise ValueError(
                    "Each retrieval result must contain a 'path' field for visualization."
                )

            image_path = Path(item["path"])

            if not image_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {image_path}")

            image = Image.open(image_path).convert("RGB")
            image = image.resize(image_size)

            x = padding + (rank - 1) * (thumb_w + padding)
            canvas.paste(image, (x, y_image))

            score = item.get("score", 0.0)
            filename = image_path.name

            caption_line_1 = f"Top {rank} | score={score:.4f}"
            caption_line_2 = _truncate_text(filename, max_length=24)

            draw.text((x, y_caption), caption_line_1, fill="black", font=font)
            draw.text((x, y_caption + 16), caption_line_2, fill="black", font=font)

        safe_prompt = _safe_filename(prompt)
        save_path = output_dir / f"topk_prompt_{prompt_idx}_{safe_prompt}.png"
        canvas.save(save_path)

        saved_paths.append(str(save_path))

    return saved_paths
