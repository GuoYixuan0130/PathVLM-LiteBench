from pathlib import Path
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_patch_images(
    image_dir: str | Path,
    max_images: int | None = None,
) -> tuple[list[Image.Image], list[str]]:
    """
    Load patch images from a folder.

    Args:
        image_dir:
            Path to the folder containing patch images.
        max_images:
            Optional maximum number of images to load.

    Returns:
        images:
            A list of PIL RGB images.
        image_paths:
            A list of image file paths as strings.

    Raises:
        FileNotFoundError:
            If image_dir does not exist.
        NotADirectoryError:
            If image_dir is not a directory.
        ValueError:
            If no supported images are found.
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {image_dir}")

    image_paths = sorted(
        [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )

    if max_images is not None:
        image_paths = image_paths[:max_images]

    if len(image_paths) == 0:
        raise ValueError(f"No supported image files found in: {image_dir}")

    images = []

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        images.append(image)

    return images, [str(path) for path in image_paths]
