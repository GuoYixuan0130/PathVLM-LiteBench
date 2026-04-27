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


def load_patch_images_from_paths(
    image_paths: list[str | Path],
    max_images: int | None = None,
) -> tuple[list[Image.Image], list[str]]:
    """
    Load patch images from an explicit list of image paths.

    Args:
        image_paths:
            A list of image paths.
        max_images:
            Optional maximum number of images to load.

    Returns:
        images:
            A list of PIL RGB images.
        loaded_paths:
            A list of loaded image file paths as strings.

    Raises:
        FileNotFoundError:
            If any image path does not exist.
        ValueError:
            If any file extension is unsupported.
    """
    path_list = [Path(path) for path in image_paths]

    if max_images is not None:
        path_list = path_list[:max_images]

    images: list[Image.Image] = []
    loaded_paths: list[str] = []

    for path in path_list:
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension for file: {path}. "
                f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}"
            )

        with Image.open(path) as image:
            images.append(image.convert("RGB"))

        loaded_paths.append(str(path))

    return images, loaded_paths
