from __future__ import annotations

from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def iter_image_batches(
    items: Sequence[T],
    batch_size: int,
    show_progress: bool = True,
    desc: str = "Encoding images",
) -> Iterator[Sequence[T]]:
    """
    Yield consecutive batches from ``items``.

    A tqdm progress bar is shown only when there is more than one batch, so
    small smoke-test runs stay quiet while long patch-set encodings get feedback.
    """
    num_items = len(items)
    num_batches = (num_items + batch_size - 1) // batch_size
    starts = range(0, num_items, batch_size)

    if show_progress and num_batches > 1:
        from tqdm import tqdm

        starts = tqdm(starts, total=num_batches, desc=desc, unit="batch")

    for start in starts:
        yield items[start : start + batch_size]
