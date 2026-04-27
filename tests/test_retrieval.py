import torch
import torch.nn.functional as F

from pathvlm_litebench.retrieval import retrieve_topk_images


def test_retrieve_topk_images_basic():
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ]
    )
    text_embeddings = torch.tensor(
        [
            [1.0, 0.0],
        ]
    )

    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    image_paths = ["img0.png", "img1.png", "img2.png"]

    results = retrieve_topk_images(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        image_paths=image_paths,
        top_k=2,
    )

    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0]["index"] == 0
    assert results[0][0]["path"] == "img0.png"


def test_retrieve_topk_images_dimension_mismatch():
    image_embeddings = torch.randn(3, 4)
    text_embeddings = torch.randn(2, 5)

    try:
        retrieve_topk_images(image_embeddings, text_embeddings)
    except ValueError as exc:
        assert "Embedding dimensions do not match" in str(exc)
    else:
        raise AssertionError("Expected ValueError for dimension mismatch.")
