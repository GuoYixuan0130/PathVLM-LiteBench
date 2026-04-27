import torch
import torch.nn.functional as F

from pathvlm_litebench.evaluation import (
    compute_text_to_image_recall_at_k,
    compute_image_to_text_recall_at_k,
    compute_mean_recall,
)


def test_retrieval_recall_metrics_basic():
    image_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )

    text_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    text_to_image_positive_pairs = {
        0: {0, 1},
        1: {2, 3},
    }

    image_to_text_positive_pairs = {
        0: {0},
        1: {0},
        2: {1},
        3: {1},
    }

    t2i_metrics = compute_text_to_image_recall_at_k(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        positive_pairs=text_to_image_positive_pairs,
        k_values=(1, 2),
    )

    i2t_metrics = compute_image_to_text_recall_at_k(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        positive_pairs=image_to_text_positive_pairs,
        k_values=(1, 2),
    )

    assert t2i_metrics["R@1"] == 1.0
    assert t2i_metrics["R@2"] == 1.0
    assert i2t_metrics["R@1"] == 1.0
    assert i2t_metrics["R@2"] == 1.0
    assert compute_mean_recall(t2i_metrics) == 1.0
