from __future__ import annotations

import torch
import torch.nn.functional as F

from pathvlm_litebench.evaluation import (
    compute_image_to_text_recall_at_k,
    compute_mean_recall,
    compute_text_to_image_recall_at_k,
)


def run_retrieval_metrics_demo() -> None:
    """
    Run a minimal retrieval metrics demo using toy embeddings.

    This demo does not use real pathology images. It only verifies that
    Recall@K metrics work correctly on simple controlled embeddings.
    """
    print("[INFO] Running retrieval metrics demo with toy embeddings.")

    image_embeddings = torch.tensor(
        [
            [1.0, 0.0],   # image 0: tumor-like
            [0.9, 0.1],   # image 1: tumor-like
            [0.0, 1.0],   # image 2: normal-like
            [0.1, 0.9],   # image 3: normal-like
        ]
    )

    text_embeddings = torch.tensor(
        [
            [1.0, 0.0],   # text 0: tumor query
            [0.0, 1.0],   # text 1: normal query
        ]
    )

    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # For text-to-image retrieval:
    # text query 0 should retrieve image 0 or 1.
    # text query 1 should retrieve image 2 or 3.
    text_to_image_positive_pairs = {
        0: {0, 1},
        1: {2, 3},
    }

    # For image-to-text retrieval:
    # image 0 and 1 should retrieve text 0.
    # image 2 and 3 should retrieve text 1.
    image_to_text_positive_pairs = {
        0: {0},
        1: {0},
        2: {1},
        3: {1},
    }

    print("\n[INFO] Positive pair definition:")
    print("  Text-to-image positive_pairs maps text_index -> set of correct image_indices.")
    print("  Image-to-text positive_pairs maps image_index -> set of correct text_indices.")

    print("\n[INFO] Text-to-image positive pairs:")
    for query_idx, target_indices in text_to_image_positive_pairs.items():
        print(f"  text {query_idx} -> images {sorted(target_indices)}")

    print("\n[INFO] Image-to-text positive pairs:")
    for query_idx, target_indices in image_to_text_positive_pairs.items():
        print(f"  image {query_idx} -> texts {sorted(target_indices)}")

    k_values = (1, 2)

    print("\n[INFO] Computing text-to-image Recall@K...")
    t2i_metrics = compute_text_to_image_recall_at_k(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        positive_pairs=text_to_image_positive_pairs,
        k_values=k_values,
    )

    print("[INFO] Computing image-to-text Recall@K...")
    i2t_metrics = compute_image_to_text_recall_at_k(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        positive_pairs=image_to_text_positive_pairs,
        k_values=k_values,
    )

    t2i_mean_recall = compute_mean_recall(t2i_metrics)
    i2t_mean_recall = compute_mean_recall(i2t_metrics)

    print("\n========== Retrieval Metrics Results ==========")

    print("\nText-to-image retrieval:")
    for name, value in t2i_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"  Mean Recall: {t2i_mean_recall:.4f}")

    print("\nImage-to-text retrieval:")
    for name, value in i2t_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"  Mean Recall: {i2t_mean_recall:.4f}")

    print("\n[INFO] Retrieval metrics demo finished successfully.")


if __name__ == "__main__":
    run_retrieval_metrics_demo()
