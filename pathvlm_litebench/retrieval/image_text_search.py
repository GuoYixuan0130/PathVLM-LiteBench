import torch


def retrieve_topk_images(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    image_paths: list[str] | None = None,
    top_k: int = 5,
) -> list[list[dict]]:
    """
    Retrieve top-k most similar images for each text embedding.

    Args:
        image_embeddings:
            Tensor of shape [num_images, embedding_dim].
        text_embeddings:
            Tensor of shape [num_texts, embedding_dim].
        image_paths:
            Optional list of image file paths. Length should match num_images.
        top_k:
            Number of top images to return for each text prompt.

    Returns:
        A nested list. Each inner list contains top-k retrieval results
        for one text prompt.
    """
    if image_embeddings.ndim != 2:
        raise ValueError("image_embeddings must have shape [num_images, embedding_dim].")

    if text_embeddings.ndim != 2:
        raise ValueError("text_embeddings must have shape [num_texts, embedding_dim].")

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions do not match: "
            f"image dim = {image_embeddings.shape[1]}, "
            f"text dim = {text_embeddings.shape[1]}"
        )

    num_images = image_embeddings.shape[0]

    if image_paths is not None and len(image_paths) != num_images:
        raise ValueError(
            f"Length of image_paths must match number of images: "
            f"{len(image_paths)} vs {num_images}"
        )

    top_k = min(top_k, num_images)

    similarity = image_embeddings @ text_embeddings.T

    all_results = []

    for text_idx in range(text_embeddings.shape[0]):
        scores = similarity[:, text_idx]
        top_scores, top_indices = torch.topk(scores, k=top_k)

        prompt_results = []
        for score, index in zip(top_scores, top_indices):
            idx = int(index.item())
            result = {
                "index": idx,
                "score": float(score.item()),
            }

            if image_paths is not None:
                result["path"] = image_paths[idx]

            prompt_results.append(result)

        all_results.append(prompt_results)

    return all_results
