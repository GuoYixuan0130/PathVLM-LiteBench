import torch
import torch.nn.functional as F

from pathvlm_litebench.evaluation import zero_shot_predict, compute_accuracy


def test_zero_shot_predict_basic():
    class_names = ["tumor", "normal"]

    class_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    image_embeddings = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )

    class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)

    results = zero_shot_predict(
        image_embeddings=image_embeddings,
        class_embeddings=class_embeddings,
        class_names=class_names,
        top_k=1,
    )

    predicted_labels = [item["predicted_label"] for item in results]

    assert predicted_labels == ["tumor", "normal"]


def test_compute_accuracy():
    predicted = ["tumor", "normal", "tumor"]
    true = ["tumor", "normal", "normal"]

    accuracy = compute_accuracy(predicted, true)

    assert accuracy == 2 / 3
