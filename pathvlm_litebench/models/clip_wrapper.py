import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from ._batching import iter_image_batches


class CLIPWrapper:
    """
    A lightweight wrapper for CLIP-style vision-language models.

    This wrapper supports:
    - text embedding extraction
    - image embedding extraction
    - automatic CPU/GPU selection
    - L2 normalization for retrieval
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a list of text prompts into normalized text embeddings.
        """
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu()

    @torch.no_grad()
    def encode_images(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of PIL images into normalized image embeddings.

        Images are encoded in batches so peak memory stays bounded, which keeps
        large patch sets feasible on CPU-only machines and laptop GPUs.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        batch_features: list[torch.Tensor] = []
        for batch in iter_image_batches(images, batch_size, show_progress):
            inputs = self.processor(
                images=batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            batch_features.append(image_features.cpu())

        return torch.cat(batch_features, dim=0)

    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between image embeddings and text embeddings.
        """
        return image_embeddings @ text_embeddings.T
