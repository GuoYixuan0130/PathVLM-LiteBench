import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


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
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
        text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features.cpu()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Encode a list of PIL images into normalized image embeddings.
        """
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        image_features = self.model.get_image_features(**inputs)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
        image_features = F.normalize(image_features, p=2, dim=-1)

        return image_features.cpu()

    def compute_similarity(self, image_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between image embeddings and text embeddings.
        """
        return image_embeddings @ text_embeddings.T
