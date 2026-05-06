from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image


class CONCHWrapper:
    """
    Optional wrapper for the CONCH pathology vision-language model.

    CONCH uses the official `mahmoodlab/CONCH` package rather than the
    Hugging Face `transformers` CLIP API, so this wrapper keeps imports lazy
    and reports missing optional dependencies or gated-model access clearly.
    """

    def __init__(
        self,
        model_name: str = "MahmoodLab/CONCH",
        device: str | None = None,
        model_cfg: str = "conch_ViT-B-16",
    ):
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = f"hf_hub:{model_name}"

        try:
            from conch.open_clip_custom import create_model_from_pretrained
            from conch.open_clip_custom.custom_tokenizer import get_tokenizer
        except ImportError as exc:
            raise ImportError(
                "CONCH support requires the optional CONCH package. "
                "Install it with: pip install git+https://github.com/Mahmoodlab/CONCH.git"
            ) from exc

        try:
            self.model, self.preprocess = create_model_from_pretrained(
                model_cfg,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load CONCH from Hugging Face. "
                "Ensure the CONCH package is installed, your Hugging Face account has "
                "access to MahmoodLab/CONCH, and you are logged in with `hf auth login`."
            ) from exc

        self.tokenizer = get_tokenizer()
        self.model.eval()

    def _tokenize_texts(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            max_length=127,
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = F.pad(
            encoded["input_ids"],
            (0, 1),
            value=self.tokenizer.pad_token_id,
        )
        return tokens.to(self.device)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encode text prompts into normalized CONCH text embeddings.
        """
        tokens = self._tokenize_texts(texts)
        text_features = self.model.encode_text(tokens)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features.cpu()

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Encode PIL images into normalized CONCH image embeddings.
        """
        image_inputs = torch.stack(
            [self.preprocess(image.convert("RGB")) for image in images],
            dim=0,
        ).to(self.device)
        image_features = self.model.encode_image(image_inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features.cpu()

    def compute_similarity(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between image embeddings and text embeddings.
        """
        return image_embeddings @ text_embeddings.T
