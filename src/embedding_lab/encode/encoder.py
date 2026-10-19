"""Text to vectors with a transformer encoder, with the pooling step written out instead of hidden in a library."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MINILM = "sentence-transformers/all-MiniLM-L6-v2"  # 22M parameters, 384 dimensions, trained with mean pooling
MPNET = "sentence-transformers/all-mpnet-base-v2"  # 110M parameters, 768 dimensions, mean pooling


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass(frozen=True)
class Encoder:
    """One model plus the choices that turn its token vectors into one vector per text.

    `pooling="mean"` averages the token vectors (padding excluded); `"cls"` takes the first token's vector.
    `normalize=True` scales each vector to unit length, which makes dot product equal cosine similarity.
    """

    model_name: str = MINILM
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True
    max_length: int = 256
    device: str = default_device()

    def __post_init__(self):
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {self.pooling!r}")

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """An array of shape (len(texts), dimensions), one row per text in order."""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = tokenizer(texts[start:start + batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = model(**tokens).last_hidden_state  # (batch, tokens, dimensions)
                batches.append(self._pool(hidden, tokens["attention_mask"]).cpu().numpy())
        return np.concatenate(batches) if batches else np.zeros((0, model.config.hidden_size), dtype=np.float32)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)  # 1 for real tokens, 0 for padding
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.nn.functional.normalize(pooled, dim=1) if self.normalize else pooled
