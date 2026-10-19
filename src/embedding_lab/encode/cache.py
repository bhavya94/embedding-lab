"""Embeddings on disk, so a corpus is encoded once per encoder configuration."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from embedding_lab.encode.encoder import Encoder


@dataclass(frozen=True)
class EmbeddingCache:
    directory: Path = Path(".cache") / "embeddings"

    def encode(self, encoder: Encoder, name: str, texts: list[str]) -> np.ndarray:
        """`encoder.encode(texts)`, read from disk when this encoder has seen exactly these texts before.

        The key hashes the encoder's settings (except the device) and the texts, so a changed model, pooling,
        length or corpus gets a new file, and so do retrained local weights. `name` only makes the file name
        readable.
        """
        settings = {key: str(value) for key, value in asdict(encoder).items() if key != "device"}
        weights = Path(encoder.model_name) / "model.safetensors"
        if weights.exists():  # a local model, such as a fine-tuned one, can be retrained in place
            settings["weights_modified"] = str(weights.stat().st_mtime_ns)
        digest = hashlib.sha256(json.dumps([settings, texts]).encode()).hexdigest()[:16]
        path = self.directory / f"{name}-{digest}.npy"
        if path.exists():
            return np.load(path)
        embeddings = encoder.encode(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
        return embeddings
