from __future__ import annotations

from typing import Iterable

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "auto"):
        from sentence_transformers import SentenceTransformer

        model_device = None if device == "auto" else device
        self.model = SentenceTransformer(model_name, device=model_device)

    def encode(self, texts: str | Iterable[str]) -> np.ndarray:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        embeddings = self.model.encode(values, convert_to_numpy=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings[0] if single else embeddings
