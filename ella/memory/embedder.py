"""Multilingual sentence embeddings using sentence-transformers.

Model: paraphrase-multilingual-MiniLM-L12-v2
- Supports 50+ languages including Chinese (zh) and English (en)
- Dimension: 384
- Fully in-process, no server needed
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ella.config import get_settings

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        settings = get_settings()
        _model = SentenceTransformer(settings.embed_model)
    return _model


def embed(text: str) -> list[float]:
    model = get_embedder()
    vector: np.ndarray = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors: np.ndarray = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()
