"""Embeddings API adapter, shared by indexing and query retrieval."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Create the embeddings client once and reuse its connection pool."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=30, max_retries=2)
    return _client


def is_valid(vector: np.ndarray) -> bool:
    """A usable embedding has the configured shape, finite values and nonzero norm."""
    return (
        vector.shape == (config.EMBEDDING_DIMENSIONS,)
        and bool(np.isfinite(vector).all())
        and bool(np.linalg.norm(vector))
    )


def embed(text: str) -> np.ndarray:
    """Embed one text as a little-endian float32 vector."""
    if not text.strip():
        raise ValueError("Cannot embed empty text")
    result = _get_client().embeddings.create(
        model=config.EMBEDDING_MODEL, dimensions=config.EMBEDDING_DIMENSIONS, input=text
    )
    vector = np.asarray(result.data[0].embedding, dtype="<f4")
    if not is_valid(vector):
        raise ValueError("Embedding API returned an invalid vector")
    return vector
