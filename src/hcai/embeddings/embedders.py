"""Embedding backends.

- hashing   : offline, deterministic, zero downloads. Used in CI and the default eval.
- fastembed : real local semantic embeddings (ONNX, BAAI/bge-small, ~130MB, CPU-friendly
              enough for an 8GB laptop). Recommended for local demos.
- voyage    : Voyage AI API (Anthropic's recommended embeddings partner).
- openai    : OpenAI embeddings API (or Azure OpenAI with the same SDK).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from hcai.config import Settings


def _normalise(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Word + character n-gram hashing. No training, no network, stable across runs."""

    def __init__(self, dim: int = 768):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._word = HashingVectorizer(
            n_features=dim, ngram_range=(1, 2), alternate_sign=False, norm=None, stop_words="english"
        )
        self._char = HashingVectorizer(
            n_features=dim, analyzer="char_wb", ngram_range=(3, 5), alternate_sign=False, norm=None
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        w = self._word.transform(texts).toarray()
        c = self._char.transform(texts).toarray()
        return _normalise(_normalise(w) + 0.5 * _normalise(c))


class FastEmbedEmbedder(Embedder):
    def __init__(self, model: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed(texts))))


class VoyageEmbedder(Embedder):
    def __init__(self, model: str = "voyage-3.5-lite"):
        import httpx

        self._http = httpx.Client(timeout=60)
        self._model = model
        self._key = os.environ.get("VOYAGE_API_KEY", "")
        self.dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._http.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self._key}"},
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        return _normalise(np.array([d["embedding"] for d in resp.json()["data"]]))


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model
        self.dim = len(self.embed(["dimension probe"])[0])

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return _normalise(np.array([d.embedding for d in resp.data]))


def get_embedder(settings: Settings) -> Embedder:
    p = settings.embedding_provider.lower()
    if p == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    if p == "fastembed":
        return FastEmbedEmbedder(settings.embedding_model)
    if p == "voyage":
        return VoyageEmbedder(settings.embedding_model)
    if p == "openai":
        return OpenAIEmbedder(settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
