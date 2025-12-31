"""Model2Vec encoder for fast semantic similarity computation."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from lettucedetect.detectors.stage2.config import Model2VecConfig

logger = logging.getLogger(__name__)


class Model2VecEncoder:
    """Wrapper for Model2Vec static embeddings - 500x faster than sentence-transformers."""

    # Default dimension if model fails to load
    DEFAULT_DIM = 256

    def __init__(self, config: Model2VecConfig | None = None):
        self.config = config or Model2VecConfig()
        self._model = None  # Lazy loading
        self._dim: int | None = None  # Store model dimension after loading

    def _load_model(self) -> None:
        """Lazy load the Model2Vec model and store its dimension."""
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.config.model_name)
            # Store the actual model dimension
            self._dim = self._model.dim

    @property
    def dim(self) -> int:
        """Return the embedding dimension, loading model if needed."""
        if self._dim is None:
            self._load_model()
        return self._dim or self.DEFAULT_DIM

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode texts into embeddings with optional L2 normalization.

        Args:
            texts: List of texts to encode.

        Returns:
            Embeddings array of shape (len(texts), embedding_dim).
            On error, returns zero embeddings with warning.
        """
        # Use actual model dimension or default
        dim = self._dim if self._dim else self.DEFAULT_DIM

        if not texts:
            return np.zeros((0, dim), dtype=np.float32)

        try:
            self._load_model()
            embeddings = self._model.encode(texts)

            if self.config.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)  # Avoid division by zero
                embeddings = embeddings / norms

            return embeddings.astype(np.float32)

        except Exception as e:
            logger.warning(f"Model2Vec encoding failed: {e}, returning zero embeddings")
            # Use stored dimension if available, otherwise default
            fallback_dim = self._dim if self._dim else self.DEFAULT_DIM
            return np.zeros((len(texts), fallback_dim), dtype=np.float32)

    def compute_ncs(self, context_texts: list[str], answer: str) -> dict:
        """Compute NCS metrics: max, mean, weighted_mean similarity.

        Args:
            context_texts: List of context passages.
            answer: The answer text to compare against context.

        Returns:
            Dict with max, mean, weighted_mean similarity scores.
            Empty context returns neutral 0.5 scores.
        """
        if not context_texts:
            return {"max": 0.5, "mean": 0.5, "weighted_mean": 0.5}

        try:
            all_embeddings = self.encode(context_texts + [answer])
            context_embs = all_embeddings[:-1]
            answer_emb = all_embeddings[-1]

            similarities = [float(np.dot(ctx, answer_emb)) for ctx in context_embs]

            if not similarities or all(s == 0 for s in similarities):
                return {"max": 0.5, "mean": 0.5, "weighted_mean": 0.5}

            weights = [len(t) for t in context_texts]
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1] * len(context_texts)

            return {
                "max": float(max(similarities)),
                "mean": float(np.mean(similarities)),
                "weighted_mean": float(np.average(similarities, weights=weights)),
            }

        except Exception as e:
            logger.warning(f"NCS computation failed: {e}, returning neutral scores")
            return {"max": 0.5, "mean": 0.5, "weighted_mean": 0.5}

    def preload(self) -> None:
        """Preload model to avoid cold-start latency."""
        self._load_model()
