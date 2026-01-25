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
    # Approximate characters per token for chunking (4 chars/token average)
    CHARS_PER_TOKEN = 4

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

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks based on approximate token count.

        Uses character-based approximation since we don't have a tokenizer.

        Args:
            text: Input text to chunk.

        Returns:
            List of text chunks.
        """
        if not self.config.enable_chunking:
            return [text]

        # Convert token limits to character limits
        chunk_chars = self.config.chunk_tokens * self.CHARS_PER_TOKEN
        overlap_chars = self.config.chunk_overlap * self.CHARS_PER_TOKEN

        if len(text) <= chunk_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))

            # Try to break at word boundary
            if end < len(text):
                # Look for last space within the chunk
                space_idx = text.rfind(" ", start, end)
                if space_idx > start:
                    end = space_idx

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - overlap_chars
            if start >= len(text):
                break
            # Avoid infinite loop
            if end == len(text):
                break

        return chunks if chunks else [text]

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
        """Compute NCS metrics with chunking and variance-aware scoring.

        Uses max-then-aggregate strategy:
        1. For each context passage, chunk it if too long
        2. Compute similarity for each chunk against answer
        3. Take max similarity within each passage
        4. Aggregate across passages (max, mean, weighted_mean)
        5. Include variance as uncertainty signal

        Args:
            context_texts: List of context passages.
            answer: The answer text to compare against context.

        Returns:
            Dict with max, mean, weighted_mean similarity scores.
            If include_variance=True, also includes std deviation.
            Empty context returns neutral 0.5 scores.
        """
        neutral_result = {"max": 0.5, "mean": 0.5, "weighted_mean": 0.5}
        if self.config.include_variance:
            neutral_result["variance"] = 0.0

        if not context_texts:
            return neutral_result

        try:
            # Chunk each context passage
            all_chunks = []
            passage_chunk_counts = []
            passage_lengths = []

            for ctx in context_texts:
                chunks = self._chunk_text(ctx)
                passage_chunk_counts.append(len(chunks))
                passage_lengths.append(len(ctx))
                all_chunks.extend(chunks)

            # Encode all chunks and answer together
            all_embeddings = self.encode(all_chunks + [answer])
            chunk_embs = all_embeddings[:-1]
            answer_emb = all_embeddings[-1]

            # Compute similarities for all chunks
            chunk_similarities = [float(np.dot(chunk, answer_emb)) for chunk in chunk_embs]

            if not chunk_similarities or all(s == 0 for s in chunk_similarities):
                return neutral_result

            # Max-within-passage aggregation
            passage_similarities = []
            idx = 0
            for chunk_count in passage_chunk_counts:
                passage_chunk_sims = chunk_similarities[idx:idx + chunk_count]
                # Max within passage
                passage_similarities.append(max(passage_chunk_sims))
                idx += chunk_count

            if not passage_similarities:
                return neutral_result

            # Use passage lengths as weights
            weights = passage_lengths
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1] * len(passage_similarities)

            result = {
                "max": float(max(passage_similarities)),
                "mean": float(np.mean(passage_similarities)),
                "weighted_mean": float(np.average(passage_similarities, weights=weights)),
            }

            # Add variance as uncertainty signal
            if self.config.include_variance and len(passage_similarities) > 1:
                result["variance"] = float(np.std(passage_similarities))
            elif self.config.include_variance:
                result["variance"] = 0.0

            return result

        except Exception as e:
            logger.warning(f"NCS computation failed: {e}, returning neutral scores")
            return neutral_result

    def preload(self) -> None:
        """Preload model to avoid cold-start latency."""
        self._load_model()
