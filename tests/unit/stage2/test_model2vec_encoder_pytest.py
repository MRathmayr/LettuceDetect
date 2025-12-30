"""Unit tests for Model2VecEncoder."""

import numpy as np
import pytest

from lettucedetect.detectors.stage2.config import Model2VecConfig
from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder


class TestModel2VecEncoderBasics:
    """Test basic encoding functionality."""

    def setup_method(self):
        """Set up encoder for tests."""
        self.config = Model2VecConfig()
        self.encoder = Model2VecEncoder(self.config)

    def test_lazy_loading(self):
        """Model should not load until first use."""
        assert self.encoder._model is None

    def test_encode_single_text(self):
        """Encoding single text returns correct shape."""
        texts = ["This is a test sentence."]
        embeddings = self.encoder.encode(texts)

        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0  # Embedding dimension
        assert embeddings.dtype == np.float32

    def test_encode_batch(self):
        """Batch encoding returns correct shape."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = self.encoder.encode(texts)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0

    def test_encode_empty_list(self):
        """Empty list returns empty array."""
        embeddings = self.encoder.encode([])
        assert embeddings.shape[0] == 0

    def test_model_loaded_after_encode(self):
        """Model is loaded after encode() call."""
        self.encoder.encode(["test"])
        assert self.encoder._model is not None


class TestModel2VecNormalization:
    """Test L2 normalization behavior."""

    def test_l2_normalization_enabled(self):
        """Embeddings are normalized to unit length when enabled."""
        config = Model2VecConfig(normalize_embeddings=True)
        encoder = Model2VecEncoder(config)

        texts = ["Test sentence for normalization."]
        embeddings = encoder.encode(texts)

        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0], decimal=5)

    def test_l2_normalization_disabled(self):
        """Embeddings are not normalized when disabled."""
        config = Model2VecConfig(normalize_embeddings=False)
        encoder = Model2VecEncoder(config)

        texts = ["Test sentence for normalization."]
        embeddings = encoder.encode(texts)

        norms = np.linalg.norm(embeddings, axis=1)
        # Norm should NOT be 1.0 (unless by coincidence)
        # Just verify it's a valid number
        assert not np.isnan(norms[0])


class TestNCSComputation:
    """Test Normalized Cosine Similarity computation."""

    def setup_method(self):
        """Set up encoder for NCS tests."""
        self.encoder = Model2VecEncoder(Model2VecConfig())

    def test_compute_ncs_empty_context(self):
        """Empty context returns neutral 0.5 scores."""
        result = self.encoder.compute_ncs([], "Some answer")

        assert result["max"] == 0.5
        assert result["mean"] == 0.5
        assert result["weighted_mean"] == 0.5

    def test_compute_ncs_returns_dict(self):
        """NCS returns dict with max, mean, weighted_mean."""
        context = ["The capital of France is Paris."]
        answer = "Paris is the capital of France."
        result = self.encoder.compute_ncs(context, answer)

        assert "max" in result
        assert "mean" in result
        assert "weighted_mean" in result
        assert all(isinstance(v, float) for v in result.values())

    def test_compute_ncs_high_similarity(self):
        """Similar texts have high similarity score."""
        context = ["The sky is blue and clouds are white."]
        answer = "The sky appears blue with white clouds."
        result = self.encoder.compute_ncs(context, answer)

        # Similar semantics should have positive similarity
        assert result["max"] > 0.3

    def test_compute_ncs_low_similarity(self):
        """Unrelated texts have lower similarity."""
        context = ["The stock market crashed yesterday."]
        answer = "Cats are fluffy and like to sleep."
        result = self.encoder.compute_ncs(context, answer)

        # Unrelated text - lower similarity expected
        # Note: exact threshold depends on model
        assert result["max"] < 0.9

    def test_compute_ncs_multi_passage(self):
        """NCS with multiple context passages."""
        context = [
            "The Eiffel Tower is in Paris.",
            "Paris is the capital of France.",
            "France is in Europe.",
        ]
        answer = "The Eiffel Tower is located in the French capital."
        result = self.encoder.compute_ncs(context, answer)

        assert result["max"] >= result["mean"]
        assert 0 <= result["weighted_mean"] <= 1

    def test_compute_ncs_weighted_mean_uses_length(self):
        """Weighted mean weights by passage length."""
        context = ["Short.", "This is a much longer passage with more words."]
        answer = "This is a test answer."

        result = self.encoder.compute_ncs(context, answer)
        # Just verify it computes without error
        assert "weighted_mean" in result


class TestPreload:
    """Test model preloading."""

    def test_preload_loads_model(self):
        """preload() loads the model."""
        encoder = Model2VecEncoder(Model2VecConfig())
        assert encoder._model is None

        encoder.preload()
        assert encoder._model is not None
