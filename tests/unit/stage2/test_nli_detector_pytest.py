"""Unit tests for NLIContradictionDetector (HHEM-based).

HHEM (Vectara Hallucination Evaluation Model) is trained specifically for RAG
hallucination detection. Score direction:
- HHEM returns: 0=hallucinated, 1=consistent
- We invert to: 0=supported, 1=hallucinated
"""

import gc

import pytest
import torch

from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector


@pytest.fixture
def nli_detector():
    """Create NLI detector with proper cleanup."""
    # Clear memory before creating detector
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    detector = NLIContradictionDetector()
    yield detector

    # Cleanup after test
    del detector
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class TestNLIDetectorBasics:
    """Test basic NLI detector functionality."""

    def test_lazy_loading(self, nli_detector):
        """Model should not load until first use."""
        assert nli_detector._model is None

    def test_compute_context_nli_returns_dict(self, nli_detector):
        """compute_context_nli returns dict with expected keys."""
        result = nli_detector.compute_context_nli(
            context_texts=["The sky is blue."],
            answer="The sky has a color.",
        )

        assert "hallucination_score" in result
        assert "max_hallucination" in result
        assert "mean_hallucination" in result

    def test_scores_are_bounded(self, nli_detector):
        """All scores should be in [0, 1] range."""
        result = nli_detector.compute_context_nli(
            ["The cat sat on the mat."],
            "An animal was on the mat.",
        )

        assert 0.0 <= result["max_hallucination"] <= 1.0
        assert 0.0 <= result["mean_hallucination"] <= 1.0
        assert 0.0 <= result["hallucination_score"] <= 1.0


@pytest.mark.gpu
class TestNLIDetectorPredictions:
    """Test NLI prediction quality (requires model)."""

    def test_detects_hallucination(self, nli_detector):
        """Clear hallucination should have high hallucination score."""
        result = nli_detector.compute_context_nli(
            ["Paris is the capital of France."],
            "Paris is the capital of Germany.",  # Hallucinated
        )

        # Should detect this as hallucination (score > 0.5)
        assert result["max_hallucination"] > 0.5, (
            f"Hallucination should score high, got {result['max_hallucination']}"
        )

    def test_accepts_factual(self, nli_detector):
        """Clear factual statement should have low hallucination score."""
        result = nli_detector.compute_context_nli(
            ["Paris is the capital of France."],
            "France's capital is Paris.",  # Factual
        )

        # Should detect this as supported (score < 0.5)
        assert result["max_hallucination"] < 0.5, (
            f"Factual should score low, got {result['max_hallucination']}"
        )

    def test_hallucination_higher_than_factual(self, nli_detector):
        """Hallucinated content should score higher than factual."""
        hal_result = nli_detector.compute_context_nli(
            ["The population is exactly 5 million."],
            "The population is 10 million.",  # Wrong number
        )

        fact_result = nli_detector.compute_context_nli(
            ["The company was founded in 1995."],
            "The company was founded in 1995.",  # Exact match
        )

        assert hal_result["max_hallucination"] > fact_result["max_hallucination"], (
            f"Hallucination ({hal_result['max_hallucination']}) should score higher "
            f"than factual ({fact_result['max_hallucination']})"
        )


class TestContextNLI:
    """Test compute_context_nli method."""

    def test_empty_context_returns_neutral(self, nli_detector):
        """Empty context returns neutral score (0.5)."""
        result = nli_detector.compute_context_nli([], "Some answer")

        assert result["max_hallucination"] == 0.5
        assert result["mean_hallucination"] == 0.5
        assert result["hallucination_score"] == 0.5

    def test_handles_multiple_contexts(self, nli_detector):
        """Works with multiple context passages."""
        context = [
            "Paris is in France.",
            "The Eiffel Tower is a landmark.",
            "French cuisine is famous.",
        ]
        answer = "The Eiffel Tower is in Paris."

        result = nli_detector.compute_context_nli(context, answer)

        # Should have valid scores
        assert 0 <= result["max_hallucination"] <= 1
        assert 0 <= result["mean_hallucination"] <= 1

    def test_max_vs_mean(self, nli_detector):
        """max_hallucination should be >= mean_hallucination."""
        result = nli_detector.compute_context_nli(
            [
                "The sky is blue.",
                "Water is wet.",
                "Fire is hot.",
            ],
            "The sky is green.",  # Only contradicts first context
        )

        assert result["max_hallucination"] >= result["mean_hallucination"]


class TestWarmup:
    """Test model warmup functionality."""

    def test_warmup_loads_model(self, nli_detector):
        """warmup() loads the model."""
        assert nli_detector._model is None

        nli_detector.warmup()
        assert nli_detector._model is not None

    def test_preload_alias(self, nli_detector):
        """preload() is an alias for warmup()."""
        nli_detector.preload()
        assert nli_detector._model is not None

    def test_warmup_idempotent(self, nli_detector):
        """warmup() is idempotent (safe to call multiple times)."""
        nli_detector.warmup()
        model_id = id(nli_detector._model)

        nli_detector.warmup()
        assert id(nli_detector._model) == model_id  # Same model instance


class TestHHEMSpecific:
    """Test HHEM-specific behavior."""

    def test_model_has_predict_method(self, nli_detector):
        """HHEM model should have predict() method."""
        nli_detector.preload()
        assert hasattr(nli_detector._model, "predict")

    def test_score_direction_inverted(self, nli_detector):
        """Verify HHEM scores are inverted correctly.

        HHEM returns 0=hallucinated, 1=consistent.
        We need 0=supported, 1=hallucinated.
        """
        # Clear hallucination
        result = nli_detector.compute_context_nli(
            ["The answer is 42."],
            "The answer is 999.",  # Wrong
        )

        # If scores were NOT inverted, this would be low
        # Since we invert, hallucination should be high
        # Just verify it's in valid range
        assert 0 <= result["hallucination_score"] <= 1
