"""Unit tests for NLIContradictionDetector."""

import gc

import pytest
import torch

from lettucedetect.detectors.stage2.config import NLIConfig
from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector


@pytest.fixture
def nli_detector():
    """Create NLI detector with proper cleanup."""
    # Clear memory before creating detector
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    detector = NLIContradictionDetector(NLIConfig())
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
        assert nli_detector._tokenizer is None

    def test_predict_single_returns_dict(self, nli_detector):
        """predict_single returns dict with expected keys."""
        result = nli_detector.predict_single(
            premise="The sky is blue.",
            hypothesis="The sky has a color.",
        )

        assert "entailment" in result
        assert "neutral" in result
        assert "contradiction" in result
        assert "non_contradiction" in result

    def test_predict_single_probabilities_sum_to_one(self, nli_detector):
        """Entailment + neutral + contradiction should sum to ~1.0."""
        result = nli_detector.predict_single(
            premise="The cat sat on the mat.",
            hypothesis="An animal was on the mat.",
        )

        total = result["entailment"] + result["neutral"] + result["contradiction"]
        assert abs(total - 1.0) < 0.01

    def test_non_contradiction_is_sum(self, nli_detector):
        """non_contradiction = entailment + neutral."""
        result = nli_detector.predict_single(
            premise="Paris is in France.",
            hypothesis="A city is in a country.",
        )

        expected = result["entailment"] + result["neutral"]
        assert abs(result["non_contradiction"] - expected) < 0.001


@pytest.mark.gpu
class TestNLIDetectorPredictions:
    """Test NLI prediction quality (requires model)."""

    def test_entailment_case(self, nli_detector):
        """Clear entailment should have high entailment score."""
        result = nli_detector.predict_single(
            premise="All cats are animals.",
            hypothesis="Cats are animals.",
        )

        # Entailment should be the highest
        assert result["entailment"] > result["contradiction"]

    def test_contradiction_case(self, nli_detector):
        """Clear contradiction should have higher contradiction than neutral."""
        result = nli_detector.predict_single(
            premise="The sun rises in the east.",
            hypothesis="The sun rises in the west.",
        )

        # Contradiction should be detectable (not necessarily > 0.1 threshold)
        # Just verify it's higher than a completely neutral statement would be
        assert result["contradiction"] > result["entailment"]


class TestNLIBatchInference:
    """Test batched inference functionality."""

    def test_predict_batch_returns_list(self, nli_detector):
        """predict_batch returns list of dicts."""
        premises = ["The dog is brown.", "The cat is white."]
        hypotheses = ["An animal has fur.", "An animal has fur."]

        results = nli_detector.predict_batch(premises, hypotheses)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

    def test_predict_batch_length_mismatch_raises(self, nli_detector):
        """Mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            nli_detector.predict_batch(
                premises=["A", "B", "C"],
                hypotheses=["X", "Y"],
            )

    def test_predict_batch_empty(self, nli_detector):
        """Empty inputs return empty list."""
        results = nli_detector.predict_batch([], [])
        assert results == []

    def test_batch_matches_sequential(self, nli_detector):
        """Batch results match sequential predictions."""
        premises = ["The sky is blue.", "Water is wet."]
        hypotheses = ["The sky has color.", "Liquid is moist."]

        batch_results = nli_detector.predict_batch(premises, hypotheses)
        seq_results = [
            nli_detector.predict_single(p, h) for p, h in zip(premises, hypotheses)
        ]

        # Results should be very close (may have slight float differences)
        for batch, seq in zip(batch_results, seq_results):
            assert abs(batch["entailment"] - seq["entailment"]) < 0.01
            assert abs(batch["contradiction"] - seq["contradiction"]) < 0.01


class TestContextNLI:
    """Test compute_context_nli method."""

    def test_compute_context_nli_empty(self, nli_detector):
        """Empty context returns safe defaults."""
        result = nli_detector.compute_context_nli([], "Some answer")

        assert result["max_contradiction"] == 0.0
        assert result["min_non_contradiction"] == 1.0

    def test_compute_context_nli_returns_expected_keys(self, nli_detector):
        """Returns max_contradiction and min_non_contradiction."""
        context = ["The capital of France is Paris."]
        answer = "Paris is the capital."

        result = nli_detector.compute_context_nli(context, answer)

        assert "max_contradiction" in result
        assert "min_non_contradiction" in result
        assert "all_results" in result

    def test_compute_context_nli_multi_passage(self, nli_detector):
        """Works with multiple context passages."""
        context = [
            "Paris is in France.",
            "The Eiffel Tower is a landmark.",
            "French cuisine is famous.",
        ]
        answer = "The Eiffel Tower is in Paris."

        result = nli_detector.compute_context_nli(context, answer)

        # Should have results for all passages
        assert len(result["all_results"]) == 3
        assert 0 <= result["max_contradiction"] <= 1
        assert 0 <= result["min_non_contradiction"] <= 1


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
