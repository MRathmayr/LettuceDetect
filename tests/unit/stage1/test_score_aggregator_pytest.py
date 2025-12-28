"""Unit tests for ScoreAggregator."""

import pytest

from lettucedetect.cascade.types import AugmentationResult
from lettucedetect.detectors.stage1.aggregator import (
    AggregationConfig,
    ScoreAggregator,
)


class TestScoreExtraction:
    """Test score extraction from predictions."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 0.5, "ner": 0.25, "numeric": 0.25}
        self.aggregator = ScoreAggregator(weights)

    def test_transformer_score_extraction_with_hallucinations(self):
        """Extract max hallucination probability."""
        preds = [
            {"token": "a", "pred": 0, "prob": 0.1},
            {"token": "b", "pred": 1, "prob": 0.8},
            {"token": "c", "pred": 1, "prob": 0.6},
        ]
        score = self.aggregator._extract_transformer_score(preds)
        assert score == 0.8

    def test_transformer_score_extraction_no_hallucinations(self):
        """No hallucination tokens = 0 score."""
        preds = [
            {"token": "a", "pred": 0, "prob": 0.1},
            {"token": "b", "pred": 0, "prob": 0.2},
        ]
        score = self.aggregator._extract_transformer_score(preds)
        assert score == 0.0

    def test_transformer_score_extraction_empty(self):
        """Empty predictions = 0 score."""
        score = self.aggregator._extract_transformer_score([])
        assert score == 0.0


class TestScoreNormalization:
    """Test score direction normalization."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 0.5, "ner": 0.25, "numeric": 0.25}
        self.aggregator = ScoreAggregator(weights)

    def test_augmentation_score_inversion(self):
        """Augmentation scores are inverted (high=supported -> high=hallucination)."""
        # score=1.0 (supported) -> 0.0 (not hallucinated)
        assert self.aggregator._normalize_augmentation_score(1.0) == 0.0
        # score=0.0 (hallucinated) -> 1.0 (hallucinated)
        assert self.aggregator._normalize_augmentation_score(0.0) == 1.0
        # score=0.5 (uncertain) -> 0.5 (uncertain)
        assert self.aggregator._normalize_augmentation_score(0.5) == 0.5


class TestAggregation:
    """Test score aggregation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 0.5, "ner": 0.25, "numeric": 0.25}
        self.aggregator = ScoreAggregator(weights)

    def test_weighted_average_all_hallucination(self):
        """All components indicate hallucination."""
        preds = [{"token": "x", "pred": 1, "prob": 1.0}]
        aug_results = {
            "ner": AugmentationResult(score=0.0, confidence=0.8, details={}, flagged_spans=[]),
            "numeric": AugmentationResult(score=0.0, confidence=0.9, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        assert result.hallucination_score == 1.0
        assert result.confident is True

    def test_weighted_average_all_supported(self):
        """All components indicate support."""
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]
        aug_results = {
            "ner": AugmentationResult(score=1.0, confidence=0.8, details={}, flagged_spans=[]),
            "numeric": AugmentationResult(score=1.0, confidence=0.9, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        assert result.hallucination_score == 0.0
        assert result.confident is True

    def test_weighted_average_mixed(self):
        """Mixed signals from components."""
        preds = [{"token": "x", "pred": 1, "prob": 0.8}]  # transformer: 0.8 hallucination
        aug_results = {
            # ner: 1.0 support -> 0.0 hallucination
            "ner": AugmentationResult(score=1.0, confidence=0.8, details={}, flagged_spans=[]),
            # numeric: 0.0 support -> 1.0 hallucination
            "numeric": AugmentationResult(score=0.0, confidence=0.9, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        # 0.5 * 0.8 + 0.25 * 0.0 + 0.25 * 1.0 = 0.4 + 0 + 0.25 = 0.65
        assert abs(result.hallucination_score - 0.65) < 0.01

    def test_missing_augmentation_skipped(self):
        """Augmentation with None score is skipped."""
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]
        aug_results = {
            "ner": AugmentationResult(score=None, confidence=0.0, details={}, flagged_spans=[]),
            "numeric": AugmentationResult(score=1.0, confidence=0.9, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        # Only transformer (0.0) and numeric (0.0 after inversion)
        # weights: 0.5 transformer + 0.25 numeric
        # score: (0.5 * 0.0 + 0.25 * 0.0) / 0.75 = 0.0
        assert result.hallucination_score == 0.0


class TestConfidenceCalculation:
    """Test confidence calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 1.0}
        self.aggregator = ScoreAggregator(weights)

    def test_confidence_at_extremes(self):
        """Extreme scores have high confidence."""
        # Score = 1.0 -> confidence = 1.0
        preds = [{"token": "x", "pred": 1, "prob": 1.0}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confidence == 1.0

        # Score = 0.0 -> confidence = 1.0
        preds = [{"token": "x", "pred": 0, "prob": 0.0}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confidence == 1.0

    def test_confidence_at_uncertainty(self):
        """Mid-range score has low confidence."""
        # Score = 0.5 -> confidence = 0.0
        preds = [{"token": "x", "pred": 1, "prob": 0.5}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confidence == 0.0


class TestRoutingDecisions:
    """Test routing decision logic."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 1.0}
        config = AggregationConfig(
            confidence_threshold_high=0.7,
            confidence_threshold_low=0.4,
        )
        self.aggregator = ScoreAggregator(weights, config)

    def test_confident_high_hallucination(self):
        """High hallucination score is confident."""
        preds = [{"token": "x", "pred": 1, "prob": 0.9}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is True
        assert result.escalate is False
        assert "High confidence hallucination" in result.routing_reason

    def test_confident_low_hallucination(self):
        """Low hallucination score (supported) is confident."""
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is True
        assert result.escalate is False
        assert "High confidence supported" in result.routing_reason

    def test_escalate_uncertain(self):
        """Uncertain score escalates."""
        preds = [{"token": "x", "pred": 1, "prob": 0.5}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is False
        assert result.escalate is True
        assert "escalating" in result.routing_reason

    def test_below_escalation_threshold(self):
        """Score below escalation threshold doesn't escalate."""
        preds = [{"token": "x", "pred": 1, "prob": 0.35}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is False
        assert result.escalate is False
        assert "below escalation threshold" in result.routing_reason


class TestSpanMerging:
    """Test span merging logic."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 0.5, "ner": 0.5}
        self.aggregator = ScoreAggregator(weights)

    def test_non_overlapping_spans(self):
        """Non-overlapping spans are all kept."""
        preds = [
            {"token": "a", "pred": 1, "prob": 0.8, "start": 0, "end": 5},
        ]
        aug_results = {
            "ner": AugmentationResult(
                score=0.5,
                confidence=0.8,
                details={},
                flagged_spans=[{"start": 10, "end": 15, "text": "b", "confidence": 0.7}],
            )
        }
        result = self.aggregator.aggregate(preds, aug_results)
        assert len(result.merged_spans) == 2

    def test_overlapping_spans_keep_higher_confidence(self):
        """Overlapping spans keep higher confidence."""
        preds = [
            {"token": "word", "pred": 1, "prob": 0.6, "start": 0, "end": 4},
        ]
        aug_results = {
            "ner": AugmentationResult(
                score=0.5,
                confidence=0.8,
                details={},
                flagged_spans=[{"start": 0, "end": 4, "text": "word", "confidence": 0.9}],
            )
        }
        result = self.aggregator.aggregate(preds, aug_results)
        assert len(result.merged_spans) == 1
        assert result.merged_spans[0]["confidence"] == 0.9
        assert result.merged_spans[0]["source"] == "ner"

    def test_spans_sorted_by_start(self):
        """Merged spans are sorted by start position."""
        preds = [
            {"token": "b", "pred": 1, "prob": 0.8, "start": 10, "end": 15},
            {"token": "a", "pred": 1, "prob": 0.8, "start": 0, "end": 5},
        ]
        result = self.aggregator.aggregate(preds, {})
        assert result.merged_spans[0]["start"] == 0
        assert result.merged_spans[1]["start"] == 10
