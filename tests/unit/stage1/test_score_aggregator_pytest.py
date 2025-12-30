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


class TestAggregation:
    """Test score aggregation logic."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 0.5, "ner": 0.25, "numeric": 0.25}
        self.aggregator = ScoreAggregator(weights)

    def test_weighted_average_all_hallucination(self):
        """All components indicate hallucination.

        Unified score direction: 0.0 = supported, 1.0 = hallucinated
        """
        preds = [{"token": "x", "pred": 1, "prob": 1.0}]  # transformer: 1.0 hallucination
        aug_results = {
            "ner": AugmentationResult(score=1.0, evidence={}, details={}, flagged_spans=[]),
            "numeric": AugmentationResult(score=1.0, evidence={}, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        assert result.hallucination_score == 1.0
        assert result.confident is True

    def test_weighted_average_all_supported(self):
        """All components indicate support.

        Unified score direction: 0.0 = supported, 1.0 = hallucinated
        """
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]  # transformer: 0.0 hallucination
        aug_results = {
            "ner": AugmentationResult(score=0.0, evidence={}, details={}, flagged_spans=[]),
            "numeric": AugmentationResult(score=0.0, evidence={}, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        assert result.hallucination_score == 0.0
        assert result.confident is True

    def test_weighted_average_mixed(self):
        """Mixed signals from components.

        Unified score direction: 0.0 = supported, 1.0 = hallucinated
        """
        preds = [{"token": "x", "pred": 1, "prob": 0.8}]  # transformer: 0.8 hallucination
        aug_results = {
            "ner": AugmentationResult(score=0.0, evidence={}, details={}, flagged_spans=[]),  # supported
            "numeric": AugmentationResult(score=1.0, evidence={}, details={}, flagged_spans=[]),  # hallucinated
        }
        result = self.aggregator.aggregate(preds, aug_results)
        # 0.5 * 0.8 + 0.25 * 0.0 + 0.25 * 1.0 = 0.4 + 0 + 0.25 = 0.65
        assert abs(result.hallucination_score - 0.65) < 0.01

    def test_missing_augmentation_skipped(self):
        """Augmentation with None score is skipped."""
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]  # transformer: 0.0 hallucination
        aug_results = {
            "ner": AugmentationResult(score=None, evidence={}, details={}, flagged_spans=[]),
            "numeric": AugmentationResult(score=0.0, evidence={}, details={}, flagged_spans=[]),  # supported
        }
        result = self.aggregator.aggregate(preds, aug_results)
        # Only transformer (0.0) and numeric (0.0)
        # weights: 0.5 transformer + 0.25 numeric
        # score: (0.5 * 0.0 + 0.25 * 0.0) / 0.75 = 0.0
        assert result.hallucination_score == 0.0


class TestAgreementCalculation:
    """Test agreement calculation.

    Agreement is computed from component score variance:
    agreement = 1.0 - min(std(scores) * 2, 1.0)

    - All components agree (std=0) -> agreement=1.0
    - Components strongly disagree (std=0.5) -> agreement=0.0
    """

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 0.5, "ner": 0.5}
        self.aggregator = ScoreAggregator(weights)

    def test_agreement_all_same_scores(self):
        """All components return same score -> perfect agreement."""
        preds = [{"token": "x", "pred": 1, "prob": 0.8}]  # transformer: 0.8
        aug_results = {
            "ner": AugmentationResult(score=0.8, evidence={}, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        # scores = [0.8, 0.8], std = 0 -> agreement = 1.0
        assert result.agreement == 1.0

    def test_agreement_strong_disagreement(self):
        """Components strongly disagree -> low agreement."""
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]  # transformer: 0.0
        aug_results = {
            "ner": AugmentationResult(score=1.0, evidence={}, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        # scores = [0.0, 1.0], std = 0.5 -> agreement = 1.0 - min(0.5 * 2, 1.0) = 0.0
        assert result.agreement == 0.0

    def test_agreement_moderate_disagreement(self):
        """Moderate disagreement -> partial agreement."""
        preds = [{"token": "x", "pred": 1, "prob": 0.6}]  # transformer: 0.6
        aug_results = {
            "ner": AugmentationResult(score=0.4, evidence={}, details={}, flagged_spans=[]),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        # scores = [0.6, 0.4], std = 0.1 -> agreement = 1.0 - 0.2 = 0.8
        assert abs(result.agreement - 0.8) < 0.01

    def test_agreement_single_component(self):
        """Single component -> full agreement (nothing to disagree with)."""
        weights = {"transformer": 1.0}
        aggregator = ScoreAggregator(weights)
        preds = [{"token": "x", "pred": 1, "prob": 0.7}]
        result = aggregator.aggregate(preds, {})
        assert result.agreement == 1.0


class TestRoutingDecisions:
    """Test routing decision logic.

    Routing uses hallucination_score thresholds AND agreement:
    - hallucination_score >= threshold_high (0.7) AND agreement >= 0.5 -> confident hallucination
    - hallucination_score <= threshold_low (0.3) AND agreement >= 0.5 -> confident supported
    - Otherwise -> uncertain, escalate
    """

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 1.0}
        config = AggregationConfig(
            threshold_high=0.7,
            threshold_low=0.3,
            agreement_threshold=0.5,
        )
        self.aggregator = ScoreAggregator(weights, config)

    def test_confident_high_hallucination(self):
        """High hallucination score (>= threshold_high) with agreement is confident."""
        preds = [{"token": "x", "pred": 1, "prob": 0.9}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is True
        assert result.escalate is False
        assert "High confidence hallucination" in result.routing_reason

    def test_confident_low_hallucination(self):
        """Low hallucination score (<= threshold_low) with agreement is confident."""
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is True
        assert result.escalate is False
        assert "High confidence supported" in result.routing_reason

    def test_escalate_uncertain(self):
        """Uncertain score (between thresholds) escalates."""
        preds = [{"token": "x", "pred": 1, "prob": 0.5}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is False
        assert result.escalate is True
        assert "escalating" in result.routing_reason

    def test_escalate_due_to_disagreement(self):
        """Even in confident zone, low agreement causes escalation."""
        weights = {"transformer": 0.5, "ner": 0.5}
        config = AggregationConfig(
            threshold_high=0.7,
            threshold_low=0.3,
            agreement_threshold=0.5,
        )
        aggregator = ScoreAggregator(weights, config)

        # Transformer says supported, NER says hallucinated -> disagreement
        preds = [{"token": "x", "pred": 0, "prob": 0.1}]  # 0.0
        aug_results = {
            "ner": AugmentationResult(score=1.0, evidence={}, details={}, flagged_spans=[]),
        }
        result = aggregator.aggregate(preds, aug_results)
        # Average score = 0.5 (not in confident zone), but also agreement = 0.0
        assert result.confident is False
        assert result.escalate is True

    def test_at_threshold_high_boundary(self):
        """Score exactly at threshold_high is confident hallucination."""
        preds = [{"token": "x", "pred": 1, "prob": 0.7}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is True
        assert result.escalate is False

    def test_at_threshold_low_boundary(self):
        """Score exactly at threshold_low is confident supported."""
        preds = [{"token": "x", "pred": 1, "prob": 0.3}]
        result = self.aggregator.aggregate(preds, {})
        assert result.confident is True
        assert result.escalate is False


class TestEvidenceAggregation:
    """Test evidence aggregation from components."""

    def setup_method(self):
        """Set up test fixtures."""
        weights = {"transformer": 0.5, "ner": 0.25, "numeric": 0.25}
        self.aggregator = ScoreAggregator(weights)

    def test_evidence_includes_transformer_stats(self):
        """Evidence includes transformer token stats."""
        preds = [
            {"token": "a", "pred": 0, "prob": 0.1},
            {"token": "b", "pred": 1, "prob": 0.8},
            {"token": "c", "pred": 1, "prob": 0.6},
        ]
        result = self.aggregator.aggregate(preds, {})
        assert result.evidence["tokens_analyzed"] == 3
        assert result.evidence["hallucination_tokens"] == 2

    def test_evidence_includes_augmentation_evidence(self):
        """Evidence includes augmentation-specific evidence with prefixes."""
        preds = [{"token": "x", "pred": 1, "prob": 0.8}]
        aug_results = {
            "ner": AugmentationResult(
                score=0.5,
                evidence={"entities_checked": 5, "entities_verified": 3},
                details={},
                flagged_spans=[],
            ),
            "numeric": AugmentationResult(
                score=0.2,
                evidence={"numbers_checked": 2, "numbers_verified": 1},
                details={},
                flagged_spans=[],
            ),
        }
        result = self.aggregator.aggregate(preds, aug_results)
        assert result.evidence["ner_entities_checked"] == 5
        assert result.evidence["ner_entities_verified"] == 3
        assert result.evidence["numeric_numbers_checked"] == 2
        assert result.evidence["numeric_numbers_verified"] == 1


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
                evidence={},
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
                evidence={},
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
