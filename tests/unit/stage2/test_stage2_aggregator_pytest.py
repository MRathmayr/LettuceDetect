"""Unit tests for Stage2Aggregator."""

import pytest

from lettucedetect.cascade.types import RoutingDecision
from lettucedetect.detectors.stage2.aggregator import (
    AggregatorConfig,
    Stage2Aggregator,
    Stage2Scores,
)
from lettucedetect.detectors.stage2.config import AggregatorConfig as InternalConfig


class TestStage2Scores:
    """Test Stage2Scores dataclass."""

    def test_create_scores(self):
        """Can create Stage2Scores with all fields."""
        scores = Stage2Scores(
            ncs_score=0.8,
            nli_score=0.7,
            stage1_agreement=0.5,
        )

        assert scores.ncs_score == 0.8
        assert scores.nli_score == 0.7
        assert scores.stage1_agreement == 0.5

    def test_create_scores_without_stage1(self):
        """Can create Stage2Scores without stage1_agreement."""
        scores = Stage2Scores(ncs_score=0.8, nli_score=0.7)
        assert scores.stage1_agreement is None


class TestAggregation:
    """Test score aggregation logic.

    Tests use raw weighted average (use_calibrated_voting=False) for predictable scores.
    """

    def setup_method(self):
        """Set up aggregator for tests."""
        self.weights = {"ncs": 0.5, "nli": 0.5}
        config = InternalConfig(use_calibrated_voting=False)
        self.aggregator = Stage2Aggregator(self.weights, config)

    def test_equal_weights_equal_scores(self):
        """Equal weights with equal scores returns expected value."""
        scores = Stage2Scores(ncs_score=0.8, nli_score=0.8)
        hal_score, is_hal = self.aggregator.aggregate(scores)

        # support_score = 0.8, hal_score = 1 - 0.8 = 0.2
        assert abs(hal_score - 0.2) < 0.01
        assert is_hal is False

    def test_different_weights(self):
        """Different weights affect aggregation correctly."""
        weights = {"ncs": 0.7, "nli": 0.3}
        config = InternalConfig(use_calibrated_voting=False)
        aggregator = Stage2Aggregator(weights, config)

        scores = Stage2Scores(ncs_score=1.0, nli_score=0.0)
        hal_score, _ = aggregator.aggregate(scores)

        # support = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        # hal_score = 1 - 0.7 = 0.3
        assert abs(hal_score - 0.3) < 0.01

    def test_high_support_low_hallucination(self):
        """High support scores result in low hallucination score."""
        scores = Stage2Scores(ncs_score=0.9, nli_score=0.9)
        hal_score, is_hal = self.aggregator.aggregate(scores)

        assert hal_score < 0.2
        assert is_hal is False

    def test_low_support_high_hallucination(self):
        """Low support scores result in high hallucination score."""
        scores = Stage2Scores(ncs_score=0.1, nli_score=0.1)
        hal_score, is_hal = self.aggregator.aggregate(scores)

        assert hal_score > 0.8
        assert is_hal is True

    def test_boundary_at_half(self):
        """Hallucination at exactly 0.5 is classified as hallucination."""
        scores = Stage2Scores(ncs_score=0.5, nli_score=0.5)
        hal_score, is_hal = self.aggregator.aggregate(scores)

        assert abs(hal_score - 0.5) < 0.01
        assert is_hal is True  # >= 0.5 is hallucination

    def test_zero_weights_returns_neutral(self):
        """Zero weights return neutral 0.5."""
        config = InternalConfig(use_calibrated_voting=False)
        aggregator = Stage2Aggregator({"ncs": 0, "nli": 0}, config)
        scores = Stage2Scores(ncs_score=0.9, nli_score=0.1)
        hal_score, _ = aggregator.aggregate(scores)

        assert hal_score == 0.5


class TestStage1AgreementIntegration:
    """Test Stage 1 agreement integration in create_stage_result."""

    def test_stage1_agreement_affects_combined_agreement(self):
        """Stage 1 agreement is combined with Stage 2 agreement (70/30 weighting)."""
        aggregator = Stage2Aggregator({"ncs": 0.5, "nli": 0.5})

        # Both scores same -> stage2_agreement = 1.0
        scores_without = Stage2Scores(ncs_score=0.8, nli_score=0.8)
        scores_with = Stage2Scores(ncs_score=0.8, nli_score=0.8, stage1_agreement=0.0)

        result_without = aggregator.create_stage_result(
            scores=scores_without,
            hallucination_score=0.2,
            is_hallucination=False,
            latency_ms=10.0,
            output=[],
            has_next_stage=True,
        )
        result_with = aggregator.create_stage_result(
            scores=scores_with,
            hallucination_score=0.2,
            is_hallucination=False,
            latency_ms=10.0,
            output=[],
            has_next_stage=True,
        )

        # Without stage1_agreement: agreement = 1.0
        # With stage1_agreement=0.0: agreement = 0.7 * 1.0 + 0.3 * 0.0 = 0.7
        assert result_without.agreement == 1.0
        assert abs(result_with.agreement - 0.7) < 0.01

    def test_hallucination_score_unaffected_by_stage1_agreement(self):
        """Stage 1 agreement does NOT affect hallucination score calculation."""
        aggregator = Stage2Aggregator({"ncs": 0.5, "nli": 0.5})

        scores_without = Stage2Scores(ncs_score=0.8, nli_score=0.8)
        scores_with = Stage2Scores(ncs_score=0.8, nli_score=0.8, stage1_agreement=0.5)

        hal_without, _ = aggregator.aggregate(scores_without)
        hal_with, _ = aggregator.aggregate(scores_with)

        # Hallucination score should be the same regardless of stage1_agreement
        # because aggregate() only uses ncs_score and nli_score
        assert hal_without == hal_with


class TestRoutingDecisions:
    """Test routing decision logic.

    Routing uses hallucination_score thresholds AND agreement:
    - hallucination_score >= threshold_high (0.85) AND agreement >= 0.5 -> confident hallucination
    - hallucination_score <= threshold_low (0.3) AND agreement >= 0.5 -> confident supported
    - agreement < agreement_threshold (0.5) -> escalate
    - Otherwise -> uncertain, escalate
    """

    def setup_method(self):
        """Set up aggregator with specific thresholds."""
        config = InternalConfig(
            threshold_high=0.85,
            threshold_low=0.3,
            agreement_threshold=0.5,
        )
        self.aggregator = Stage2Aggregator({"ncs": 0.5, "nli": 0.5}, config)

    def test_high_hallucination_returns_confident(self):
        """High hallucination score (>=0.85) with agreement returns RETURN_CONFIDENT."""
        decision = self.aggregator.make_routing_decision(
            hallucination_score=0.9, agreement=1.0, has_next_stage=True
        )
        assert decision == RoutingDecision.RETURN_CONFIDENT

    def test_low_hallucination_returns_confident(self):
        """Low hallucination score (<=0.3) with agreement returns RETURN_CONFIDENT."""
        decision = self.aggregator.make_routing_decision(
            hallucination_score=0.2, agreement=1.0, has_next_stage=True
        )
        assert decision == RoutingDecision.RETURN_CONFIDENT

    def test_uncertain_with_next_stage_escalates(self):
        """Uncertain score with next stage escalates."""
        decision = self.aggregator.make_routing_decision(
            hallucination_score=0.5, agreement=1.0, has_next_stage=True
        )
        assert decision == RoutingDecision.ESCALATE

    def test_uncertain_without_next_stage_returns_uncertain(self):
        """Uncertain score without next stage returns uncertain."""
        decision = self.aggregator.make_routing_decision(
            hallucination_score=0.5, agreement=1.0, has_next_stage=False
        )
        assert decision == RoutingDecision.RETURN_UNCERTAIN

    def test_threshold_boundary_high(self):
        """Exactly at high threshold with agreement returns confident."""
        decision = self.aggregator.make_routing_decision(
            hallucination_score=0.85, agreement=1.0, has_next_stage=True
        )
        assert decision == RoutingDecision.RETURN_CONFIDENT

    def test_threshold_boundary_low(self):
        """Exactly at low threshold (0.3) with agreement returns confident."""
        decision = self.aggregator.make_routing_decision(
            hallucination_score=0.3, agreement=1.0, has_next_stage=True
        )
        assert decision == RoutingDecision.RETURN_CONFIDENT

    def test_low_agreement_escalates_even_in_confident_zone(self):
        """Low agreement causes escalation even in confident zone."""
        decision = self.aggregator.make_routing_decision(
            hallucination_score=0.9, agreement=0.3, has_next_stage=True
        )
        assert decision == RoutingDecision.ESCALATE


class TestStageResultCreation:
    """Test StageResult creation."""

    def setup_method(self):
        """Set up aggregator for tests."""
        self.aggregator = Stage2Aggregator({"ncs": 0.5, "nli": 0.5})

    def test_create_stage_result_basic(self):
        """Can create basic StageResult."""
        scores = Stage2Scores(ncs_score=0.8, nli_score=0.7)

        result = self.aggregator.create_stage_result(
            scores=scores,
            hallucination_score=0.25,
            is_hallucination=False,
            latency_ms=15.5,
            output=[],
            has_next_stage=True,
        )

        assert result.stage_name == "stage2"
        assert result.is_hallucination is False
        assert result.latency_ms == 15.5

    def test_stage_result_component_scores(self):
        """StageResult contains correct component_scores."""
        scores = Stage2Scores(ncs_score=0.8, nli_score=0.7)

        result = self.aggregator.create_stage_result(
            scores=scores,
            hallucination_score=0.25,
            is_hallucination=False,
            latency_ms=10.0,
            output=[],
            has_next_stage=False,
        )

        assert result.component_scores["ncs"] == 0.8
        assert result.component_scores["nli"] == 0.7
        assert result.hallucination_score == 0.25
        assert "supported" in result.routing_reason.lower()

    def test_agreement_calculation(self):
        """Agreement is computed from component score variance."""
        scores = Stage2Scores(ncs_score=0.8, nli_score=0.8)

        # When both components agree (same score), agreement should be 1.0
        result = self.aggregator.create_stage_result(
            scores=scores,
            hallucination_score=0.2,
            is_hallucination=False,
            latency_ms=10.0,
            output=[],
            has_next_stage=True,
        )

        # Both NCS and NLI = 0.8, std = 0 -> agreement = 1.0
        assert result.agreement == 1.0

    def test_agreement_with_disagreement(self):
        """Agreement decreases when components disagree."""
        scores = Stage2Scores(ncs_score=0.9, nli_score=0.1)

        result = self.aggregator.create_stage_result(
            scores=scores,
            hallucination_score=0.5,
            is_hallucination=True,
            latency_ms=10.0,
            output=[],
            has_next_stage=True,
        )

        # NCS=0.9, NLI=0.1, std=0.4 -> agreement = 1.0 - min(0.4*2, 1.0) = 0.2
        assert abs(result.agreement - 0.2) < 0.01


class TestCalibratedVoting:
    """Test calibrated voting behavior (use_calibrated_voting=True).

    Calibrated voting converts support scores to hallucination scores,
    then applies per-component optimal thresholds to get binary votes.
    """

    def setup_method(self):
        """Set up aggregator with calibrated voting enabled."""
        self.weights = {"ncs": 0.5, "nli": 0.5}
        self.config = InternalConfig(
            use_calibrated_voting=True,
            optimal_thresholds={
                "ncs": 0.5,  # Simpler thresholds for testing
                "nli": 0.5,
            },
        )
        self.aggregator = Stage2Aggregator(self.weights, self.config)

    def test_calibrated_all_below_threshold_hallucination(self):
        """High support (low hallucination) below threshold -> vote 0 -> hal_score = 0."""
        # Support scores: ncs=0.8, nli=0.8
        # Hallucination scores: 1-0.8=0.2, 1-0.8=0.2
        # Both < 0.5 threshold -> votes = 0, 0 -> weighted avg = 0.0
        scores = Stage2Scores(ncs_score=0.8, nli_score=0.8)
        hal_score, is_hal = self.aggregator.aggregate(scores)
        assert hal_score == 0.0
        assert is_hal is False

    def test_calibrated_all_above_threshold_hallucination(self):
        """Low support (high hallucination) above threshold -> vote 1 -> hal_score = 1."""
        # Support scores: ncs=0.2, nli=0.2
        # Hallucination scores: 1-0.2=0.8, 1-0.2=0.8
        # Both >= 0.5 threshold -> votes = 1, 1 -> weighted avg = 1.0
        scores = Stage2Scores(ncs_score=0.2, nli_score=0.2)
        hal_score, is_hal = self.aggregator.aggregate(scores)
        assert hal_score == 1.0
        assert is_hal is True

    def test_calibrated_mixed_votes(self):
        """Mixed support scores -> mixed votes."""
        # Support scores: ncs=0.8 (hal=0.2 < 0.5 -> vote 0), nli=0.2 (hal=0.8 >= 0.5 -> vote 1)
        # Weighted: (0.5*0 + 0.5*1) / 1.0 = 0.5
        scores = Stage2Scores(ncs_score=0.8, nli_score=0.2)
        hal_score, is_hal = self.aggregator.aggregate(scores)
        assert hal_score == 0.5
        assert is_hal is True  # >= 0.5 is hallucination

    def test_calibrated_voting_enabled_by_default(self):
        """Calibrated voting is enabled by default in AggregatorConfig."""
        config = InternalConfig()
        assert config.use_calibrated_voting is True

    def test_default_thresholds_from_ragtruth(self):
        """Default thresholds match RAGTruth benchmark values."""
        config = InternalConfig()
        assert config.optimal_thresholds["ncs"] == 0.123
        assert config.optimal_thresholds["nli"] == 0.472
