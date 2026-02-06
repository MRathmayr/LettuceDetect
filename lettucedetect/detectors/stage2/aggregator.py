"""Score aggregation and routing logic for Stage 2."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lettucedetect.cascade.types import RoutingDecision, StageResult
from lettucedetect.detectors.stage2.config import AggregatorConfig

logger = logging.getLogger(__name__)


@dataclass
class Stage2Scores:
    """Scores from Stage 2 components.

    All scores use high=supported convention (inverted for hallucination_score).
    """

    ncs_score: float  # 0.0 = dissimilar, 1.0 = similar
    nli_score: float  # 0.0 = contradiction, 1.0 = non-contradiction
    stage1_agreement: float | None = None  # Agreement from Stage 1 (if available)


class Stage2Aggregator:
    """Aggregates Stage 2 component scores with routing logic."""

    def __init__(self, weights: dict[str, float], config: AggregatorConfig | None = None):
        """Initialize aggregator with component weights.

        Args:
            weights: Dict with keys "ncs", "nli" and float weights
            config: AggregatorConfig with thresholds
        """
        self.weights = weights
        self.config = config or AggregatorConfig()

    def aggregate(self, scores: Stage2Scores) -> tuple[float, bool]:
        """Compute weighted average of component scores.

        All component scores use convention: high = supported, low = hallucination.
        Final score is converted to: high = hallucination probability.

        With calibrated voting enabled, scores are converted to binary votes
        using per-component optimal thresholds before weighted averaging.

        Args:
            scores: Stage2Scores with ncs_score and nli_score.

        Returns:
            (hallucination_score, is_hallucination) where:
            - hallucination_score: 0.0 = supported, 1.0 = hallucinated
            - is_hallucination: True if score >= 0.5
        """
        # Support scores: high = supported, low = hallucination
        support_scores = {
            "ncs": scores.ncs_score,
            "nli": scores.nli_score,
        }

        if self.config.use_calibrated_voting:
            hallucination_score = self._calibrated_voting(support_scores)
        else:
            hallucination_score = self._raw_weighted_average(support_scores)

        is_hallucination = hallucination_score >= 0.5
        return hallucination_score, is_hallucination

    def _calibrated_voting(self, support_scores: dict[str, float]) -> float:
        """Compute weighted vote using calibrated binary thresholds.

        Converts support scores to hallucination scores, then applies
        per-component optimal thresholds to get binary votes.

        Args:
            support_scores: Dict of name -> support score (high = supported)

        Returns:
            Weighted average of binary votes (0.0 to 1.0, hallucination direction)
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for name, support_score in support_scores.items():
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            # Convert support score to hallucination score
            hal_score = 1.0 - support_score
            threshold = self.config.optimal_thresholds.get(name, 0.5)
            vote = 1.0 if hal_score >= threshold else 0.0
            weighted_sum += vote * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _raw_weighted_average(self, support_scores: dict[str, float]) -> float:
        """Compute weighted average of raw scores (original behavior).

        Args:
            support_scores: Dict of name -> support score (high = supported)

        Returns:
            Hallucination score (weighted average converted from support)
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for name, score in support_scores.items():
            weight = self.weights.get(name, 0.0)
            if weight > 0:
                weighted_sum += score * weight
                total_weight += weight

        if total_weight == 0:
            support_score = 0.5
        else:
            support_score = weighted_sum / total_weight

        return 1.0 - support_score

    def make_routing_decision(
        self, hallucination_score: float, agreement: float, has_next_stage: bool
    ) -> RoutingDecision:
        """Determine routing based on hallucination score, agreement, and stage availability.

        Routing logic:
        - hallucination_score >= threshold_high (0.85): RETURN_CONFIDENT (hallucinated)
        - hallucination_score <= threshold_low (0.3): RETURN_CONFIDENT (supported)
        - agreement < agreement_threshold: ESCALATE (components disagree)
        - Otherwise (uncertain zone):
          - If has_next_stage: ESCALATE
          - Else: RETURN_UNCERTAIN

        Args:
            hallucination_score: Score where 0.0 = supported, 1.0 = hallucinated
            agreement: Ensemble agreement (0.0-1.0)
            has_next_stage: Whether Stage 3 is available

        Returns:
            RoutingDecision enum value
        """
        # Check if in confident zone
        in_confident_zone = (
            hallucination_score >= self.config.threshold_high
            or hallucination_score <= self.config.threshold_low
        )

        # Check if agreement is sufficient
        has_sufficient_agreement = agreement >= self.config.agreement_threshold

        # Return confident only if both in confident zone AND has agreement
        if in_confident_zone and has_sufficient_agreement:
            return RoutingDecision.RETURN_CONFIDENT

        # Uncertain or disagreeing - escalate if possible
        if has_next_stage:
            return RoutingDecision.ESCALATE
        else:
            return RoutingDecision.RETURN_UNCERTAIN

    def create_stage_result(
        self,
        scores: Stage2Scores,
        hallucination_score: float,
        is_hallucination: bool,
        latency_ms: float,
        output: list[dict],
        has_next_stage: bool,
    ) -> StageResult:
        """Build complete StageResult for cascade integration.

        Args:
            scores: Stage2Scores with component scores
            hallucination_score: Aggregated hallucination score
            is_hallucination: Binary prediction
            latency_ms: Stage latency in milliseconds
            output: Formatted output (tokens or spans)
            has_next_stage: Whether Stage 3 is available

        Returns:
            StageResult for cascade
        """
        # Compute Stage 2's own agreement between NCS and NLI
        component_scores = [scores.ncs_score, scores.nli_score]
        valid_scores = [s for s in component_scores if s is not None]
        if len(valid_scores) > 1:
            stage2_agreement = 1.0 - min(float(np.std(valid_scores)) * 2, 1.0)
        else:
            stage2_agreement = 1.0

        # Combined agreement: consider stage1 agreement if available
        # Weight: 70% stage2, 30% stage1
        if scores.stage1_agreement is not None:
            agreement = 0.7 * stage2_agreement + 0.3 * scores.stage1_agreement
        else:
            agreement = stage2_agreement

        routing = self.make_routing_decision(hallucination_score, agreement, has_next_stage)

        # Build evidence dict
        evidence = {
            "ncs_computed": scores.ncs_score is not None,
            "nli_computed": scores.nli_score is not None,
            "stage1_agreement": scores.stage1_agreement,
            "stage2_agreement": stage2_agreement,
        }

        # Generate routing reason
        if routing == RoutingDecision.RETURN_CONFIDENT:
            if hallucination_score >= self.config.threshold_high:
                routing_reason = f"High confidence hallucination (score={hallucination_score:.2f}, agreement={agreement:.2f})"
            else:
                routing_reason = f"High confidence supported (score={hallucination_score:.2f}, agreement={agreement:.2f})"
        elif routing == RoutingDecision.ESCALATE:
            if agreement < self.config.agreement_threshold:
                routing_reason = f"Components disagree, escalating (score={hallucination_score:.2f}, agreement={agreement:.2f})"
            else:
                routing_reason = f"Uncertain, escalating to Stage 3 (score={hallucination_score:.2f}, agreement={agreement:.2f})"
        else:
            routing_reason = f"Uncertain, no further stages (score={hallucination_score:.2f}, agreement={agreement:.2f})"

        return StageResult(
            stage_name="stage2",
            hallucination_score=hallucination_score,
            agreement=agreement,
            is_hallucination=is_hallucination,
            routing_decision=routing,
            latency_ms=latency_ms,
            output=output,
            component_scores={
                "ncs": scores.ncs_score,
                "nli": scores.nli_score,
            },
            evidence=evidence,
            routing_reason=routing_reason,
        )
