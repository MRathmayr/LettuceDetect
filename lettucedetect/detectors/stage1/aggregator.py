"""Score aggregation and routing logic for Stage 1."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from lettucedetect.cascade.types import AugmentationResult

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for score aggregation and routing decisions.

    Thresholds define when we're confident enough to return a result:
    - hallucination_score >= threshold_high → confident it's hallucinated
    - hallucination_score <= threshold_low → confident it's supported
    - agreement < agreement_threshold → escalate even if score is in confident zone
    - Otherwise → uncertain, may escalate to next stage
    """

    threshold_high: float = 0.7  # Above this = confident hallucination
    threshold_low: float = 0.3   # Below this = confident supported
    agreement_threshold: float = 0.5  # Below this = escalate due to component disagreement


@dataclass
class AggregatedScore:
    """Result of score aggregation across transformer and augmentations."""

    hallucination_score: float  # 0.0 = supported, 1.0 = hallucinated
    agreement: float  # Ensemble agreement (0.0-1.0), high = components agree
    confident: bool  # True = return result, False = consider escalation
    escalate: bool  # True = send to Stage 2
    component_scores: dict  # Individual scores from each component
    evidence: dict  # Aggregated factual metadata from all components
    merged_spans: list  # Union of flagged spans from all components
    routing_reason: str  # Human-readable explanation


class ScoreAggregator:
    """Aggregates transformer and augmentation scores.

    All scores use unified direction: 0.0 = supported, 1.0 = hallucinated.
    Transformer and augmentations both return hallucination probability.
    """

    def __init__(self, weights: dict[str, float], config: AggregationConfig | None = None) -> None:
        """Initialize aggregator with weights from Stage1Config.

        Args:
            weights: Weight dict from Stage1Config.weights
                     e.g., {"transformer": 0.5, "ner": 0.2, "numeric": 0.15, "lexical": 0.15}
            config: AggregationConfig with thresholds
        """
        self.weights = weights
        config = config or AggregationConfig()
        self.threshold_high = config.threshold_high
        self.threshold_low = config.threshold_low
        self.agreement_threshold = config.agreement_threshold

    def _extract_transformer_score(self, preds: list[dict]) -> float:
        """Extract hallucination score from transformer predictions.

        Uses max probability of hallucination tokens.
        """
        if not preds:
            return 0.0
        hal_probs = [p["prob"] for p in preds if p.get("pred") == 1]
        return max(hal_probs) if hal_probs else 0.0

    def aggregate(
        self,
        transformer_preds: list[dict],
        aug_results: dict[str, AugmentationResult],
    ) -> AggregatedScore:
        """Weighted average of component scores.

        All scores use unified direction: 0.0 = supported, 1.0 = hallucinated.

        Args:
            transformer_preds: Token predictions from TransformerDetector
            aug_results: Dict of augmentation name -> AugmentationResult

        Returns:
            AggregatedScore with unified hallucination score and routing decision
        """
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # Transformer score (0 = supported, 1 = hallucinated)
        t_score = self._extract_transformer_score(transformer_preds)
        scores["transformer"] = t_score
        t_weight = self.weights.get("transformer", 0.5)
        weighted_sum += t_score * t_weight
        total_weight += t_weight

        # Augmentation scores (already in correct direction: 0 = supported, 1 = hallucinated)
        for name, result in aug_results.items():
            if result.score is None:
                continue
            scores[name] = result.score
            weight = self.weights.get(name, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight

        # Default to 0.5 (neutral/uncertain) if no weights configured
        hallucination_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Compute agreement (ensemble disagreement measure)
        # agreement = 1.0 - min(std * 2, 1.0)
        # All components agree (std=0) → agreement=1.0
        # Components strongly disagree (std=0.5) → agreement=0.0
        all_scores = list(scores.values())
        if len(all_scores) > 1:
            agreement = 1.0 - min(float(np.std(all_scores)) * 2, 1.0)
        else:
            agreement = 1.0  # Single component = full agreement

        # Aggregate evidence from all augmentations
        evidence = self._aggregate_evidence(transformer_preds, aug_results)

        # Routing logic: consider both score AND agreement
        in_confident_zone = (
            hallucination_score >= self.threshold_high  # Confident it's hallucinated
            or hallucination_score <= self.threshold_low  # Confident it's supported
        )
        has_sufficient_agreement = agreement >= self.agreement_threshold

        confident = in_confident_zone and has_sufficient_agreement
        escalate = not confident  # Uncertain or disagreeing components should escalate

        return AggregatedScore(
            hallucination_score=hallucination_score,
            agreement=agreement,
            confident=confident,
            escalate=escalate,
            component_scores=scores,
            evidence=evidence,
            merged_spans=self._merge_spans(transformer_preds, aug_results),
            routing_reason=self._get_routing_reason(
                hallucination_score, confident, escalate, agreement
            ),
        )

    def _aggregate_evidence(
        self,
        transformer_preds: list[dict],
        aug_results: dict[str, AugmentationResult],
    ) -> dict:
        """Combine evidence from all components.

        Args:
            transformer_preds: Token predictions from TransformerDetector
            aug_results: Dict of augmentation name -> AugmentationResult

        Returns:
            Dict with aggregated evidence counts
        """
        evidence = {
            "tokens_analyzed": len(transformer_preds),
            "hallucination_tokens": sum(1 for p in transformer_preds if p.get("pred") == 1),
        }
        for name, result in aug_results.items():
            for key, value in result.evidence.items():
                evidence[f"{name}_{key}"] = value
        return evidence

    def _merge_spans(
        self,
        transformer_preds: list[dict],
        aug_results: dict[str, AugmentationResult],
    ) -> list[dict]:
        """Merge flagged spans from transformer and augmentations.

        Strategy:
        - Collect all spans from transformer predictions (tokens with pred=1)
        - Collect flagged_spans from each augmentation result
        - Deduplicate overlapping spans, keeping highest confidence
        - Return sorted by start position
        """
        all_spans = []

        # Convert transformer token predictions to spans
        for pred in transformer_preds:
            if pred.get("pred") == 1:
                all_spans.append(
                    {
                        "start": pred.get("start", 0),
                        "end": pred.get("end", 0),
                        "text": pred.get("token", ""),
                        "confidence": pred.get("prob", 0.5),
                        "source": "transformer",
                    }
                )

        # Collect augmentation spans
        for name, result in aug_results.items():
            for span in result.flagged_spans:
                span_copy = span.copy()
                span_copy["source"] = name
                all_spans.append(span_copy)

        # Deduplicate overlapping spans (keep highest confidence)
        merged = []
        for span in sorted(
            all_spans, key=lambda s: (s.get("start", 0), -s.get("confidence", 0))
        ):
            overlaps = False
            for existing in merged:
                if self._spans_overlap(span, existing):
                    overlaps = True
                    if span.get("confidence", 0) > existing.get("confidence", 0):
                        merged.remove(existing)
                        merged.append(span)
                    break
            if not overlaps:
                merged.append(span)

        return sorted(merged, key=lambda s: s.get("start", 0))

    def _spans_overlap(self, a: dict, b: dict) -> bool:
        """Check if two spans overlap."""
        return not (
            a.get("end", 0) <= b.get("start", 0) or b.get("end", 0) <= a.get("start", 0)
        )

    def _get_routing_reason(
        self, score: float, confident: bool, escalate: bool, agreement: float
    ) -> str:
        """Generate human-readable routing explanation."""
        if confident:
            if score >= self.threshold_high:
                return f"High confidence hallucination (score={score:.2f}, agreement={agreement:.2f})"
            else:
                return f"High confidence supported (score={score:.2f}, agreement={agreement:.2f})"
        elif escalate:
            if agreement < self.agreement_threshold:
                return f"Components disagree, escalating (score={score:.2f}, agreement={agreement:.2f})"
            return f"Uncertain, escalating to Stage 2 (score={score:.2f}, agreement={agreement:.2f})"
        else:
            return f"Uncertain but below escalation threshold (score={score:.2f}, agreement={agreement:.2f})"
