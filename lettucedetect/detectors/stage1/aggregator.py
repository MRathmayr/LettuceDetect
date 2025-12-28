"""Score aggregation and routing logic for Stage 1."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lettucedetect.cascade.types import AugmentationResult

logger = logging.getLogger(__name__)


@dataclass
class AggregationConfig:
    """Configuration for score aggregation and routing decisions."""

    confidence_threshold_high: float = 0.7
    confidence_threshold_low: float = 0.4


@dataclass
class AggregatedScore:
    """Result of score aggregation across transformer and augmentations."""

    hallucination_score: float  # 0.0 = supported, 1.0 = hallucinated
    confidence: float  # How confident aggregator is in this score
    confident: bool  # True = return result, False = consider escalation
    escalate: bool  # True = send to Stage 2
    component_scores: dict  # Individual scores from each component
    merged_spans: list  # Union of flagged spans from all components
    routing_reason: str  # Human-readable explanation


class ScoreAggregator:
    """Aggregates transformer and augmentation scores with proper normalization.

    Score directions:
    - Transformer: high = hallucination (correct direction)
    - Augmentations: high = supported (needs inversion)
    - Output: high = hallucination (unified direction)
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
        self.confidence_threshold_high = config.confidence_threshold_high
        self.confidence_threshold_low = config.confidence_threshold_low

    def _normalize_augmentation_score(self, aug_score: float) -> float:
        """Convert augmentation score (high=supported) to hallucination score (high=hallucinated)."""
        return 1.0 - aug_score

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
        """Weighted average with proper score direction normalization.

        Args:
            transformer_preds: Token predictions from TransformerDetector
            aug_results: Dict of augmentation name -> AugmentationResult

        Returns:
            AggregatedScore with unified hallucination score and routing decision
        """
        scores = {}
        total_weight = 0.0
        weighted_sum = 0.0

        # Transformer score (already in correct direction)
        t_score = self._extract_transformer_score(transformer_preds)
        scores["transformer"] = t_score
        t_weight = self.weights.get("transformer", 0.5)
        weighted_sum += t_score * t_weight
        total_weight += t_weight

        # Augmentation scores (need inversion)
        for name, result in aug_results.items():
            if result.score is None:
                continue
            normalized = self._normalize_augmentation_score(result.score)
            scores[name] = normalized
            weight = self.weights.get(name, 0.1)
            weighted_sum += normalized * weight
            total_weight += weight

        # Default to 0.5 (neutral/uncertain) if no weights configured
        hallucination_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Routing logic
        confident = (
            hallucination_score >= self.confidence_threshold_high
            or hallucination_score <= (1 - self.confidence_threshold_high)
        )
        escalate = not confident and hallucination_score > self.confidence_threshold_low

        # Calculate confidence as distance from uncertainty (0.5)
        confidence = abs(hallucination_score - 0.5) * 2

        return AggregatedScore(
            hallucination_score=hallucination_score,
            confidence=confidence,
            confident=confident,
            escalate=escalate,
            component_scores=scores,
            merged_spans=self._merge_spans(transformer_preds, aug_results),
            routing_reason=self._get_routing_reason(hallucination_score, confident, escalate),
        )

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

    def _get_routing_reason(self, score: float, confident: bool, escalate: bool) -> str:
        """Generate human-readable routing explanation."""
        if confident:
            if score >= self.confidence_threshold_high:
                return f"High confidence hallucination (score={score:.2f})"
            else:
                return f"High confidence supported (score={score:.2f})"
        elif escalate:
            return f"Uncertain, escalating to Stage 2 (score={score:.2f})"
        else:
            return f"Uncertain but below escalation threshold (score={score:.2f})"
