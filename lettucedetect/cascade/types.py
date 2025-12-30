"""Core types for cascade detector stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RoutingDecision(Enum):
    """Routing decision after a stage completes."""

    RETURN_CONFIDENT = "return_confident"
    ESCALATE = "escalate"
    RETURN_UNCERTAIN = "return_uncertain"


@dataclass
class StageResult:
    """Result from a single stage in the cascade.

    Attributes:
        stage_name: Identifier for the stage (e.g., "stage1", "stage2")
        hallucination_score: Detection score where 0.0 = supported, 1.0 = hallucinated
        agreement: Ensemble agreement (0.0-1.0). High = components agree, low = components disagree.
        is_hallucination: True if hallucination_score >= 0.5
        routing_decision: What to do next (return, escalate, or uncertain)
        latency_ms: Time taken by this stage in milliseconds
        output: Token/span predictions in requested format

        component_scores: Individual scores from each component (transformer, augmentations, etc.)
        evidence: Aggregated factual metadata about what was checked
        routing_reason: Human-readable explanation of routing decision

        degraded: True if any component failed during inference
        component_errors: List of error messages from failed components
    """

    stage_name: str
    hallucination_score: float  # 0.0 = supported, 1.0 = hallucinated
    agreement: float  # Ensemble agreement (0.0-1.0)
    is_hallucination: bool  # True if hallucination_score >= 0.5
    routing_decision: RoutingDecision
    latency_ms: float
    output: list[dict]  # Token/span predictions

    # Explicit fields instead of metadata dict
    component_scores: dict[str, float]
    evidence: dict  # Aggregated evidence from all components
    routing_reason: str

    # Degradation tracking
    degraded: bool = False
    component_errors: list[str] = field(default_factory=list)


@dataclass
class CascadeInput:
    """Input to a cascade stage."""

    context: list[str]
    answer: str
    question: str | None = None
    prompt: str | None = None
    previous_stage_result: StageResult | None = None


@dataclass
class AugmentationResult:
    """Result from a Stage 1 augmentation.

    All scores use unified direction: 0.0 = supported, 1.0 = hallucinated.

    Attributes:
        score: Hallucination probability (0.0 = supported, 1.0 = hallucinated)
        evidence: Factual metadata about what was checked (e.g., entities_checked, numbers_verified)
        details: Component-specific details (e.g., verified entities, matched numbers)
        flagged_spans: Spans flagged by this augmentation as potentially hallucinated
    """

    score: float  # 0.0 = supported, 1.0 = hallucinated
    evidence: dict  # Factual metadata (counts, ratios)
    details: dict  # Component-specific details
    flagged_spans: list[dict]  # Spans flagged by this augmentation
