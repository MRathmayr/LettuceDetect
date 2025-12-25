"""Core types for cascade detector stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class RoutingDecision(Enum):
    """Routing decision after a stage completes."""

    RETURN_CONFIDENT = "return_confident"
    ESCALATE = "escalate"
    RETURN_UNCERTAIN = "return_uncertain"


@dataclass
class StageResult:
    """Result from a single stage in the cascade."""

    stage_name: str
    confidence: float
    is_hallucination: bool
    routing_decision: RoutingDecision
    latency_ms: float
    output_format_result: list[dict]
    metadata: dict


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

    Defined here to avoid circular imports between utils/ and detectors/.
    """

    score: float  # 0.0 = hallucination, 1.0 = supported
    confidence: float  # Confidence in this score
    details: dict  # Component-specific details
    flagged_spans: list[dict]  # Spans flagged by this augmentation


class BaseStage(ABC):
    """Abstract base class for cascade stage detectors.

    Each stage must implement run() to process CascadeInput and return StageResult.
    """

    @abstractmethod
    def run(
        self,
        input: CascadeInput,
        output_format: str = "tokens",
        has_next_stage: bool = True,
    ) -> StageResult:
        """Process input and return stage result with routing decision.

        Args:
            input: CascadeInput with context, answer, and optional previous result
            output_format: Output format ("tokens" or "spans")
            has_next_stage: Whether there's a subsequent stage (affects routing)

        Returns:
            StageResult with confidence, prediction, and routing decision
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Warmup models for consistent latency."""
        pass
