"""Abstract base class for Stage 1 augmentations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from lettucedetect.cascade.types import AugmentationResult

logger = logging.getLogger(__name__)


class BaseAugmentation(ABC):
    """Abstract base class for Stage 1 augmentations.

    All augmentations must implement:
    - name: Unique identifier for the augmentation
    - score: Compute hallucination score for answer against context

    Score direction: 0.0 = supported (no hallucination), 1.0 = hallucinated
    Evidence: Factual metadata about what was checked (counts, ratios)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this augmentation (used as dict key)."""
        pass

    @abstractmethod
    def score(
        self,
        context: list[str],
        answer: str,
        question: str | None,
        token_predictions: list[dict] | None,
    ) -> AugmentationResult:
        """Score the answer against context.

        Args:
            context: List of context passages
            answer: Generated answer to verify
            question: Optional question that was asked
            token_predictions: Optional transformer token predictions

        Returns:
            AugmentationResult with:
            - score: Hallucination probability (0.0 = supported, 1.0 = hallucinated)
            - evidence: Factual metadata about what was checked
            - details: Component-specific verification details
            - flagged_spans: Specific spans identified as potentially hallucinated
        """
        pass

    def preload(self) -> None:
        """Optional: preload resources for consistent latency."""
        pass

    def safe_score(
        self,
        context: list[str],
        answer: str,
        question: str | None,
        token_predictions: list[dict] | None,
    ) -> AugmentationResult:
        """Wrapper with error handling - graceful degradation on failure."""
        try:
            return self.score(context, answer, question, token_predictions)
        except Exception as e:
            logger.warning(f"Augmentation {self.name} failed: {e}")
            return AugmentationResult(
                score=0.5,  # Neutral score on failure
                evidence={},  # No evidence on failure
                details={"error": str(e)},
                flagged_spans=[],
                is_active=False,  # Error = no valid signal
            )
