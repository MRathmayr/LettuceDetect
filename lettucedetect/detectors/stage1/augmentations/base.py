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
    - score: Compute support score for answer against context

    Score direction: 0.0 = hallucination, 1.0 = fully supported
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
            AugmentationResult with score (0=hallucination, 1=supported)
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
                confidence=0.0,
                details={"error": str(e)},
                flagged_spans=[],
            )
