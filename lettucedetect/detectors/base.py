"""Abstract base class for hallucination detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lettucedetect.cascade.types import CascadeInput, StageResult


class BaseDetector(ABC):
    """Base class for all hallucination detectors.

    Supports both standalone usage (predict methods) and cascade integration
    (predict_stage method). Stage detectors should override predict_stage()
    for custom routing logic.
    """

    @abstractmethod
    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Predict hallucination tokens or spans given passages and an answer.

        Args:
            context: List of passages that were supplied to the LLM / user.
            answer: Model-generated answer to inspect.
            question: Original question (None for summarisation).
            output_format: Output format, one of:
                - "tokens": Token-level predictions. Returns list of dicts:
                    [{"token": "word", "start": 0, "end": 4, "pred": 1, "prob": 0.85}, ...]
                    pred=1 means hallucinated, pred=0 means supported.
                    prob is confidence in the prediction.

                - "spans": Contiguous hallucinated spans. Returns list of dicts:
                    [{"start": 0, "end": 15, "text": "hallucinated text", "confidence": 0.85}, ...]
                    Only spans detected as hallucinations are returned.
                    Empty list means no hallucinations detected.

        Returns:
            List of predictions in the requested format.

        Example:
            >>> detector = Stage1Detector()
            >>> # Token-level output
            >>> tokens = detector.predict(context, answer, output_format="tokens")
            >>> flagged = [t for t in tokens if t["pred"] == 1]
            >>>
            >>> # Span-level output
            >>> spans = detector.predict(context, answer, output_format="spans")
            >>> if spans:
            ...     print(f"Found {len(spans)} hallucinated spans")
        """
        pass

    @abstractmethod
    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucinations when you already have a single full prompt string.

        Args:
            prompt: Pre-formatted prompt containing context.
            answer: Model-generated answer to inspect.
            output_format: Output format ("tokens" or "spans"). See predict() for details.

        Returns:
            List of predictions in the requested format.
        """
        pass

    @abstractmethod
    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "tokens"
    ) -> list:
        """Batch version of predict_prompt.

        Args:
            prompts: List of pre-formatted prompts.
            answers: List of answers corresponding to each prompt.
            output_format: Output format ("tokens" or "spans"). See predict() for details.

        Returns:
            List of prediction lists, one per prompt-answer pair.
        """
        pass

    def predict_stage(
        self,
        input: "CascadeInput",
        output_format: str = "tokens",
        has_next_stage: bool = True,
    ) -> "StageResult":
        """Cascade prediction returning StageResult with routing decision.

        This method is used by CascadeDetector to run stages. The default
        implementation calls predict() and wraps the result. Stage detectors
        should override this method for custom routing logic and to populate
        component_scores, confidence, etc.

        Args:
            input: CascadeInput with context, answer, and optional previous result.
            output_format: Output format ("tokens" or "spans").
            has_next_stage: Whether there's a subsequent stage (affects routing).

        Returns:
            StageResult with hallucination_score, confidence, routing decision,
            component scores, and output predictions.

        Note:
            Subclasses should override this method to provide proper routing
            logic. The default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement predict_stage() for cascade integration"
        )

    def warmup(self) -> None:
        """Warmup models for consistent latency.

        Call this method before timing-sensitive operations to ensure models
        are loaded and JIT compilation is complete. The default implementation
        does nothing; subclasses should override if warmup is needed.
        """
        pass
