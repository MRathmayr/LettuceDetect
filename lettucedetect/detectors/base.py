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

        :param context: List of passages that were supplied to the LLM / user.
        :param answer: Model-generated answer to inspect.
        :param question: Original question (``None`` for summarisation).
        :param output_format: ``"tokens"`` for token-level dicts, ``"spans"`` for character spans.
        :returns: List of predictions in requested format.
        """
        pass

    @abstractmethod
    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict hallucinations from a pre-built prompt string.

        :param prompt: Full prompt (context + question already concatenated).
        :param answer: Model-generated answer to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :returns: List of predictions in requested format.
        """
        pass

    @abstractmethod
    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "tokens"
    ) -> list:
        """Batch version of :meth:`predict_prompt`.

        :param prompts: List of full prompt strings.
        :param answers: List of answers to inspect.
        :param output_format: ``"tokens"`` or ``"spans"``.
        :returns: List of prediction lists, one per input pair.
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

        :param input: CascadeInput with context, answer, and optional previous result.
        :param output_format: Output format ("tokens" or "spans").
        :param has_next_stage: Whether there's a subsequent stage (affects routing).
        :returns: StageResult with hallucination_score, confidence, routing decision,
            component scores, and output predictions.
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
