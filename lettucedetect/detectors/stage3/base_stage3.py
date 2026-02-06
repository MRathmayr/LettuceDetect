"""Abstract base class for Stage 3 detectors.

Stage 3 operates at the response level (like Stage 2) and is always the final
cascade stage. Routing is RETURN_CONFIDENT or RETURN_UNCERTAIN (never ESCALATE).
"""

from __future__ import annotations

import time
from abc import abstractmethod

from lettucedetect.cascade.types import CascadeInput, RoutingDecision, StageResult
from lettucedetect.detectors.base import BaseDetector


class Stage3Detector(BaseDetector):
    """Abstract base for Stage 3 uncertainty quantification detectors.

    Subclasses must implement predict_uncertainty() which returns a StageResult.
    The base class provides predict(), predict_prompt(), predict_prompt_batch(),
    and predict_stage() by delegating to predict_uncertainty().
    """

    @abstractmethod
    def predict_uncertainty(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
    ) -> StageResult:
        """Compute uncertainty-based hallucination detection.

        Args:
            context: List of context passages.
            answer: Model-generated answer to inspect.
            question: Optional question that was asked.

        Returns:
            StageResult with hallucination score and routing decision.
        """
        pass

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        result = self.predict_uncertainty(context, answer, question)
        if output_format == "spans":
            return self._format_spans(
                result.is_hallucination, result.hallucination_score, answer
            )
        return self._format_tokens(
            result.is_hallucination, result.hallucination_score, answer
        )

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens"
    ) -> list:
        return self.predict([prompt], answer, question=None, output_format=output_format)

    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "tokens"
    ) -> list:
        return [
            self.predict_prompt(p, a, output_format) for p, a in zip(prompts, answers)
        ]

    def predict_stage(
        self,
        input: CascadeInput,
        output_format: str = "tokens",
        has_next_stage: bool = True,
    ) -> StageResult:
        start = time.perf_counter()

        result = self.predict_uncertainty(input.context, input.answer, input.question)

        latency_ms = (time.perf_counter() - start) * 1000
        result.latency_ms = latency_ms

        # Stage 3 is always final - never escalate
        if output_format == "spans":
            result.output = self._format_spans(
                result.is_hallucination, result.hallucination_score, input.answer
            )
        else:
            result.output = self._format_tokens(
                result.is_hallucination, result.hallucination_score, input.answer
            )

        return result

    def _format_tokens(
        self, is_hallucination: bool, score: float, answer: str
    ) -> list[dict]:
        """Response-level: all tokens get same score if hallucinated."""
        if not is_hallucination:
            return []
        return [{"token": t, "pred": 1, "prob": score} for t in answer.split()]

    def _format_spans(
        self, is_hallucination: bool, score: float, answer: str
    ) -> list[dict]:
        """Single span covering entire answer if hallucinated."""
        if not is_hallucination:
            return []
        return [
            {"start": 0, "end": len(answer), "text": answer, "confidence": score}
        ]
