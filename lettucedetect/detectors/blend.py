"""Blend detector: runs Stage 1 + Stage 3 and combines scores."""

from __future__ import annotations

import logging
import time

from lettucedetect.cascade.types import CascadeInput, StageResult
from lettucedetect.configs import CascadeConfig
from lettucedetect.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class BlendDetector(BaseDetector):
    """Detector that blends Stage 1 and Stage 3 scores.

    Unlike CascadeDetector (sequential early-exit routing), BlendDetector
    runs both stages unconditionally and combines their scores:

        blended = alpha * s1_score + (1 - alpha) * s3_score

    The blended score is compared against a threshold to make the final decision.
    Token-level localization comes from Stage 1 when the blended decision is hallucinated.
    """

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self._alpha = config.blend_alpha
        self._threshold = config.blend_threshold

        # Initialize Stage 1
        from lettucedetect.detectors.stage1 import Stage1Detector

        self._stage1 = Stage1Detector(config=config.stage1)

        # Initialize Stage 3
        from lettucedetect.detectors.stage3.grounding_probe_detector import (
            GroundingProbeDetector,
        )

        self._stage3 = GroundingProbeDetector(
            model_name_or_path=config.stage3.llm_model,
            probe_path=config.stage3.probe_path,
            probe_repo_id=config.stage3.probe_repo_id,
            probe_filename=config.stage3.probe_filename,
            layer_index=config.stage3.layer_index,
            token_position=config.stage3.token_position,
            threshold=config.stage3.threshold,
        )

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list | dict:
        """Run both stages and blend scores.

        Args:
            context: List of context passages.
            answer: The answer to check for hallucination.
            question: Optional question.
            output_format: 'tokens', 'spans', or 'detailed'.
        """
        start_time = time.perf_counter()

        cascade_input = CascadeInput(
            context=context,
            answer=answer,
            question=question,
        )

        # Run Stage 1
        s1_start = time.perf_counter()
        s1_result = self._stage1.predict_stage(
            input=cascade_input, output_format=output_format, has_next_stage=False
        )
        s1_result.latency_ms = (time.perf_counter() - s1_start) * 1000

        # Run Stage 3 (with fallback on failure)
        s3_result = self._run_stage3(cascade_input, output_format)

        # Blend scores
        if s3_result is not None:
            blended_score = (
                self._alpha * s1_result.hallucination_score
                + (1 - self._alpha) * s3_result.hallucination_score
            )
        else:
            blended_score = s1_result.hallucination_score

        is_hallucination = blended_score >= self._threshold

        # Build output: S1 predictions when hallucinated, empty when not
        output = s1_result.output if is_hallucination else []

        total_latency = (time.perf_counter() - start_time) * 1000

        if output_format == "detailed":
            return self._format_detailed(
                s1_result, s3_result, blended_score, is_hallucination, output, total_latency
            )

        return output

    def _run_stage3(
        self, cascade_input: CascadeInput, output_format: str
    ) -> StageResult | None:
        """Run Stage 3 with error fallback."""
        try:
            s3_start = time.perf_counter()
            result = self._stage3.predict_stage(
                input=cascade_input, output_format=output_format, has_next_stage=False
            )
            result.latency_ms = (time.perf_counter() - s3_start) * 1000
            return result
        except Exception:
            logger.warning("Stage 3 failed, falling back to Stage 1 only", exc_info=True)
            return None

    def _format_detailed(
        self,
        s1_result: StageResult,
        s3_result: StageResult | None,
        blended_score: float,
        is_hallucination: bool,
        output: list,
        total_latency: float,
    ) -> dict:
        """Format detailed output with blending metadata."""
        s3_degraded = s3_result is None

        return {
            "spans": output,
            "blending": {
                "alpha": self._alpha,
                "threshold": self._threshold,
                "blended_score": blended_score,
                "is_hallucination": is_hallucination,
                "stage1_score": s1_result.hallucination_score,
                "stage3_score": s3_result.hallucination_score if s3_result else None,
                "stage3_degraded": s3_degraded,
            },
            "routing": {
                "stages_executed": [1, 3] if not s3_degraded else [1],
                "total_latency_ms": total_latency,
                "stage_latencies_ms": {
                    "stage1": s1_result.latency_ms,
                    "stage3": s3_result.latency_ms if s3_result else None,
                },
            },
            "scores": {
                "final_score": blended_score,
                "per_stage": {
                    "stage1": s1_result.component_scores,
                    "stage3": s3_result.component_scores if s3_result else {},
                },
            },
        }

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens"
    ) -> list | dict:
        """Predict with pre-formatted prompt."""
        return self.predict(
            context=[prompt], answer=answer, question=None, output_format=output_format
        )

    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "tokens"
    ) -> list:
        """Batch prediction with prompts."""
        return [
            self.predict_prompt(p, a, output_format=output_format)
            for p, a in zip(prompts, answers)
        ]

    def warmup(self) -> None:
        """Warmup both stage detectors."""
        if hasattr(self._stage1, "warmup"):
            self._stage1.warmup()
        if hasattr(self._stage3, "warmup"):
            self._stage3.warmup()
