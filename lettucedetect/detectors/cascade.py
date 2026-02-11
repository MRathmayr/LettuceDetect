"""Cascade detector with latency-tiered stages."""

from __future__ import annotations

import logging
import time

from lettucedetect.cascade.types import CascadeInput, RoutingDecision, StageResult
from lettucedetect.configs import CascadeConfig, Stage3Method
from lettucedetect.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class CascadeDetector(BaseDetector):
    """Cascade detector that routes through multiple stages.

    Implements the latency-tiered cascade architecture:
    - Stage 1: Fast detection (<30ms)
    - Stage 2: Embedding analysis (<30ms)
    - Stage 3: Uncertainty quantification (<100ms)

    Routing decisions are made based on confidence thresholds.
    """

    def __init__(self, config: CascadeConfig) -> None:
        """Initialize cascade detector with config."""
        self.config = config
        self._stages: dict = {}
        self._classifier = None
        if config.task_routing:
            from lettucedetect.detectors.task_classifier import TaskTypeClassifier

            self._classifier = TaskTypeClassifier()
        self._initialize_stages()

    def _initialize_stages(self) -> None:
        """Initialize detector for each configured stage."""
        for stage_num in self.config.stages:
            if stage_num == 1:
                self._stages[1] = self._init_stage1()
            elif stage_num == 2:
                self._stages[2] = self._init_stage2()
            elif stage_num == 3:
                self._stages[3] = self._init_stage3()

    def _init_stage1(self):
        """Initialize Stage 1 detector."""
        from lettucedetect.detectors.stage1 import Stage1Detector

        return Stage1Detector(config=self.config.stage1)

    def _init_stage2(self):
        """Initialize Stage 2 detector."""
        from lettucedetect.detectors.stage2 import Stage2Detector

        return Stage2Detector(config=self.config.stage2)

    def _init_stage3(self):
        """Initialize Stage 3 detector based on configured method."""
        if self.config.stage3.method == Stage3Method.READING_PROBE:
            from lettucedetect.detectors.stage3.reading_probe_detector import (
                ReadingProbeDetector,
            )

            return ReadingProbeDetector(
                model_name_or_path=self.config.stage3.llm_model,
                probe_path=self.config.stage3.probe_path,
                layer_index=self.config.stage3.layer_index,
                token_position=self.config.stage3.token_position,
                threshold=self.config.stage3.threshold,
            )
        elif self.config.stage3.method == Stage3Method.SEMANTIC_ENTROPY:
            raise NotImplementedError(
                "Semantic entropy detector not yet implemented (offline baseline only)"
            )
        else:
            raise ValueError(f"Unknown Stage 3 method: {self.config.stage3.method}")

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        task_type: str | None = None,
        output_format: str = "tokens",
    ) -> list | dict:
        """Main prediction - routes through cascade stages.

        Args:
            context: List of context passages.
            answer: The answer to check for hallucination.
            question: Optional question (used for task-type classification).
            task_type: Explicit task type ('qa', 'summarization', 'data2txt', 'unknown').
                If None and task_routing is configured, auto-detects from question/context.
            output_format: 'tokens', 'spans', or 'detailed'.
        """
        start_time = time.perf_counter()

        # Auto-detect task type if routing is configured and type not provided
        if self.config.task_routing and task_type is None and self._classifier is not None:
            task_type = self._classifier.classify(question, context)
            logger.debug("Auto-detected task_type=%s", task_type)

        # Determine active stages for this input
        if self.config.task_routing and task_type:
            active_stages = self.config.task_routing.get(task_type, [1])
        else:
            active_stages = self.config.stages

        cascade_input = CascadeInput(
            context=context,
            answer=answer,
            question=question,
            task_type=task_type,
        )

        stage_results = []
        final_result = None

        for stage_num in sorted(active_stages):
            stage = self._stages.get(stage_num)
            if stage is None:
                continue

            has_next_stage = stage_num < max(active_stages)
            result = self._run_stage(
                stage, stage_num, cascade_input, has_next_stage, output_format
            )
            stage_results.append(result)

            if result.routing_decision == RoutingDecision.RETURN_CONFIDENT:
                final_result = result
                break
            elif result.routing_decision == RoutingDecision.RETURN_UNCERTAIN:
                final_result = result
                break
            else:
                cascade_input.previous_stage_result = result

        if final_result is None and stage_results:
            final_result = stage_results[-1]

        total_latency = (time.perf_counter() - start_time) * 1000

        if output_format == "detailed":
            return self._format_detailed(
                final_result, stage_results, total_latency, task_type
            )
        elif output_format == "spans":
            return final_result.output if final_result else []
        else:
            return final_result.output if final_result else []

    def _run_stage(
        self,
        stage,
        stage_num: int,
        cascade_input: CascadeInput,
        has_next_stage: bool,
        output_format: str = "tokens",
    ) -> StageResult:
        """Run a single stage and return result.

        Args:
            stage: The stage detector instance (implements predict_stage)
            stage_num: Stage number (1, 2, or 3)
            cascade_input: Input data for the stage
            has_next_stage: Whether there's a subsequent stage (affects routing)
            output_format: Output format ("tokens" or "spans")

        Returns:
            StageResult with hallucination_score, agreement, routing decision,
            component scores, evidence, and output predictions.
        """
        stage_start = time.perf_counter()

        # Run the stage detector via predict_stage()
        result = stage.predict_stage(
            input=cascade_input,
            output_format=output_format,
            has_next_stage=has_next_stage,
        )

        # Calculate latency
        latency_ms = (time.perf_counter() - stage_start) * 1000
        result.latency_ms = latency_ms

        logger.debug(
            f"Stage {stage_num} completed in {latency_ms:.2f}ms, "
            f"agreement={result.agreement:.3f}, "
            f"routing={result.routing_decision.value}"
        )

        return result

    def _format_detailed(
        self,
        final_result: StageResult | None,
        stage_results: list[StageResult],
        total_latency: float,
        task_type: str | None = None,
    ) -> dict:
        """Format detailed output with routing info."""
        if final_result is None:
            return {"spans": [], "routing": {}, "scores": {}}

        return {
            "spans": final_result.output,
            "routing": {
                "task_type": task_type,
                "resolved_at_stage": int(final_result.stage_name[-1]),
                "stages_executed": [int(r.stage_name[-1]) for r in stage_results],
                "total_latency_ms": total_latency,
                "stage_latencies_ms": {
                    r.stage_name: r.latency_ms for r in stage_results
                },
            },
            "scores": {
                "final_score": final_result.hallucination_score,
                "confident": final_result.routing_decision
                == RoutingDecision.RETURN_CONFIDENT,
                "escalated": final_result.routing_decision == RoutingDecision.ESCALATE,
                "per_stage": {
                    r.stage_name: r.component_scores for r in stage_results
                },
            },
        }

    def predict_prompt(
        self,
        prompt: str,
        answer: str,
        task_type: str | None = None,
        output_format: str = "tokens",
    ) -> list | dict:
        """Predict with pre-formatted prompt - implements BaseDetector interface."""
        return self.predict(
            context=[prompt],
            answer=answer,
            question=None,
            task_type=task_type,
            output_format=output_format,
        )

    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        task_type: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Batch prediction with prompts - implements BaseDetector interface."""
        return [
            self.predict_prompt(p, a, task_type=task_type, output_format=output_format)
            for p, a in zip(prompts, answers)
        ]

    def warmup(self) -> None:
        """Warmup all stage detectors."""
        for stage in self._stages.values():
            if stage is not None and hasattr(stage, "warmup"):
                stage.warmup()
