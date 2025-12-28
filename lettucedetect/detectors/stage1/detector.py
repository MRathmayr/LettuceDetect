"""Stage 1 Detector: TransformerDetector + Augmentations."""

from __future__ import annotations

import logging
import time

from lettucedetect.cascade.types import (
    BaseStage,
    CascadeInput,
    RoutingDecision,
    StageResult,
)
from lettucedetect.configs.models import Stage1Config
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.stage1.aggregator import (
    AggregatedScore,
    AggregationConfig,
    ScoreAggregator,
)
from lettucedetect.detectors.stage1.augmentations.base import BaseAugmentation

logger = logging.getLogger(__name__)


class Stage1Detector(BaseDetector, BaseStage):
    """Stage 1 of the cascade: fast detection with augmentations.

    Combines TransformerDetector with optional augmentations:
    - NER verification (spaCy)
    - Numeric validation (regex)
    - Lexical overlap (n-gram Jaccard)

    Target latency: <30ms on GPU
    """

    def __init__(
        self,
        config: Stage1Config | None = None,
        augmentations: list[str] | None = None,
        aggregation_config: AggregationConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize Stage 1 detector.

        Args:
            config: Stage1Config from cascade configuration
            augmentations: List of augmentation names (overrides config.augmentations)
            aggregation_config: Optional custom aggregation config
            **kwargs: Additional arguments (model_path overrides config)
        """
        self.config = config or Stage1Config()

        # Augmentations can be passed directly (from factory) or from config
        aug_list = augmentations if augmentations is not None else list(self.config.augmentations)

        # Initialize TransformerDetector as foundation
        from lettucedetect.detectors.transformer import TransformerDetector

        self._transformer = TransformerDetector(
            model_path=kwargs.get("model_path", self.config.model_path),
            max_length=self.config.max_length,
            device=self.config.device,
        )

        # Initialize augmentation modules based on augmentations list
        self._augmentations: list[BaseAugmentation] = []
        if "ner" in aug_list:
            from lettucedetect.detectors.stage1.augmentations.ner_verifier import NERVerifier

            self._augmentations.append(NERVerifier())
        if "numeric" in aug_list:
            from lettucedetect.detectors.stage1.augmentations.numeric_validator import (
                NumericValidator,
            )

            self._augmentations.append(NumericValidator())
        if "lexical" in aug_list:
            from lettucedetect.utils.lexical import LexicalOverlapCalculator

            self._augmentations.append(LexicalOverlapCalculator())

        # Initialize aggregator with weights from config
        agg_config = aggregation_config or AggregationConfig()
        self._aggregator = ScoreAggregator(self.config.weights, agg_config)

    @property
    def stage_name(self) -> str:
        """Return the stage name for cascade identification."""
        return "stage1"

    def _run_augmentations(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        transformer_preds: list[dict] | None = None,
    ) -> dict:
        """Run augmentations only (for testing without transformer).

        Args:
            context: Context passages
            answer: Answer to verify
            question: Optional question
            transformer_preds: Optional transformer predictions for span info

        Returns:
            Dict mapping augmentation names to AugmentationResult
        """
        aug_results = {}
        for aug in self._augmentations:
            result = aug.safe_score(context, answer, question, transformer_preds)
            aug_results[aug.name] = result
        return aug_results

    def warmup(self) -> None:
        """Preload all models to avoid cold-start latency."""
        dummy_context = ["This is a test context."]
        dummy_answer = "This is a test answer."
        self._transformer.predict(dummy_context, dummy_answer)

        for aug in self._augmentations:
            aug.preload()

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Main prediction method - implements BaseDetector interface."""
        # 1. Run TransformerDetector
        transformer_preds = self._transformer.predict(
            context, answer, question, output_format="tokens"
        )

        # 2. Run augmentations (sequential - GIL limits parallel benefit)
        aug_results = {}
        for aug in self._augmentations:
            result = aug.safe_score(context, answer, question, transformer_preds)
            aug_results[aug.name] = result

        # 3. Aggregate scores
        aggregated = self._aggregator.aggregate(transformer_preds, aug_results)

        # 4. Format output
        if output_format == "spans":
            return self._format_spans(aggregated, answer)
        return self._format_tokens(aggregated, transformer_preds)

    def predict_prompt(self, prompt: str, answer: str, output_format: str = "tokens") -> list:
        """Predict with pre-formatted prompt - implements BaseDetector interface."""
        context = self._extract_context_from_prompt(prompt)
        return self.predict(context, answer, question=None, output_format=output_format)

    def predict_prompt_batch(
        self,
        prompts: list[str],
        answers: list[str],
        output_format: str = "tokens",
    ) -> list[list]:
        """Batch prediction with prompts - implements BaseDetector interface."""
        return [self.predict_prompt(p, a, output_format) for p, a in zip(prompts, answers)]

    def get_routing_decision(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
    ) -> dict:
        """Get routing decision without full prediction formatting."""
        transformer_preds = self._transformer.predict(context, answer, question, "tokens")
        aug_results = {}
        for aug in self._augmentations:
            aug_results[aug.name] = aug.safe_score(context, answer, question, transformer_preds)
        aggregated = self._aggregator.aggregate(transformer_preds, aug_results)
        return {
            "confident": aggregated.confident,
            "escalate": aggregated.escalate,
            "score": aggregated.hallucination_score,
            "reason": aggregated.routing_reason,
            "component_scores": aggregated.component_scores,
        }

    def run(
        self,
        input: CascadeInput,
        output_format: str = "tokens",
        has_next_stage: bool = True,
    ) -> StageResult:
        """Run stage - implements BaseStage interface.

        Args:
            input: CascadeInput with context, answer, and optional previous result
            output_format: Output format ("tokens" or "spans")
            has_next_stage: Whether there's a subsequent stage (affects routing)

        Returns:
            StageResult with confidence, prediction, and routing decision
        """
        start = time.perf_counter()

        # Run transformer and augmentations
        transformer_preds = self._transformer.predict(
            input.context, input.answer, input.question, output_format="tokens"
        )
        aug_results = {}
        for aug in self._augmentations:
            aug_results[aug.name] = aug.safe_score(
                input.context, input.answer, input.question, transformer_preds
            )

        # Aggregate scores
        aggregated = self._aggregator.aggregate(transformer_preds, aug_results)

        # Determine routing decision
        if aggregated.confident:
            routing = RoutingDecision.RETURN_CONFIDENT
        elif aggregated.escalate and has_next_stage:
            routing = RoutingDecision.ESCALATE
        else:
            routing = RoutingDecision.RETURN_UNCERTAIN

        latency_ms = (time.perf_counter() - start) * 1000

        # Format output
        if output_format == "spans":
            output = self._format_spans(aggregated, input.answer)
        else:
            output = self._format_tokens(aggregated, transformer_preds)

        return StageResult(
            stage_name="stage1",
            confidence=aggregated.confidence,
            is_hallucination=aggregated.hallucination_score >= 0.5,
            routing_decision=routing,
            latency_ms=latency_ms,
            output_format_result=output,
            metadata={
                "scores": aggregated.component_scores,
                "routing_reason": aggregated.routing_reason,
            },
        )

    def _extract_context_from_prompt(self, prompt: str) -> list[str]:
        """Extract context passages from a formatted RAG prompt."""
        return [prompt]

    def _format_tokens(
        self,
        aggregated: AggregatedScore,
        transformer_preds: list[dict],
    ) -> list[dict]:
        """Format output in tokens format."""
        return transformer_preds

    def _format_spans(self, aggregated: AggregatedScore, answer: str) -> list[dict]:
        """Format output in spans format."""
        return aggregated.merged_spans
