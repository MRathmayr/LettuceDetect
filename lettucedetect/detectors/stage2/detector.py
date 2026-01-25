"""Stage 2 Detector: Embedding-based semantic analysis.

Stage 2 operates at the **response level**, not token level. It computes semantic
similarity (NCS) and contradiction detection (NLI) for the entire answer against
the context passages. This differs from Stage 1's token-level predictions.

When outputting in "tokens" format, Stage 2 synthesizes token-level predictions
by marking ALL tokens with the same hallucination score if the response is
classified as hallucinated. For true token-level predictions, use Stage 1.
"""

from __future__ import annotations

import logging
import time

from lettucedetect.cascade.types import CascadeInput, StageResult
from lettucedetect.configs import Stage2Config
from lettucedetect.detectors.base import BaseDetector
from lettucedetect.detectors.stage2.aggregator import (
    AggregatorConfig,
    Stage2Aggregator,
    Stage2Scores,
)
from lettucedetect.detectors.stage2.config import Model2VecConfig
from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder
from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Detect available device (CUDA GPU or CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Stage2Detector(BaseDetector):
    """Stage 2 of the cascade: semantic embedding and NLI analysis.

    Components:
    - NCS (Model2Vec embeddings): Fast semantic similarity via static embeddings
    - NLI (HHEM): Hallucination detection via Vectara's HHEM model

    HHEM is trained specifically for RAG hallucination detection, unlike generic
    NLI models like DeBERTa-MNLI. It outperforms GPT-3.5 and GPT-4 on benchmarks.

    This stage operates at RESPONSE LEVEL, not token level. The hallucination
    score represents the probability that the entire answer is unsupported by
    the context, based on semantic similarity and contradiction signals.

    Target latency: <30ms on GPU, degraded performance on CPU.
    """

    def __init__(self, config: Stage2Config | None = None):
        """Initialize Stage2Detector with optional configuration.

        Args:
            config: Stage2Config with components, model paths, and weights.
                   If None, uses default configuration.

        Note:
            GPU is recommended for optimal latency. On CPU-only systems,
            Stage 2 will still function but with degraded performance.
        """
        self.config = config or Stage2Config()
        components = self.config.components

        # Detect device and warn if CPU-only
        self._device = _detect_device()
        if self._device == "cpu":
            logger.warning(
                "GPU not available. Stage 2 running on CPU with degraded latency. "
                "For production use, GPU is recommended."
            )

        # Initialize Model2Vec encoder if NCS enabled
        self._encoder = None
        if "ncs" in components:
            m2v_config = Model2VecConfig(
                model_name=self.config.ncs_model,
                normalize_embeddings=self.config.ncs_normalize_embeddings,
            )
            self._encoder = Model2VecEncoder(m2v_config)

        # Initialize NLI detector if enabled (uses HHEM)
        self._nli = None
        if "nli" in components:
            self._nli = NLIContradictionDetector(device=self._device)

        # Initialize aggregator with weights and thresholds from config
        agg_config = AggregatorConfig(
            threshold_high=self.config.routing_threshold_high,
            threshold_low=self.config.routing_threshold_low,
            use_stage1_score=self.config.use_stage1_score,
            stage1_weight=self.config.stage1_weight,
        )
        self._aggregator = Stage2Aggregator(self.config.weights, agg_config)

    def warmup(self) -> None:
        """Preload all models to avoid cold-start latency."""
        dummy_context = ["This is a test context."]
        dummy_answer = "This is a test answer."

        if self._encoder:
            self._encoder.encode(dummy_context + [dummy_answer])
        if self._nli:
            self._nli.warmup()

    def _compute_scores(
        self,
        context: list[str],
        answer: str,
        stage1_agreement: float | None = None,
    ) -> tuple[Stage2Scores, float, bool]:
        """Core score computation - shared by predict() and predict_stage().

        Args:
            context: List of context passages.
            answer: The answer to check for hallucination.
            stage1_agreement: Optional Stage 1 agreement for combined routing.

        Returns:
            Tuple of (Stage2Scores, hallucination_score, is_hallucination)
        """
        # Compute NCS (if enabled)
        ncs_score = 0.5
        if self._encoder:
            ncs_scores = self._encoder.compute_ncs(context, answer)
            # NCS is cosine similarity in [-1, 1], convert to [0, 1] support score
            # -1 (opposite) -> 0, 0 (orthogonal) -> 0.5, 1 (identical) -> 1
            ncs_score = (ncs_scores["max"] + 1.0) / 2.0

        # Compute NLI (if enabled)
        nli_score = 0.5
        if self._nli:
            nli_scores = self._nli.compute_context_nli(context, answer)
            # Use weighted combined hallucination score, then invert for support convention
            # hallucination_score: high = hallucination, we need high = support
            nli_score = 1.0 - nli_scores["hallucination_score"]

        # Build scores object and aggregate
        scores = Stage2Scores(
            ncs_score=ncs_score,
            nli_score=nli_score,
            stage1_agreement=stage1_agreement,
        )
        hallucination_score, is_hallucination = self._aggregator.aggregate(scores)

        return scores, hallucination_score, is_hallucination

    def predict(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        output_format: str = "tokens",
    ) -> list:
        """Main prediction method - implements BaseDetector interface.

        Args:
            context: List of context passages.
            answer: The answer to check for hallucination.
            question: Optional question (not used in Stage 2).
            output_format: "tokens" or "spans"

        Returns:
            List of token predictions or spans.
        """
        _, hallucination_score, is_hallucination = self._compute_scores(context, answer)

        # Format output
        if output_format == "spans":
            return self._format_spans(is_hallucination, hallucination_score, answer)
        return self._format_tokens(is_hallucination, hallucination_score, answer)

    def predict_prompt(
        self, prompt: str, answer: str, output_format: str = "tokens"
    ) -> list:
        """Predict with pre-formatted prompt - implements BaseDetector interface.

        Args:
            prompt: The full prompt string.
            answer: The answer to check.
            output_format: "tokens" or "spans"

        Returns:
            List of predictions.
        """
        context = self._extract_context_from_prompt(prompt)
        return self.predict(context, answer, question=None, output_format=output_format)

    def predict_prompt_batch(
        self, prompts: list[str], answers: list[str], output_format: str = "tokens"
    ) -> list:
        """Batch prediction with prompts - implements BaseDetector interface.

        Args:
            prompts: List of prompt strings.
            answers: List of answers.
            output_format: "tokens" or "spans"

        Returns:
            List of prediction lists.
        """
        return [
            self.predict_prompt(p, a, output_format) for p, a in zip(prompts, answers)
        ]

    def predict_stage(
        self,
        input: CascadeInput,
        output_format: str = "tokens",
        has_next_stage: bool = True,
    ) -> StageResult:
        """Cascade prediction returning StageResult with routing decision.

        Args:
            input: CascadeInput with context, answer, and optional previous result.
            output_format: "tokens" or "spans"
            has_next_stage: Whether Stage 3 is available.

        Returns:
            StageResult with hallucination_score, confidence, routing decision,
            component scores, and output predictions.
        """
        return self._predict_cascade(input, output_format, has_next_stage)

    def _predict_cascade(
        self,
        cascade_input: CascadeInput,
        output_format: str = "tokens",
        has_next_stage: bool = True,
    ) -> StageResult:
        """Internal cascade prediction method.

        Args:
            cascade_input: CascadeInput with context and previous stage result.
            output_format: "tokens" or "spans"
            has_next_stage: Whether Stage 3 is available.

        Returns:
            StageResult with full cascade data.
        """
        start = time.perf_counter()

        # Use Stage 1 agreement if available
        stage1_agreement = None
        if cascade_input.previous_stage_result:
            stage1_agreement = cascade_input.previous_stage_result.agreement

        # Use shared method for score computation
        scores, hallucination_score, is_hallucination = self._compute_scores(
            cascade_input.context, cascade_input.answer, stage1_agreement
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # Format output
        if output_format == "spans":
            output = self._format_spans(
                is_hallucination, hallucination_score, cascade_input.answer
            )
        else:
            output = self._format_tokens(
                is_hallucination, hallucination_score, cascade_input.answer
            )

        return self._aggregator.create_stage_result(
            scores=scores,
            hallucination_score=hallucination_score,
            is_hallucination=is_hallucination,
            latency_ms=latency_ms,
            output=output,
            has_next_stage=has_next_stage,
        )

    def get_detailed_scores(self, context: list[str], answer: str) -> dict:
        """Debug method returning all component scores.

        Args:
            context: List of context passages.
            answer: The answer to analyze.

        Returns:
            Dict with NCS and NLI component scores.
        """
        return {
            "ncs": self._encoder.compute_ncs(context, answer) if self._encoder else {},
            "nli": self._nli.compute_context_nli(context, answer) if self._nli else {},
        }

    def _extract_context_from_prompt(self, prompt: str) -> list[str]:
        """Extract context passages from a formatted RAG prompt.

        Args:
            prompt: Full prompt string.

        Returns:
            List of context passages.
        """
        return [prompt]

    def _format_tokens(
        self, is_hallucination: bool, confidence: float, answer: str
    ) -> list[dict]:
        """Format as token predictions (synthesized from response-level).

        Stage 2 operates at response level. If hallucination detected,
        all tokens are marked as hallucinated.

        Args:
            is_hallucination: Whether the answer is hallucinated.
            confidence: Confidence score.
            answer: The answer text.

        Returns:
            List of token dicts or empty list.
        """
        if not is_hallucination:
            return []
        return [{"token": t, "pred": 1, "prob": confidence} for t in answer.split()]

    def _format_spans(
        self, is_hallucination: bool, confidence: float, answer: str
    ) -> list[dict]:
        """Format as spans (response-level = single span if hallucinated).

        Args:
            is_hallucination: Whether the answer is hallucinated.
            confidence: Confidence score.
            answer: The answer text.

        Returns:
            List with single span or empty list.
        """
        if not is_hallucination:
            return []
        return [
            {"start": 0, "end": len(answer), "text": answer, "confidence": confidence}
        ]
