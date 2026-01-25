"""NLI-based hallucination detection using HHEM (Vectara).

HHEM (Hallucination Evaluation Model) is trained specifically for RAG hallucination
detection, unlike generic NLI models. It outperforms GPT-3.5 and GPT-4 on
hallucination detection benchmarks.

Note: HHEM requires trust_remote_code=True which executes code from HuggingFace.
This is acceptable for research/development. If enterprise security policies
prohibit this, the code can be vendored locally.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


class NLIContradictionDetector:
    """NLI-based hallucination detection using HHEM model.

    HHEM is a cross-encoder trained specifically for detecting hallucinations
    in RAG systems. It takes (context, answer) pairs and returns consistency scores.

    Score direction:
    - HHEM returns: 0.0 = hallucinated, 1.0 = consistent
    - We invert to: 0.0 = supported, 1.0 = hallucinated
    """

    MODEL_NAME = "vectara/hallucination_evaluation_model"

    def __init__(self, device: str | None = None):
        """Initialize NLI detector.

        Args:
            device: Device to run model on ("cuda", "cpu", or None for auto-detect).
        """
        self._model = None
        self._device = device

    def preload(self) -> None:
        """Load HHEM model to GPU/CPU."""
        if self._model is not None:
            return

        logger.info("Loading HHEM model (Vectara hallucination evaluation)")
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
        )

        # Move to device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()
        logger.info(f"HHEM model loaded on {self._device}")

    def warmup(self) -> None:
        """Alias for preload() to match augmentation interface."""
        self.preload()

    def compute_context_nli(self, context_texts: list[str], answer: str) -> dict:
        """Compute hallucination scores for answer against context.

        Args:
            context_texts: List of context passages (premises).
            answer: Generated answer to check (hypothesis).

        Returns:
            Dict with hallucination_score, max_hallucination, mean_hallucination.
            All scores are in range [0, 1] where 0 = supported, 1 = hallucinated.
        """
        if not context_texts:
            return {
                "hallucination_score": 0.5,
                "max_hallucination": 0.5,
                "mean_hallucination": 0.5,
            }

        if self._model is None:
            self.preload()

        # Create premise-hypothesis pairs
        pairs = [(ctx, answer) for ctx in context_texts]

        try:
            # HHEM's predict() handles batching and tokenization internally
            scores = self._model.predict(pairs)

            # Convert to list robustly
            if hasattr(scores, "tolist"):
                scores_list = scores.tolist()
            else:
                scores_list = list(scores)

            # INVERT: HHEM returns 0=hallucinated, 1=consistent
            # We need 0=supported, 1=hallucinated
            hallucination_scores = [1.0 - s for s in scores_list]

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            # Critical errors should propagate
            logger.error(f"HHEM critical failure: {e}")
            raise
        except Exception as e:
            logger.warning(f"HHEM predict() failed: {e}. Returning neutral scores.")
            hallucination_scores = [0.5] * len(pairs)

        if not hallucination_scores:
            return {
                "hallucination_score": 0.5,
                "max_hallucination": 0.5,
                "mean_hallucination": 0.5,
            }

        max_hal = max(hallucination_scores)
        mean_hal = sum(hallucination_scores) / len(hallucination_scores)

        return {
            "hallucination_score": max_hal,  # Use max as primary (consistent with DeBERTa approach)
            "max_hallucination": max_hal,
            "mean_hallucination": mean_hal,
        }
