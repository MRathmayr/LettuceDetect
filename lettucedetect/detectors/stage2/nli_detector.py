"""NLI-based hallucination detection using MiniCheck-Flan-T5-Large.

MiniCheck outperforms HHEM on LLM-AggreFact (75% vs 71.8% balanced accuracy).
It uses a seq2seq approach: given (document, claim), generates "Yes" or "No".
We extract probabilities from the logits to get a continuous score.

Model: lytang/MiniCheck-Flan-T5-Large (~0.8B parameters)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class NLIContradictionDetector:
    """NLI-based hallucination detection using MiniCheck-Flan-T5-Large.

    MiniCheck is a seq2seq model that outputs "Yes" (supported) or "No" (not supported)
    for document-claim pairs. We extract probabilities from the output logits.

    Score direction:
    - MiniCheck "Yes" = supported, "No" = not supported
    - We return: 0.0 = supported, 1.0 = hallucinated
    """

    MODEL_NAME = "lytang/MiniCheck-Flan-T5-Large"

    def __init__(self, device: str | None = None):
        """Initialize NLI detector.

        Args:
            device: Device to run model on ("cuda", "cpu", or None for auto-detect).
        """
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._device = device
        self._yes_id: int | None = None
        self._no_id: int | None = None

    def preload(self) -> None:
        """Load MiniCheck model to GPU/CPU."""
        if self._model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info(f"Loading MiniCheck model: {self.MODEL_NAME}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)

        # Move to device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

        # Cache token IDs for Yes/No
        self._yes_id = self._tokenizer.encode("Yes", add_special_tokens=False)[0]
        self._no_id = self._tokenizer.encode("No", add_special_tokens=False)[0]

        logger.info(f"MiniCheck loaded on {self._device}")

    def warmup(self) -> None:
        """Alias for preload() to match augmentation interface."""
        self.preload()

    def _score_single(self, context: str, answer: str) -> float:
        """Score a single context-answer pair.

        Args:
            context: Context/document text.
            answer: Claim/answer to verify.

        Returns:
            Hallucination score in [0, 1] where 0 = supported, 1 = hallucinated.
        """
        # MiniCheck format
        input_text = f"Document: {context}\nClaim: {answer}"
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            # Generate with output scores
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Get logits for first generated token
            logits = outputs.scores[0][0]  # Shape: (vocab_size,)

            # Extract Yes/No logits and compute softmax
            yes_no_logits = torch.tensor(
                [logits[self._yes_id], logits[self._no_id]],
                device=self._device,
            )
            probs = torch.softmax(yes_no_logits, dim=0)

            # prob_supported = probs[0] (Yes), prob_not_supported = probs[1] (No)
            # hallucination_score = prob_not_supported
            return probs[1].item()

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

        hallucination_scores = []

        for ctx in context_texts:
            try:
                score = self._score_single(ctx, answer)
                hallucination_scores.append(score)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.error(f"MiniCheck critical failure: {e}")
                raise
            except Exception as e:
                logger.warning(f"MiniCheck scoring failed for context: {e}")
                hallucination_scores.append(0.5)

        if not hallucination_scores:
            return {
                "hallucination_score": 0.5,
                "max_hallucination": 0.5,
                "mean_hallucination": 0.5,
            }

        max_hal = max(hallucination_scores)
        mean_hal = sum(hallucination_scores) / len(hallucination_scores)

        return {
            "hallucination_score": max_hal,  # Use max as primary
            "max_hallucination": max_hal,
            "mean_hallucination": mean_hal,
        }
