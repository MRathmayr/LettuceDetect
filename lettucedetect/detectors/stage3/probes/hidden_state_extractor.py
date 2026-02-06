"""Hidden state extraction from causal LMs for probe-based detection.

Extracts hidden states at a specified layer and pools over answer tokens
using the configured token position strategy. The prompt format must match
what was used during probe training to avoid distribution shift.

Prompt format (from read-training/benchmark/evaluate_benchmark.py):
  With context:
    "Context:\n{context}\n\nAnswer in 1-5 words only. Do not explain. Do not add notes or context.\n\nQuestion: {question}\nAnswer:{response}"
  Without context:
    "Answer in 1-5 words only. Do not explain. Do not add notes or context.\n\nQuestion: {question}\nAnswer:{response}"
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class HiddenStateExtractor:
    """Extracts hidden states from a causal LM at a given layer.

    Token position strategies (validated by hyperparameter sweeps):
    - "mean": Mean pooling over answer tokens - best performance (+0.085 AUROC vs SLT)
    - "slt": Second-Last Token (per original SEPs paper)
    - "tbg": Token Before Generation (worst performer)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        layer_index: int = -16,
        token_position: Literal["slt", "tbg", "mean"] = "mean",
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._layer_index = layer_index
        self._token_position = token_position

    def _build_prompt(
        self,
        answer: str,
        question: str | None = None,
        context: list[str] | None = None,
    ) -> str:
        """Build prompt matching training pipeline format exactly.

        Must match read-training/benchmark/evaluate_benchmark.py build_prompt().
        """
        if context:
            context_str = "\n".join(context)
            return (
                f"Context:\n{context_str}\n\n"
                "Answer in 1-5 words only. "
                "Do not explain. Do not add notes or context.\n\n"
                f"Question: {question}\n"
                f"Answer:{answer}"
            )
        else:
            return (
                "Answer in 1-5 words only. "
                "Do not explain. Do not add notes or context.\n\n"
                f"Question: {question}\n"
                f"Answer:{answer}"
            )

    def extract(
        self,
        answer: str,
        question: str | None = None,
        context: list[str] | None = None,
    ) -> torch.Tensor:
        """Extract hidden state vector for the answer tokens.

        Args:
            answer: The model-generated answer text.
            question: The question that was asked. Required -- the probe was
                trained on question-answer prompts and cannot produce meaningful
                hidden states without a question.
            context: Optional list of context passages.

        Returns:
            1D tensor of hidden state features, shape (hidden_dim,).

        Raises:
            ValueError: If question is None (probe requires a question).
        """
        if question is None:
            raise ValueError(
                "ReadingProbeDetector requires a question. The probe was trained "
                "on question-answer pairs and cannot produce meaningful hidden "
                "states without one."
            )
        full_text = self._build_prompt(answer, question, context)

        # Tokenize full prompt+answer
        inputs = self._tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs["attention_mask"].to(self._model.device)

        # Determine where answer tokens start by tokenizing prompt only
        prompt_text = self._build_prompt("", question, context)
        prompt_ids = self._tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # Forward pass
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract hidden states at specified layer
        hidden_states = outputs.hidden_states[self._layer_index]  # (1, seq_len, hidden_dim)
        hidden_states = hidden_states.squeeze(0)  # (seq_len, hidden_dim)

        # Pool over answer tokens
        answer_states = hidden_states[prompt_len:]  # (answer_len, hidden_dim)

        if answer_states.shape[0] == 0:
            logger.warning("No answer tokens found, using last token of prompt")
            return hidden_states[-1]

        return self._pool(answer_states)

    def _pool(self, states: torch.Tensor) -> torch.Tensor:
        """Pool hidden states based on token position strategy.

        Args:
            states: Answer token hidden states, shape (answer_len, hidden_dim).

        Returns:
            1D tensor of shape (hidden_dim,).
        """
        if self._token_position == "mean":
            return states.mean(dim=0)
        elif self._token_position == "slt":
            # Second-Last Token
            idx = -2 if states.shape[0] >= 2 else -1
            return states[idx]
        elif self._token_position == "tbg":
            # Token Before Generation = last answer token
            return states[-1]
        else:
            raise ValueError(f"Unknown token position: {self._token_position}")
