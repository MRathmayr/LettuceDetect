"""Hidden state extraction from causal LMs for probe-based detection.

Extracts hidden states at a specified layer and pools over response tokens
using the configured token position strategy. The prompt format must match
what was used during probe training to avoid distribution shift.

Prompt format (from hallu-training/py/model_utils.py):
  With context:
    "Context:\n{context}\n\nQuestion: {question}\nResponse:"
  Without context:
    "Question: {question}\nResponse:"

Response is tokenized separately with " " prefix and add_special_tokens=False,
then concatenated with the prompt tokens. This preserves the exact prompt
boundary for accurate response token extraction.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 4096
PROMPT_FALLBACK_LEN = 3800


class HiddenStateExtractor:
    """Extracts hidden states from a causal LM at a given layer.

    Token position strategies (validated by hyperparameter sweeps):
    - "mean": Mean pooling over response tokens - best performance (+0.085 AUROC vs SLT)
    - "slt": Second-Last Token (per original SEPs paper)
    - "tbg": Token Before Generation (worst performer)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        layer_index: int = -15,
        token_position: Literal["slt", "tbg", "mean"] = "mean",
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._layer_index = layer_index
        self._token_position = token_position

    def _build_prompt(
        self,
        question: str | None = None,
        context: list[str] | None = None,
    ) -> str:
        """Build prompt matching hallu-training pipeline format exactly.

        Must match hallu-training/py/model_utils.py build_prompt().
        Returns prompt WITHOUT the answer (ends at "Response:").
        """
        parts = []
        if context:
            context_str = "\n".join(context)
            parts.append(f"Context:\n{context_str}\n")
        parts.append(f"Question: {question}")
        parts.append("Response:")
        return "\n".join(parts)

    def extract(
        self,
        answer: str,
        question: str | None = None,
        context: list[str] | None = None,
    ) -> torch.Tensor:
        """Extract hidden state vector for the response tokens.

        Args:
            answer: The model-generated response text.
            question: The question that was asked. Required -- the probe was
                trained on question-response prompts and cannot produce meaningful
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
                "on question-response pairs and cannot produce meaningful hidden "
                "states without one."
            )

        # Tokenize prompt and response separately (matches hallu-training pipeline)
        prompt = self._build_prompt(question, context)
        prompt_ids = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        response_ids = self._tokenizer(" " + answer, return_tensors="pt", add_special_tokens=False)["input_ids"]

        prompt_len = prompt_ids.shape[1]
        max_response_len = MAX_SEQ_LEN - prompt_len

        if max_response_len <= 0:
            # Prompt alone exceeds max_length -- truncate prompt, keep some response
            prompt_ids = prompt_ids[:, :PROMPT_FALLBACK_LEN]
            prompt_len = prompt_ids.shape[1]
            max_response_len = MAX_SEQ_LEN - prompt_len

        # Truncate response from the right
        response_ids = response_ids[:, :max_response_len]

        # Concatenate
        input_ids = torch.cat([prompt_ids, response_ids], dim=1).to(self._model.device)
        attention_mask = torch.ones_like(input_ids)

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

        # Pool over response tokens
        response_states = hidden_states[prompt_len:]  # (response_len, hidden_dim)

        if response_states.shape[0] == 0:
            logger.warning("No response tokens found, using last token of prompt")
            return hidden_states[-1]

        return self._pool(response_states, hidden_states, prompt_len)

    def _pool(self, response_states: torch.Tensor, all_states: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Pool hidden states based on token position strategy.

        Args:
            response_states: Response token hidden states, shape (response_len, hidden_dim).
            all_states: Full sequence hidden states, shape (seq_len, hidden_dim).
            prompt_len: Number of prompt tokens (for tbg position).

        Returns:
            1D tensor of shape (hidden_dim,).
        """
        if self._token_position == "mean":
            return response_states.mean(dim=0)
        elif self._token_position == "slt":
            # Second-Last Token of response
            idx = -2 if response_states.shape[0] >= 2 else -1
            return response_states[idx]
        elif self._token_position == "tbg":
            # Token Before Generation = last prompt token (before response starts)
            return all_states[prompt_len - 1]
        else:
            raise ValueError(f"Unknown token position: {self._token_position}")
