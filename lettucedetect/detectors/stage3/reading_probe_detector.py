"""Reading Probe Detector: Hidden state probe for hallucination detection.

Uses a trained sklearn LogisticRegression probe on LLM hidden states to predict
P(correct). Low P(correct) indicates hallucination.

Score direction: P(correct) is inverted to hallucination convention:
  hallucination_score = 1.0 - P(correct)
  0.0 = supported (high P(correct)), 1.0 = hallucinated (low P(correct))
"""

from __future__ import annotations

import logging

import numpy as np

from lettucedetect.cascade.types import RoutingDecision, StageResult
from lettucedetect.detectors.stage3.base_stage3 import Stage3Detector
from lettucedetect.detectors.stage3.probes.hidden_state_extractor import (
    HiddenStateExtractor,
)
from lettucedetect.detectors.stage3.probes.reading_probe import ReadingProbe

logger = logging.getLogger(__name__)


class ReadingProbeDetector(Stage3Detector):
    """Stage 3 detector using reading probes on LLM hidden states.

    Loads a causal LM (optionally quantized) and a trained sklearn probe.
    For each input, extracts hidden states and predicts P(correct).

    Requires:
    - GPU with CUDA for quantized models (4-bit/8-bit via bitsandbytes)
    - A trained .joblib probe file from the read-training pipeline
    """

    def __init__(
        self,
        model_name_or_path: str,
        probe_path: str | None,
        layer_index: int = -16,
        token_position: str = "mean",
        threshold: float = 0.5,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        if probe_path is None:
            raise ValueError(
                "probe_path is required for ReadingProbeDetector. "
                "Train a probe with read-training/py/train_reading_probe.py "
                "or download a pre-trained .joblib file."
            )

        self._threshold = threshold
        self._layer_index = layer_index

        # Check GPU availability for quantized models
        if load_in_4bit or load_in_8bit:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA GPU required for quantized model loading (load_in_4bit/load_in_8bit). "
                    "Set load_in_4bit=False and load_in_8bit=False for CPU inference "
                    "(requires sufficient RAM for full-precision model)."
                )

        # Load LLM
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        logger.info(f"Loading LLM: {model_name_or_path} (4bit={load_in_4bit}, 8bit={load_in_8bit})")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load sklearn probe
        logger.info(f"Loading reading probe: {probe_path}")
        self._probe = ReadingProbe.load(probe_path)

        # Hidden state extractor
        self._extractor = HiddenStateExtractor(
            self._model, self._tokenizer, layer_index, token_position
        )

    def predict_uncertainty(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
    ) -> StageResult:
        hidden_state = self._extractor.extract(answer, question, context)
        hidden_np = hidden_state.cpu().float().numpy().reshape(1, -1)

        p_correct = float(self._probe.predict_proba(hidden_np)[0])
        hallucination_score = 1.0 - p_correct
        is_hallucination = p_correct < self._threshold

        # Confidence: how far from the decision boundary (0.5)
        confidence = abs(p_correct - 0.5) * 2.0

        routing = (
            RoutingDecision.RETURN_CONFIDENT
            if confidence >= 0.3
            else RoutingDecision.RETURN_UNCERTAIN
        )

        return StageResult(
            stage_name="stage3",
            hallucination_score=hallucination_score,
            agreement=confidence,
            is_hallucination=is_hallucination,
            routing_decision=routing,
            latency_ms=0.0,  # Set by predict_stage()
            output=[],  # Set by predict_stage()
            component_scores={
                "reading_probe": hallucination_score,
                "p_correct": p_correct,
            },
            evidence={
                "layer_index": self._layer_index,
                "threshold": self._threshold,
                "probe_metadata": self._probe.metadata,
            },
            routing_reason=(
                f"P(correct)={p_correct:.3f}, "
                f"threshold={self._threshold}, "
                f"confidence={confidence:.3f}"
            ),
        )
