"""Hallucination Probe Detector: Hidden state probe for hallucination detection.

Uses a trained sklearn LogisticRegression probe on LLM hidden states to predict
P(hallucinated) directly. Trained on RAGTruth with human hallucination labels.

Score direction: P(hallucinated) is used directly:
  hallucination_score = P(hallucinated)
  0.0 = supported, 1.0 = hallucinated
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
    """Stage 3 detector using hallucination probes on LLM hidden states.

    Loads a causal LM in fp16 and a trained sklearn probe.
    For each input, extracts hidden states and predicts P(hallucinated).

    Requires:
    - GPU with CUDA and sufficient VRAM for fp16 model
    - A trained .joblib probe file from the hallu-training pipeline
    """

    def __init__(
        self,
        model_name_or_path: str,
        probe_path: str | None,
        layer_index: int = -15,
        token_position: str = "mean",
        threshold: float = 0.5,
    ):
        if probe_path is None:
            raise ValueError(
                "probe_path is required for ReadingProbeDetector. "
                "Train a probe with hallu-training/py/train_probe.py "
                "or download a pre-trained .joblib file."
            )

        self._threshold = threshold
        self._layer_index = layer_index

        # Load LLM in fp16
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM: {model_name_or_path} (fp16)")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load sklearn probe
        logger.info(f"Loading hallu probe: {probe_path}")
        self._probe = ReadingProbe.load(probe_path)

        # Warn if probe layer_index doesn't match configured layer_index
        probe_layer = self._probe.metadata.get("layer_index")
        if probe_layer is not None and probe_layer != layer_index:
            logger.warning(
                f"Probe was trained at layer_index={probe_layer} but configured "
                f"with layer_index={layer_index}. This may degrade performance."
            )

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

        hallucination_score = float(self._probe.predict_proba(hidden_np)[0])
        is_hallucination = hallucination_score >= self._threshold

        # Confidence: how far from the decision boundary (0.5)
        confidence = abs(hallucination_score - 0.5) * 2.0

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
                "hallu_probe": hallucination_score,
            },
            evidence={
                "layer_index": self._layer_index,
                "threshold": self._threshold,
                "probe_metadata": self._probe.metadata,
            },
            routing_reason=(
                f"P(hallucinated)={hallucination_score:.3f}, "
                f"threshold={self._threshold}, "
                f"confidence={confidence:.3f}"
            ),
        )
