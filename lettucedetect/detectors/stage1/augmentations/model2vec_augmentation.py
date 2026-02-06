"""Model2Vec NCS as Stage 1 augmentation.

Reuses Stage 2's Model2VecEncoder for fast semantic similarity computation.
Cosine similarity is converted to hallucination score convention:
  cosine [-1, 1] -> support [0, 1] -> hallucination [0, 1] (inverted)
"""

from __future__ import annotations

from lettucedetect.cascade.types import AugmentationResult
from lettucedetect.detectors.stage1.augmentations.base import BaseAugmentation


class Model2VecAugmentation(BaseAugmentation):
    """Model2Vec NCS as Stage 1 augmentation."""

    def __init__(self):
        from lettucedetect.detectors.stage2.config import Model2VecConfig
        from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder

        self._encoder = Model2VecEncoder(Model2VecConfig())

    @property
    def name(self) -> str:
        return "model2vec"

    def preload(self) -> None:
        self._encoder.preload()

    def score(
        self,
        context: list[str],
        answer: str,
        question: str | None,
        token_predictions: list[dict] | None,
    ) -> AugmentationResult:
        ncs = self._encoder.compute_ncs(context, answer)
        # cosine [-1, 1] -> support [0, 1] -> hallucination [0, 1]
        support = (ncs["max"] + 1.0) / 2.0
        return AugmentationResult(
            score=1.0 - support,
            evidence={"ncs_max": ncs["max"], "ncs_mean": ncs["mean"]},
            details=ncs,
            flagged_spans=[],
            is_active=True,
        )
