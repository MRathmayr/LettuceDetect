"""Internal configuration dataclasses for Stage 2 components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Model2VecConfig:
    """Configuration for Model2Vec encoder."""

    model_name: str = "minishlab/potion-base-32M"
    normalize_embeddings: bool = True
    batch_size: int = 32


@dataclass
class NLIConfig:
    """Configuration for NLI contradiction detector."""

    model_name: str = "cross-encoder/nli-deberta-v3-base"
    device: str | None = None
    max_length: int = 512
    batch_size: int = 8


@dataclass
class AggregatorConfig:
    """Configuration for score aggregation and routing.

    Thresholds define when we're confident enough to return a result:
    - hallucination_score >= threshold_high → confident it's hallucinated
    - hallucination_score <= threshold_low → confident it's supported
    - agreement < agreement_threshold → escalate even if score is in confident zone
    - Otherwise → uncertain, may escalate to next stage
    """

    threshold_high: float = 0.85  # Above this = confident hallucination
    threshold_low: float = 0.3   # Below this = confident supported
    agreement_threshold: float = 0.5  # Below this = escalate due to component disagreement
    use_stage1_score: bool = True
    stage1_weight: float = 0.3
