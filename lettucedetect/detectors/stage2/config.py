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
    """Configuration for NLI contradiction detector.

    Default model is FEVER-trained for better fact verification.
    Uses max_contradiction as hallucination score (best AUROC on RAGTruth).
    """

    # FEVER-trained model for fact verification (184M params, ~15-20ms)
    # Trained on: MNLI, FEVER-NLI, ANLI - specifically designed for fact checking
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: str | None = None
    max_length: int = 512
    batch_size: int = 8
    # Score mode: "contradiction" (best) or "weighted" (experimental)
    score_mode: str = "contradiction"
    # Weights for weighted mode (entailment is anti-correlated on RAGTruth)
    entailment_weight: float = 0.0  # Disabled - anti-correlated
    contradiction_weight: float = 1.0  # Full weight on contradiction


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
