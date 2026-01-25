"""Internal configuration dataclasses for Stage 2 components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Model2VecConfig:
    """Configuration for Model2Vec encoder.

    Context chunking splits long contexts into overlapping chunks for better
    similarity matching when contexts exceed typical embedding quality limits.

    Variance-aware scoring adds std deviation of chunk similarities as an
    uncertainty signal - high variance suggests inconsistent context support.
    """

    model_name: str = "minishlab/potion-base-32M"
    normalize_embeddings: bool = True
    batch_size: int = 32
    chunk_tokens: int = 350  # Target tokens per chunk
    chunk_overlap: int = 50  # Overlap between chunks
    enable_chunking: bool = True  # Whether to chunk long contexts
    include_variance: bool = True  # Include similarity variance in output


@dataclass
class NLIConfig:
    """Configuration for NLI contradiction detector (HHEM).

    HHEM is the default and only supported NLI model for Stage 2.
    Model name is hardcoded since HHEM requires trust_remote_code=True.
    """

    device: str | None = None
    # HHEM handles tokenization internally, max_length not needed


@dataclass
class AggregatorConfig:
    """Configuration for score aggregation and routing.

    Thresholds define when we're confident enough to return a result:
    - hallucination_score >= threshold_high -> confident it's hallucinated
    - hallucination_score <= threshold_low -> confident it's supported
    - agreement < agreement_threshold -> escalate even if score is in confident zone
    - Otherwise -> uncertain, may escalate to next stage
    """

    threshold_high: float = 0.85  # Above this = confident hallucination
    threshold_low: float = 0.3   # Below this = confident supported
    agreement_threshold: float = 0.5  # Below this = escalate due to component disagreement
    use_stage1_score: bool = True
    stage1_weight: float = 0.3
