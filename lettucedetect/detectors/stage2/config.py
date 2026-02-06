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
    """Configuration for NLI contradiction detector (MiniCheck).

    MiniCheck-Flan-T5-Large outperforms HHEM on LLM-AggreFact (75% vs 71.8%).
    It's a seq2seq model that outputs Yes/No for document-claim pairs.
    """

    device: str | None = None
    max_length: int = 512  # Max input tokens for context + claim


@dataclass
class AggregatorConfig:
    """Configuration for score aggregation and routing.

    Thresholds define when we're confident enough to return a result:
    - hallucination_score >= threshold_high -> confident it's hallucinated
    - hallucination_score <= threshold_low -> confident it's supported
    - agreement < agreement_threshold -> escalate even if score is in confident zone
    - Otherwise -> uncertain, may escalate to next stage

    Calibrated voting converts raw scores to binary votes using per-component
    optimal thresholds from RAGTruth benchmark, then computes weighted vote.
    """

    threshold_high: float = 0.85  # Above this = confident hallucination
    threshold_low: float = 0.3   # Below this = confident supported
    agreement_threshold: float = 0.5  # Below this = escalate due to component disagreement
    use_stage1_score: bool = True
    stage1_weight: float = 0.3

    # Calibrated voting: convert scores to binary using optimal thresholds
    use_calibrated_voting: bool = True
    optimal_thresholds: dict | None = None  # Per-component optimal thresholds

    def __post_init__(self):
        # Default optimal thresholds from RAGTruth benchmark (run_4)
        # NCS uses hallucination score direction (low similarity = high hallucination)
        if self.optimal_thresholds is None:
            self.optimal_thresholds = {
                "ncs": 0.123,  # model2vec hallucination score threshold
                "nli": 0.472,  # NLI hallucination score threshold
            }
