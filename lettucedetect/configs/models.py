"""Pydantic configuration models for cascade detector."""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Stage3Method(str, Enum):
    """Available methods for Stage 3 detection."""

    SEPS = "seps"
    SELF_CONSISTENCY = "self_consistency"
    SEMANTIC_ENTROPY = "semantic_entropy"


class Stage1Config(BaseModel):
    """Configuration for Stage 1: Transformer + Augmentations.

    Weights are optimized for hallucination detection:
    - transformer: 0.70 (primary signal, trained on RAGTruth)
    - lexical/ner/numeric: 0.10 each (heuristic augmentations)

    Routing thresholds calibrated on RAGTruth (2026-01-25):
    - confident_high: 0.79 (score above = skip to hallucinated)
    - confident_low: 0.09 (score below = skip to supported)
    - optimal: 0.15 (threshold that maximizes F1)
    """

    model_path: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    augmentations: list[Literal["ner", "numeric", "lexical"]] = []
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "transformer": 0.70,  # Primary signal (trained on RAGTruth)
            "lexical": 0.10,      # Weak heuristic
            "ner": 0.10,          # Conditional (fires when entities present)
            "numeric": 0.10,      # Conditional (fires when numbers present)
        }
    )
    max_length: int = 4096
    device: str = "cuda"
    lang: str = "en"

    # Routing thresholds (calibrated on RAGTruth 2026-01-25)
    routing_threshold_high: float = 0.79  # Score >= this = confident hallucination
    routing_threshold_low: float = 0.09   # Score <= this = confident supported
    classification_threshold: float = 0.15  # Optimal threshold for F1

    @field_validator("weights")
    @classmethod
    def validate_weights_sum(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure weights sum to 1.0 (with small tolerance for float precision)."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return v


class Stage2Config(BaseModel):
    """Configuration for Stage 2: NCS + NLI semantic analysis.

    Uses HHEM (Vectara Hallucination Evaluation Model) for NLI component.
    HHEM is trained specifically for RAG hallucination detection.

    Weights are optimized for hallucination detection:
    - nli (HHEM): 0.70 (strong hallucination-specific signal)
    - ncs: 0.30 (embedding similarity)

    Routing thresholds calibrated on RAGTruth (2026-01-25):
    - confident_high: 0.44 (score above = hallucinated)
    - confident_low: 0.06 (score below = supported)
    - optimal: 0.18 (threshold that maximizes F1)
    """

    # Component selection
    components: list[Literal["ncs", "nli"]] = ["ncs", "nli"]

    # Model selection (HHEM is hardcoded for NLI, only NCS model is configurable)
    ncs_model: str = "minishlab/potion-base-32M"

    # Aggregation weights (must sum to 1.0)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "ncs": 0.30,  # Embedding similarity
            "nli": 0.70,  # HHEM - strong hallucination-specific signal
        }
    )

    # NCS configuration
    ncs_normalize_embeddings: bool = True
    ncs_batch_size: int = 32

    # Routing thresholds (calibrated on RAGTruth 2026-01-25)
    routing_threshold_high: float = 0.44  # Score >= this = confident hallucination
    routing_threshold_low: float = 0.06   # Score <= this = confident supported
    classification_threshold: float = 0.18  # Optimal threshold for F1

    # Stage 1 integration
    use_stage1_score: bool = True
    stage1_weight: float = 0.3

    @field_validator("weights")
    @classmethod
    def validate_weights_sum(cls, v: dict[str, float]) -> dict[str, float]:
        """Ensure weights sum to 1.0 (with small tolerance for float precision)."""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return v


class Stage3Config(BaseModel):
    """Configuration for Stage 3: SEPs / Self-Consistency / Semantic Entropy."""

    method: Stage3Method = Stage3Method.SEPS

    # SEPs config
    llm_model: str = "meta-llama/Llama-3.2-3B"
    probe_path: str | None = None
    layer_index: int = -1
    token_position: Literal["last", "mean"] = "last"
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # Self-Consistency config
    api_model: str = "gpt-4o-mini"
    num_samples: int = 5
    temperature: float = 0.7
    consistency_method: Literal["bertscore", "nli", "hybrid"] = "hybrid"

    # Semantic Entropy config
    clustering_method: Literal["nli", "embedding"] = "nli"


class RoutingConfig(BaseModel):
    """Configuration for inter-stage routing decisions."""

    threshold_1to2: float = 0.7
    threshold_2to3: float = 0.7


class CascadeConfig(BaseModel):
    """Main configuration for cascade detector."""

    stages: list[Literal[1, 2, 3]] = [1, 2, 3]
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    stage3: Stage3Config = Field(default_factory=Stage3Config)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    @field_validator("stages")
    @classmethod
    def validate_stages_order(cls, v: list[int]) -> list[int]:
        """Ensure stages are in ascending order."""
        if v != sorted(v):
            raise ValueError(f"Stages must be in ascending order, got {v}")
        return v

    @classmethod
    def from_json(cls, path: str) -> CascadeConfig:
        """Load config from JSON file."""
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
