"""Pydantic configuration models for cascade detector."""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Stage3Method(str, Enum):
    """Available methods for Stage 3 detection."""

    READING_PROBE = "reading_probe"
    SEMANTIC_ENTROPY = "semantic_entropy"


class Stage1Config(BaseModel):
    """Configuration for Stage 1: Transformer + Augmentations.

    Weights optimized on RAGTruth benchmark (voting analysis AUROC 0.878):
    - transformer: 0.70 (primary signal, trained on RAGTruth)
    - lexical: 0.30 (complementary heuristic)
    - ner/numeric: 0.0 (disabled - hurt AUROC by flipping correct decisions)

    Routing thresholds calibrated on RAGTruth (2026-01-25):
    - confident_high: 0.79 (score above = skip to hallucinated)
    - confident_low: 0.09 (score below = skip to supported)
    - optimal: 0.15 (threshold that maximizes F1)
    """

    model_path: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    augmentations: list[Literal["ner", "numeric", "lexical", "model2vec"]] = ["lexical", "model2vec"]
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "transformer": 0.65,  # Primary signal (trained on RAGTruth)
            "lexical": 0.25,      # Complementary heuristic
            "model2vec": 0.10,    # NCS embedding similarity
            "ner": 0.0,           # Disabled - hurts AUROC (flips correct decisions)
            "numeric": 0.0,       # Disabled - hurts AUROC (flips correct decisions)
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
    """Configuration for Stage 2: NLI semantic analysis.

    Model2Vec (NCS) moved to Stage 1 as augmentation. Stage 2 is now NLI-only.
    Inactive by default (not in default cascade stages=[1, 3]).
    Re-enable by setting stages=[1, 2, 3] in CascadeConfig.

    Routing thresholds calibrated on RAGTruth (2026-01-25):
    - confident_high: 0.44 (score above = hallucinated)
    - confident_low: 0.06 (score below = supported)
    - optimal: 0.18 (threshold that maximizes F1)
    """

    # Component selection (Model2Vec moved to Stage 1)
    components: list[Literal["ncs", "nli"]] = ["nli"]

    # Model selection (HHEM is hardcoded for NLI, only NCS model is configurable)
    ncs_model: str = "minishlab/potion-base-32M"

    # Aggregation weights (must sum to 1.0)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "ncs": 0.0,   # Model2Vec moved to Stage 1
            "nli": 1.0,   # NLI is the only Stage 2 component
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
    """Configuration for Stage 3: Hallu Probe / Semantic Entropy."""

    method: Stage3Method = Stage3Method.READING_PROBE

    # Hallu Probe config
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    probe_path: str | None = None
    layer_index: int = -15
    token_position: Literal["slt", "tbg", "mean"] = "mean"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    threshold: float = 0.5  # P(hallucinated) above this = hallucination

    # Semantic Entropy config (offline baseline, not yet implemented)
    clustering_method: Literal["nli", "embedding"] = "nli"


class RoutingConfig(BaseModel):
    """Configuration for inter-stage routing decisions."""

    threshold_1to2: float = 0.7
    threshold_2to3: float = 0.7


class CascadeConfig(BaseModel):
    """Main configuration for cascade detector."""

    stages: list[Literal[1, 2, 3]] = [1, 3]
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    stage3: Stage3Config = Field(default_factory=Stage3Config)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    task_routing: dict[str, list[int]] | None = None

    @field_validator("stages")
    @classmethod
    def validate_stages_order(cls, v: list[int]) -> list[int]:
        """Ensure stages are in ascending order."""
        if v != sorted(v):
            raise ValueError(f"Stages must be in ascending order, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_task_routing(self) -> CascadeConfig:
        if self.task_routing:
            all_routed_stages: set[int] = set()
            for stages_list in self.task_routing.values():
                for s in stages_list:
                    if s not in (1, 2, 3):
                        raise ValueError(f"Invalid stage {s} in task_routing (must be 1, 2, or 3)")
                    all_routed_stages.add(s)
            missing = all_routed_stages - set(self.stages)
            if missing:
                raise ValueError(
                    f"task_routing references stages {missing} not in stages={self.stages}. "
                    f"Add them to stages so they get initialized."
                )
        return self

    @classmethod
    def from_json(cls, path: str) -> CascadeConfig:
        """Load config from JSON file."""
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
