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
    """Configuration for Stage 1: Transformer + Augmentations."""

    model_path: str = "KRLabsOrg/lettucedect-base-modernbert-en-v1"
    augmentations: list[Literal["ner", "numeric", "lexical"]] = []
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "transformer": 0.5,
            "ner": 0.2,
            "numeric": 0.15,
            "lexical": 0.15,
        }
    )
    max_length: int = 4096
    device: str = "cuda"
    lang: str = "en"


class Stage2Config(BaseModel):
    """Configuration for Stage 2: NCS + NLI + Lexical."""

    components: list[Literal["ncs", "nli", "lexical"]] = ["ncs", "nli", "lexical"]
    ncs_model: str = "minishlab/potion-base-32M"
    nli_model: str = "microsoft/deberta-v3-base-mnli"
    weights: dict[str, float] = Field(
        default_factory=lambda: {"ncs": 0.4, "nli": 0.4, "lexical": 0.2}
    )


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
