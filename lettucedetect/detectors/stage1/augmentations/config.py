"""Configuration dataclasses for Stage 1 augmentations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NERConfig:
    """Configuration for NER verification augmentation."""

    spacy_model: str = "en_core_web_sm"
    entity_types: list[str] = field(
        default_factory=lambda: [
            "PERSON",
            "ORG",
            "GPE",
            "LOC",
            "DATE",
            "TIME",
            "MONEY",
            "PERCENT",
        ]
    )
    fuzzy_threshold: float = 0.85


@dataclass
class NumericConfig:
    """Configuration for numeric validation augmentation."""

    tolerance_percent: float = 1.0
    extract_currencies: bool = True
    extract_dates: bool = True
