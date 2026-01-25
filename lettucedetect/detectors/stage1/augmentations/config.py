"""Configuration dataclasses for Stage 1 augmentations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class NERConfig:
    """Configuration for NER verification augmentation.

    Supports two NER backends:
    - spacy: Traditional NER using spaCy models (default)
    - gliner: Zero-shot NER using GLiNER models (higher recall, GPU recommended)

    GLiNER is particularly good for domain-specific entities but requires GPU
    for reasonable latency. Consider sequential model loading to avoid OOM.
    """

    model: Literal["spacy", "gliner"] = "spacy"

    # spaCy settings
    spacy_model: str = "en_core_web_sm"

    # GLiNER settings
    gliner_model: str = "urchade/gliner_small-v2.1"
    gliner_threshold: float = 0.5  # Confidence threshold for GLiNER predictions

    # Common settings
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

    # GLiNER-specific entity labels (lowercase, more natural language)
    gliner_entity_labels: list[str] = field(
        default_factory=lambda: [
            "person",
            "organization",
            "location",
            "date",
            "money",
            "product",
            "event",
        ]
    )

    fuzzy_threshold: float = 0.85
    use_gpu: bool = True

    # Sentence-level scoring
    sentence_level: bool = True  # Score each sentence separately, use max
    sentence_aggregation: Literal["max", "mean"] = "mean"  # How to aggregate sentence scores


@dataclass
class NumericConfig:
    """Configuration for numeric validation augmentation.

    Per-type tolerances allow fine-grained control:
    - Percentages: exact match (critical for statistics)
    - Floats: 5% tolerance (measurement variations)
    - Prices/currency: 1% tolerance (small rounding)
    - Integers: exact match
    """

    tolerance_percent: float = 1.0  # Default tolerance for unspecified types
    tolerance_float: float = 5.0  # Tolerance for float comparisons
    tolerance_currency: float = 1.0  # Tolerance for currency comparisons
    tolerance_percentage: float = 0.0  # Exact match for percentages
    extract_currencies: bool = True
    extract_dates: bool = True
    normalize_word_numbers: bool = True  # Use word2number for text-to-digit
