"""Configuration models and presets for cascade detector."""

from lettucedetect.configs.models import (
    CascadeConfig,
    RoutingConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    Stage3Method,
)
from lettucedetect.configs.presets import (
    ACCURATE,
    BALANCED,
    FAST,
    FAST_CASCADE,
    FULL_CASCADE,
    PRESETS,
    STAGE1_MINIMAL,
    STAGE2_ONLY,
    STAGE3_GROUNDING_PROBE,
    WITH_NLI,
)

__all__ = [
    # Models
    "CascadeConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "RoutingConfig",
    "Stage3Method",
    # Production presets
    "FAST",
    "BALANCED",
    "ACCURATE",
    # Legacy / testing presets
    "FULL_CASCADE",
    "FAST_CASCADE",
    "WITH_NLI",
    "STAGE1_MINIMAL",
    "STAGE2_ONLY",
    "STAGE3_GROUNDING_PROBE",
    "PRESETS",
]
