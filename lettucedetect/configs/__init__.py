"""Configuration models and presets for cascade detector."""

from lettucedetect.configs.models import (
    CascadeConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
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
    "ACCURATE",
    "BALANCED",
    # Production presets
    "FAST",
    "FAST_CASCADE",
    # Legacy / testing presets
    "FULL_CASCADE",
    "PRESETS",
    "STAGE1_MINIMAL",
    "STAGE2_ONLY",
    "STAGE3_GROUNDING_PROBE",
    "WITH_NLI",
    # Models
    "CascadeConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
]
