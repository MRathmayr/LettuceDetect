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
    FAST_CASCADE,
    FULL_CASCADE,
    PRESETS,
    STAGE1_AUGMENTED,
    STAGE1_MINIMAL,
    STAGE2_ONLY,
    STAGE3_SELF_CONSISTENCY,
    STAGE3_SEPS,
)

__all__ = [
    # Models
    "CascadeConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "RoutingConfig",
    "Stage3Method",
    # Presets
    "FULL_CASCADE",
    "FAST_CASCADE",
    "STAGE1_AUGMENTED",
    "STAGE1_MINIMAL",
    "STAGE2_ONLY",
    "STAGE3_SEPS",
    "STAGE3_SELF_CONSISTENCY",
    "PRESETS",
]
