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
    STAGE1_MINIMAL,
    STAGE2_ONLY,
    STAGE3_READING_PROBE,
    TASK_ROUTED,
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
    # Presets
    "FULL_CASCADE",
    "FAST_CASCADE",
    "WITH_NLI",
    "STAGE1_MINIMAL",
    "STAGE2_ONLY",
    "STAGE3_READING_PROBE",
    "TASK_ROUTED",
    "PRESETS",
]
