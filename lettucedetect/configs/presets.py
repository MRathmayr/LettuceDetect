"""Preset configurations for common cascade use cases."""

from lettucedetect.configs.models import (
    CascadeConfig,
    RoutingConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    Stage3Method,
)

# Full 3-stage cascade
FULL_CASCADE = CascadeConfig(
    stages=[1, 2, 3],
    stage1=Stage1Config(augmentations=["ner", "numeric", "lexical"]),
    stage2=Stage2Config(),
    stage3=Stage3Config(method=Stage3Method.SEPS),
    routing=RoutingConfig(threshold_1to2=0.7, threshold_2to3=0.7),
)

# Fast 2-stage cascade (no LLM calls)
FAST_CASCADE = CascadeConfig(
    stages=[1, 2],
    stage1=Stage1Config(augmentations=["ner", "numeric", "lexical"]),
    stage2=Stage2Config(),
    routing=RoutingConfig(threshold_1to2=0.7),
)

# Stage 1 only with augmentations
STAGE1_AUGMENTED = CascadeConfig(
    stages=[1],
    stage1=Stage1Config(augmentations=["ner", "numeric", "lexical"]),
)

# Stage 1 only, no augmentations (equivalent to legacy transformer)
STAGE1_MINIMAL = CascadeConfig(
    stages=[1],
    stage1=Stage1Config(augmentations=[]),
)

# Stage 2 only (for comparison testing)
STAGE2_ONLY = CascadeConfig(
    stages=[2],
    stage2=Stage2Config(),
)

# Stage 3 SEPs only
STAGE3_SEPS = CascadeConfig(
    stages=[3],
    stage3=Stage3Config(method=Stage3Method.SEPS),
)

# Stage 3 Self-Consistency
STAGE3_SELF_CONSISTENCY = CascadeConfig(
    stages=[3],
    stage3=Stage3Config(
        method=Stage3Method.SELF_CONSISTENCY,
        num_samples=5,
    ),
)

# All presets dict for easy access
PRESETS = {
    "full_cascade": FULL_CASCADE,
    "fast_cascade": FAST_CASCADE,
    "stage1_augmented": STAGE1_AUGMENTED,
    "stage1_minimal": STAGE1_MINIMAL,
    "stage2_only": STAGE2_ONLY,
    "stage3_seps": STAGE3_SEPS,
    "stage3_self_consistency": STAGE3_SELF_CONSISTENCY,
}
