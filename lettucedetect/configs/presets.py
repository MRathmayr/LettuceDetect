"""Preset configurations for common cascade use cases."""

from lettucedetect.configs.models import (
    CascadeConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    Stage3Method,
)

# Main preset: Stage 1 + Hallu Probes (default)
FULL_CASCADE = CascadeConfig(
    stages=[1, 3],
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
    stage3=Stage3Config(method=Stage3Method.READING_PROBE),
)

# Fast mode: Stage 1 only
FAST_CASCADE = CascadeConfig(
    stages=[1],
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
)

# With NLI: all 3 stages
WITH_NLI = CascadeConfig(
    stages=[1, 2, 3],
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
    stage2=Stage2Config(),
    stage3=Stage3Config(method=Stage3Method.READING_PROBE),
)

# Stage 1 only (no augmentations) - legacy transformer baseline
STAGE1_MINIMAL = CascadeConfig(
    stages=[1],
    stage1=Stage1Config(augmentations=[]),
)

# Stage 2 only (for comparison testing)
STAGE2_ONLY = CascadeConfig(
    stages=[2],
    stage2=Stage2Config(),
)

# Stage 3 Hallu Probe only (for standalone probe evaluation)
STAGE3_READING_PROBE = CascadeConfig(
    stages=[3],
    stage3=Stage3Config(method=Stage3Method.READING_PROBE),
)

# Deprecated: Task-routed cascade. Hallu probes work on all task types,
# so task routing is no longer needed.
TASK_ROUTED = CascadeConfig(
    stages=[1, 3],
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
    stage3=Stage3Config(method=Stage3Method.READING_PROBE),
    task_routing={
        "qa": [1, 3],
        "summarization": [1],
        "data2txt": [1],
    },
)

# All presets dict for easy access
PRESETS = {
    "full_cascade": FULL_CASCADE,
    "fast_cascade": FAST_CASCADE,
    "with_nli": WITH_NLI,
    "stage1_minimal": STAGE1_MINIMAL,
    "stage2_only": STAGE2_ONLY,
    "stage3_reading_probe": STAGE3_READING_PROBE,
    "task_routed": TASK_ROUTED,
}
