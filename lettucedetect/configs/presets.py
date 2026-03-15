"""Preset configurations for common cascade use cases."""

from lettucedetect.configs.models import (
    CascadeConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
)

_HF_REPO = "mrathmayr/lettucedetect-grounding-probes"

# --- Production presets ---

# Fast: Stage 1 only (transformer + augmentations, ~19ms)
FAST = CascadeConfig(
    stages=[1],
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
)

# Balanced: Blend Stage 1 (base) + Stage 3 (3B Qwen probe), ~49ms
BALANCED = CascadeConfig(
    stages=[1, 3],
    strategy="blend",
    blend_alpha=0.50,
    blend_threshold=0.43,
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
    stage3=Stage3Config(

        llm_model="Qwen/Qwen2.5-3B-Instruct",
        probe_repo_id=_HF_REPO,
        probe_filename="probe_3b_qwen_pca512.joblib",
        layer_index=-15,
    ),
)

# Accurate: Blend Stage 1 (large) + Stage 3 (14B Qwen probe), ~94ms
ACCURATE = CascadeConfig(
    stages=[1, 3],
    strategy="blend",
    blend_alpha=0.55,
    blend_threshold=0.40,
    stage1=Stage1Config(
        model_path="KRLabsOrg/lettucedect-large-modernbert-en-v1",
        augmentations=["lexical", "model2vec"],
    ),
    stage3=Stage3Config(

        llm_model="Qwen/Qwen2.5-14B-Instruct",
        probe_repo_id=_HF_REPO,
        probe_filename="probe_14b_qwen_pca512.joblib",
        layer_index=-20,
    ),
)

# --- Legacy aliases ---

# Cascade with early-exit routing (Stage 1 + Stage 3)
FULL_CASCADE = CascadeConfig(
    stages=[1, 3],
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
    stage3=Stage3Config(),
)

FAST_CASCADE = FAST

# --- Testing / comparison presets ---

# With NLI: all 3 stages
WITH_NLI = CascadeConfig(
    stages=[1, 2, 3],
    stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
    stage2=Stage2Config(),
    stage3=Stage3Config(),
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

# Stage 3 Grounding Probe only (for standalone probe evaluation)
STAGE3_GROUNDING_PROBE = CascadeConfig(
    stages=[3],
    stage3=Stage3Config(),
)

# All presets dict for easy access
PRESETS = {
    "fast": FAST,
    "balanced": BALANCED,
    "accurate": ACCURATE,
    "full_cascade": FULL_CASCADE,
    "fast_cascade": FAST_CASCADE,
    "with_nli": WITH_NLI,
    "stage1_minimal": STAGE1_MINIMAL,
    "stage2_only": STAGE2_ONLY,
    "stage3_grounding_probe": STAGE3_GROUNDING_PROBE,
}
