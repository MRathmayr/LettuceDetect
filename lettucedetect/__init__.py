"""LettuceDetect: Hallucination detection and generation for RAG systems."""

# Main detection interface
# Core data structures
from lettucedetect.datasets.hallucination_dataset import (
    HallucinationData,
    HallucinationDataset,
    HallucinationSample,
)

# Generation interface
from lettucedetect.models.generation import HallucinationGenerator
from lettucedetect.models.inference import HallucinationDetector

# Direct RAGFactChecker access for advanced users
from lettucedetect.ragfactchecker import RAGFactChecker

# Cascade configuration
from lettucedetect.configs import (
    ACCURATE,
    BALANCED,
    CascadeConfig,
    FAST,
    FAST_CASCADE,
    FULL_CASCADE,
    PRESETS,
    RoutingConfig,
    STAGE1_MINIMAL,
    STAGE2_ONLY,
    STAGE3_GROUNDING_PROBE,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    Stage3Method,
    WITH_NLI,
)

__version__ = "0.2.0"

__all__ = [
    # Existing
    "HallucinationData",
    "HallucinationDataset",
    "HallucinationDetector",
    "HallucinationGenerator",
    "HallucinationSample",
    "RAGFactChecker",
    # Configs
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
