"""LettuceDetect: Hallucination detection and generation for RAG systems."""

# Main detection interface
# Core data structures
# Cascade configuration
from lettucedetect.configs import (
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
    CascadeConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
)
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

__version__ = "0.2.0"

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
    # Configs
    "CascadeConfig",
    # Existing
    "HallucinationData",
    "HallucinationDataset",
    "HallucinationDetector",
    "HallucinationGenerator",
    "HallucinationSample",
    "RAGFactChecker",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
]
