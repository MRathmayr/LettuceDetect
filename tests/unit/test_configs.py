"""Unit tests for cascade configuration models."""

import pytest

# Import directly from submodules to avoid heavy imports from main __init__.py
from lettucedetect.configs.models import (
    CascadeConfig,
    RoutingConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    Stage3Method,
)
from lettucedetect.configs.presets import (
    FULL_CASCADE,
    FAST_CASCADE,
    STAGE1_AUGMENTED,
    PRESETS,
)


class TestCascadeConfigDefaults:
    """Test default configuration values."""

    def test_cascade_config_defaults(self):
        """CascadeConfig should have sensible defaults."""
        config = CascadeConfig()
        assert config.stages == [1, 2, 3]
        assert isinstance(config.stage1, Stage1Config)
        assert isinstance(config.stage2, Stage2Config)
        assert isinstance(config.stage3, Stage3Config)
        assert isinstance(config.routing, RoutingConfig)

    def test_stage1_config_defaults(self):
        """Stage1Config should have correct defaults."""
        config = Stage1Config()
        assert config.model_path == "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        assert config.augmentations == []
        assert config.device == "cuda"
        assert config.max_length == 4096

    def test_stage2_config_defaults(self):
        """Stage2Config should have correct defaults."""
        config = Stage2Config()
        assert config.components == ["ncs", "nli", "lexical"]
        assert config.ncs_model == "minishlab/potion-base-32M"
        assert config.nli_model == "microsoft/deberta-v3-base-mnli"

    def test_stage3_config_defaults(self):
        """Stage3Config should have correct defaults."""
        config = Stage3Config()
        assert config.method == Stage3Method.SEPS
        assert config.llm_model == "meta-llama/Llama-3.2-3B"
        assert config.num_samples == 5

    def test_routing_config_defaults(self):
        """RoutingConfig should have correct defaults."""
        config = RoutingConfig()
        assert config.threshold_1to2 == 0.7
        assert config.threshold_2to3 == 0.7


class TestCascadeConfigValidation:
    """Test configuration validation."""

    def test_stages_must_be_ascending(self):
        """Stages must be in ascending order."""
        with pytest.raises(ValueError, match="ascending order"):
            CascadeConfig(stages=[2, 1])

    def test_stages_descending_fails(self):
        """Descending order should fail validation."""
        with pytest.raises(ValueError, match="ascending order"):
            CascadeConfig(stages=[3, 2, 1])

    def test_valid_stage_orders(self):
        """Valid stage orders should work."""
        CascadeConfig(stages=[1])
        CascadeConfig(stages=[1, 2])
        CascadeConfig(stages=[1, 3])
        CascadeConfig(stages=[2, 3])
        CascadeConfig(stages=[1, 2, 3])

    def test_cascade_config_from_dict(self):
        """CascadeConfig should be creatable from dict."""
        config = CascadeConfig.model_validate({
            "stages": [1, 2],
            "stage1": {"augmentations": ["ner"]},
        })
        assert config.stages == [1, 2]
        assert config.stage1.augmentations == ["ner"]

    def test_stage1_augmentation_validation(self):
        """Stage1 augmentations must be valid options."""
        # Valid augmentations
        config = Stage1Config(augmentations=["ner", "numeric", "lexical"])
        assert config.augmentations == ["ner", "numeric", "lexical"]

        # Invalid augmentation should fail
        with pytest.raises(ValueError):
            Stage1Config(augmentations=["invalid"])


class TestPresets:
    """Test preset configurations."""

    def test_presets_exist(self):
        """All expected presets should exist."""
        assert "full_cascade" in PRESETS
        assert "fast_cascade" in PRESETS
        assert "stage1_augmented" in PRESETS
        assert "stage1_minimal" in PRESETS
        assert "stage2_only" in PRESETS
        assert "stage3_seps" in PRESETS
        assert "stage3_self_consistency" in PRESETS

    def test_full_cascade_preset(self):
        """FULL_CASCADE should have 3 stages with augmentations."""
        assert FULL_CASCADE.stages == [1, 2, 3]
        assert FULL_CASCADE.stage1.augmentations == ["ner", "numeric", "lexical"]
        assert FULL_CASCADE.stage3.method == Stage3Method.SEPS

    def test_fast_cascade_preset(self):
        """FAST_CASCADE should have only stages 1 and 2."""
        assert FAST_CASCADE.stages == [1, 2]
        assert FAST_CASCADE.stage1.augmentations == ["ner", "numeric", "lexical"]

    def test_stage1_augmented_preset(self):
        """STAGE1_AUGMENTED should be stage 1 only with all augmentations."""
        assert STAGE1_AUGMENTED.stages == [1]
        assert STAGE1_AUGMENTED.stage1.augmentations == ["ner", "numeric", "lexical"]


class TestStage3Method:
    """Test Stage3Method enum."""

    def test_stage3_methods(self):
        """All Stage3 methods should be available."""
        assert Stage3Method.SEPS.value == "seps"
        assert Stage3Method.SELF_CONSISTENCY.value == "self_consistency"
        assert Stage3Method.SEMANTIC_ENTROPY.value == "semantic_entropy"

    def test_stage3_method_in_config(self):
        """Stage3Method can be used in Stage3Config."""
        config = Stage3Config(method=Stage3Method.SELF_CONSISTENCY)
        assert config.method == Stage3Method.SELF_CONSISTENCY

    def test_stage3_method_from_string(self):
        """Stage3Method can be set from string value."""
        config = Stage3Config(method="self_consistency")
        assert config.method == Stage3Method.SELF_CONSISTENCY
