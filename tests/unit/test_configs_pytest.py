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
    ACCURATE,
    BALANCED,
    FAST,
    FAST_CASCADE,
    FULL_CASCADE,
    PRESETS,
    STAGE1_MINIMAL,
    WITH_NLI,
)


class TestCascadeConfigDefaults:
    """Test default configuration values."""

    def test_cascade_config_defaults(self):
        """CascadeConfig should have sensible defaults."""
        config = CascadeConfig()
        assert config.stages == [1, 3]
        assert config.strategy == "cascade"
        assert config.blend_alpha == 0.55
        assert config.blend_threshold == 0.40
        assert isinstance(config.stage1, Stage1Config)
        assert isinstance(config.stage2, Stage2Config)
        assert isinstance(config.stage3, Stage3Config)
        assert isinstance(config.routing, RoutingConfig)

    def test_stage1_config_defaults(self):
        """Stage1Config should have correct defaults."""
        config = Stage1Config()
        assert config.model_path == "KRLabsOrg/lettucedect-base-modernbert-en-v1"
        assert config.augmentations == ["lexical", "model2vec"]
        assert config.device == "cuda"
        assert config.max_length == 4096

    def test_stage1_weights_sum(self):
        """Stage1Config weights must sum to 1.0."""
        config = Stage1Config()
        assert abs(sum(config.weights.values()) - 1.0) < 0.01

    def test_stage2_config_defaults(self):
        """Stage2Config should have correct defaults."""
        config = Stage2Config()
        assert config.components == ["nli"]
        assert config.ncs_model == "minishlab/potion-base-32M"
        assert config.weights == {"ncs": 0.0, "nli": 1.0}

    def test_stage3_config_defaults(self):
        """Stage3Config should have correct defaults."""
        config = Stage3Config()
        assert config.method == Stage3Method.GROUNDING_PROBE
        assert config.llm_model == "Qwen/Qwen2.5-3B-Instruct"
        assert config.layer_index == -15
        assert config.token_position == "mean"  # noqa: S105
        assert config.threshold == 0.5
        assert config.probe_repo_id is None
        assert config.probe_filename is None

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
        config = CascadeConfig.model_validate(
            {
                "stages": [1, 2],
                "stage1": {"augmentations": ["ner"]},
            }
        )
        assert config.stages == [1, 2]
        assert config.stage1.augmentations == ["ner"]

    def test_stage1_augmentation_validation(self):
        """Stage1 augmentations must be valid options."""
        config = Stage1Config(augmentations=["ner", "numeric", "lexical", "model2vec"])
        assert config.augmentations == ["ner", "numeric", "lexical", "model2vec"]

        with pytest.raises(ValueError):
            Stage1Config(augmentations=["invalid"])

    def test_blend_strategy_default(self):
        """Strategy defaults to 'cascade'."""
        config = CascadeConfig()
        assert config.strategy == "cascade"

    def test_blend_requires_stage1_and_stage3(self):
        """Blend strategy with missing stages should fail."""
        with pytest.raises(ValueError, match="strategy='blend' requires"):
            CascadeConfig(stages=[1], strategy="blend")

        with pytest.raises(ValueError, match="strategy='blend' requires"):
            CascadeConfig(stages=[3], strategy="blend")

        # Valid: stages=[1, 3] with blend
        config = CascadeConfig(stages=[1, 3], strategy="blend")
        assert config.strategy == "blend"

    def test_blend_alpha_range(self):
        """Blend alpha outside [0,1] should fail."""
        with pytest.raises(ValueError):
            CascadeConfig(stages=[1, 3], strategy="blend", blend_alpha=1.5)
        with pytest.raises(ValueError):
            CascadeConfig(stages=[1, 3], strategy="blend", blend_alpha=-0.1)

    def test_blend_threshold_range(self):
        """Blend threshold outside [0,1] should fail."""
        with pytest.raises(ValueError):
            CascadeConfig(stages=[1, 3], strategy="blend", blend_threshold=1.5)

    def test_extra_fields_ignored(self):
        """Old serialized configs with task_routing should not crash."""
        config = CascadeConfig.model_validate(
            {
                "stages": [1, 3],
                "task_routing": {"qa": [1, 3]},
            }
        )
        assert config.stages == [1, 3]
        assert not hasattr(config, "task_routing")

    def test_stage3_probe_repo_fields(self):
        """Stage3Config should accept probe_repo_id and probe_filename."""
        config = Stage3Config(
            probe_repo_id="MRathmayr/lettucedetect-grounding-probes",
            probe_filename="probe_3b_qwen_pca512.joblib",
        )
        assert config.probe_repo_id == "MRathmayr/lettucedetect-grounding-probes"
        assert config.probe_filename == "probe_3b_qwen_pca512.joblib"


class TestPresets:
    """Test preset configurations."""

    def test_presets_exist(self):
        """All expected presets should exist."""
        assert "fast" in PRESETS
        assert "balanced" in PRESETS
        assert "accurate" in PRESETS
        assert "full_cascade" in PRESETS
        assert "fast_cascade" in PRESETS
        assert "with_nli" in PRESETS
        assert "stage1_minimal" in PRESETS
        assert "stage2_only" in PRESETS
        assert "stage3_grounding_probe" in PRESETS

    def test_fast_preset(self):
        """FAST should have only stage 1 with cascade strategy."""
        assert FAST.stages == [1]
        assert FAST.strategy == "cascade"
        assert FAST.stage1.augmentations == ["lexical", "model2vec"]

    def test_balanced_preset(self):
        """BALANCED should use blend strategy with 3B Qwen probe."""
        assert BALANCED.stages == [1, 3]
        assert BALANCED.strategy == "blend"
        assert BALANCED.blend_alpha == 0.50
        assert BALANCED.blend_threshold == 0.43
        assert BALANCED.stage3.llm_model == "Qwen/Qwen2.5-3B-Instruct"
        assert BALANCED.stage3.probe_filename == "probe_3b_qwen_pca512.joblib"
        assert BALANCED.stage3.layer_index == -15

    def test_accurate_preset(self):
        """ACCURATE should use blend strategy with 14B Qwen probe."""
        assert ACCURATE.stages == [1, 3]
        assert ACCURATE.strategy == "blend"
        assert ACCURATE.blend_alpha == 0.55
        assert ACCURATE.blend_threshold == 0.40
        assert ACCURATE.stage3.llm_model == "Qwen/Qwen2.5-14B-Instruct"
        assert ACCURATE.stage3.probe_filename == "probe_14b_qwen_pca512.joblib"
        assert ACCURATE.stage3.layer_index == -20

    def test_full_cascade_preset(self):
        """FULL_CASCADE should have stages [1, 3] with cascade strategy."""
        assert FULL_CASCADE.stages == [1, 3]
        assert FULL_CASCADE.strategy == "cascade"
        assert FULL_CASCADE.stage1.augmentations == ["lexical", "model2vec"]
        assert FULL_CASCADE.stage3.method == Stage3Method.GROUNDING_PROBE

    def test_fast_cascade_is_fast_alias(self):
        """FAST_CASCADE should be an alias for FAST."""
        assert FAST_CASCADE is FAST

    def test_with_nli_preset(self):
        """WITH_NLI should have all 3 stages."""
        assert WITH_NLI.stages == [1, 2, 3]
        assert WITH_NLI.stage1.augmentations == ["lexical", "model2vec"]

    def test_stage1_minimal_preset(self):
        """STAGE1_MINIMAL should be stage 1 only with no augmentations."""
        assert STAGE1_MINIMAL.stages == [1]
        assert STAGE1_MINIMAL.stage1.augmentations == []


class TestStage3Method:
    """Test Stage3Method enum."""

    def test_stage3_methods(self):
        """Only GROUNDING_PROBE should be available."""
        assert Stage3Method.GROUNDING_PROBE.value == "grounding_probe"
        assert len(Stage3Method) == 1

    def test_stage3_method_from_string(self):
        """Stage3Method can be set from string value."""
        config = Stage3Config(method="grounding_probe")
        assert config.method == Stage3Method.GROUNDING_PROBE
