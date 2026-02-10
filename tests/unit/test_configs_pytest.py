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
    STAGE1_MINIMAL,
    TASK_ROUTED,
    WITH_NLI,
    PRESETS,
)


class TestCascadeConfigDefaults:
    """Test default configuration values."""

    def test_cascade_config_defaults(self):
        """CascadeConfig should have sensible defaults."""
        config = CascadeConfig()
        assert config.stages == [1, 3]
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
        assert config.method == Stage3Method.READING_PROBE
        assert config.llm_model == "Qwen/Qwen2.5-3B-Instruct"
        assert config.layer_index == -15
        assert config.token_position == "mean"
        assert config.load_in_4bit is True
        assert config.threshold == 0.5

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
        config = Stage1Config(augmentations=["ner", "numeric", "lexical", "model2vec"])
        assert config.augmentations == ["ner", "numeric", "lexical", "model2vec"]

        with pytest.raises(ValueError):
            Stage1Config(augmentations=["invalid"])

    def test_task_routing_none_by_default(self):
        """task_routing should be None by default (backward compat)."""
        config = CascadeConfig()
        assert config.task_routing is None

    def test_task_routing_valid(self):
        """Valid task_routing should pass validation."""
        config = CascadeConfig(
            stages=[1, 3],
            task_routing={"qa": [1, 3], "summarization": [1]},
        )
        assert config.task_routing == {"qa": [1, 3], "summarization": [1]}

    def test_task_routing_invalid_stage_number(self):
        """task_routing with invalid stage number should fail."""
        with pytest.raises(ValueError, match="Invalid stage 5"):
            CascadeConfig(stages=[1, 3], task_routing={"qa": [1, 5]})

    def test_task_routing_references_uninitialized_stage(self):
        """task_routing referencing a stage not in stages should fail."""
        with pytest.raises(ValueError, match="not in stages"):
            CascadeConfig(stages=[1], task_routing={"qa": [1, 3]})

    def test_task_routing_serialization_roundtrip(self):
        """task_routing should survive JSON serialization roundtrip."""
        config = CascadeConfig(
            stages=[1, 3],
            task_routing={"qa": [1, 3], "data2txt": [1]},
        )
        json_str = config.model_dump_json()
        restored = CascadeConfig.model_validate_json(json_str)
        assert restored.task_routing == config.task_routing


class TestPresets:
    """Test preset configurations."""

    def test_presets_exist(self):
        """All expected presets should exist."""
        assert "full_cascade" in PRESETS
        assert "fast_cascade" in PRESETS
        assert "with_nli" in PRESETS
        assert "stage1_minimal" in PRESETS
        assert "stage2_only" in PRESETS
        assert "stage3_reading_probe" in PRESETS
        assert "task_routed" in PRESETS

    def test_full_cascade_preset(self):
        """FULL_CASCADE should have stages [1, 3] with augmentations."""
        assert FULL_CASCADE.stages == [1, 3]
        assert FULL_CASCADE.stage1.augmentations == ["lexical", "model2vec"]
        assert FULL_CASCADE.stage3.method == Stage3Method.READING_PROBE

    def test_fast_cascade_preset(self):
        """FAST_CASCADE should have only stage 1."""
        assert FAST_CASCADE.stages == [1]
        assert FAST_CASCADE.stage1.augmentations == ["lexical", "model2vec"]

    def test_with_nli_preset(self):
        """WITH_NLI should have all 3 stages."""
        assert WITH_NLI.stages == [1, 2, 3]
        assert WITH_NLI.stage1.augmentations == ["lexical", "model2vec"]

    def test_stage1_minimal_preset(self):
        """STAGE1_MINIMAL should be stage 1 only with no augmentations."""
        assert STAGE1_MINIMAL.stages == [1]
        assert STAGE1_MINIMAL.stage1.augmentations == []

    def test_task_routed_preset(self):
        """TASK_ROUTED should have stages [1, 3] with routing config."""
        assert TASK_ROUTED.stages == [1, 3]
        assert TASK_ROUTED.task_routing == {
            "qa": [1, 3],
            "summarization": [1],
            "data2txt": [1],
        }
        assert TASK_ROUTED.stage3.method == Stage3Method.READING_PROBE


class TestStage3Method:
    """Test Stage3Method enum."""

    def test_stage3_methods(self):
        """All Stage3 methods should be available."""
        assert Stage3Method.READING_PROBE.value == "reading_probe"
        assert Stage3Method.SEMANTIC_ENTROPY.value == "semantic_entropy"

    def test_stage3_method_in_config(self):
        """Stage3Method can be used in Stage3Config."""
        config = Stage3Config(method=Stage3Method.SEMANTIC_ENTROPY)
        assert config.method == Stage3Method.SEMANTIC_ENTROPY

    def test_stage3_method_from_string(self):
        """Stage3Method can be set from string value."""
        config = Stage3Config(method="reading_probe")
        assert config.method == Stage3Method.READING_PROBE
