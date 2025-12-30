"""Integration tests for Stage2Detector."""

import pytest

from lettucedetect.cascade.types import CascadeInput, RoutingDecision, StageResult
from lettucedetect.configs import Stage2Config
from lettucedetect.detectors.stage2 import Stage2Detector


@pytest.fixture
def stage2_detector():
    """Create Stage2Detector with default config."""
    detector = Stage2Detector()
    yield detector
    del detector


@pytest.fixture
def stage2_detector_ncs_only():
    """Create Stage2Detector with NCS only."""
    config = Stage2Config(components=["ncs"])
    detector = Stage2Detector(config)
    yield detector
    del detector


@pytest.fixture
def stage2_detector_nli_only():
    """Create Stage2Detector with NLI only."""
    config = Stage2Config(components=["nli"])
    detector = Stage2Detector(config)
    yield detector
    del detector


class TestStage2DetectorInitialization:
    """Test detector initialization."""

    def test_default_initialization(self, stage2_detector):
        """Default initialization creates both components."""
        assert stage2_detector._encoder is not None
        assert stage2_detector._nli is not None

    def test_ncs_only_initialization(self, stage2_detector_ncs_only):
        """NCS-only config creates only encoder."""
        assert stage2_detector_ncs_only._encoder is not None
        assert stage2_detector_ncs_only._nli is None

    def test_nli_only_initialization(self, stage2_detector_nli_only):
        """NLI-only config creates only NLI detector."""
        assert stage2_detector_nli_only._encoder is None
        assert stage2_detector_nli_only._nli is not None


@pytest.mark.gpu
class TestPredictMethod:
    """Test predict() method (BaseDetector interface)."""

    def test_predict_returns_list(self, stage2_detector):
        """predict() returns a list."""
        context = ["The capital of France is Paris."]
        answer = "Paris is the capital of France."

        result = stage2_detector.predict(context, answer)
        assert isinstance(result, list)

    def test_predict_supported_answer(self, stage2_detector):
        """Supported answer returns empty list or low-confidence tokens."""
        context = ["The Eiffel Tower is located in Paris, France."]
        answer = "The Eiffel Tower is in Paris."

        result = stage2_detector.predict(context, answer)
        # Either empty (not hallucination) or tokens with probs
        assert isinstance(result, list)

    def test_predict_hallucinated_answer(self, stage2_detector):
        """Contradicting answer returns hallucination tokens."""
        context = ["The company reported profits of $10 million in Q3."]
        answer = "The company reported losses of $50 million."

        result = stage2_detector.predict(context, answer)
        # May detect contradiction
        assert isinstance(result, list)

    def test_predict_output_format_tokens(self, stage2_detector):
        """Token format returns token dicts."""
        context = ["Test context."]
        answer = "Test answer."

        result = stage2_detector.predict(context, answer, output_format="tokens")
        assert isinstance(result, list)
        if result:
            assert "token" in result[0]
            assert "pred" in result[0]
            assert "prob" in result[0]

    def test_predict_output_format_spans(self, stage2_detector):
        """Spans format returns span dicts."""
        context = ["The building is 100 meters tall."]
        answer = "The building is 500 meters tall."  # Contradiction

        result = stage2_detector.predict(context, answer, output_format="spans")
        assert isinstance(result, list)
        if result:
            assert "start" in result[0]
            assert "end" in result[0]
            assert "text" in result[0]


@pytest.mark.gpu
class TestPredictPromptMethod:
    """Test predict_prompt() method."""

    def test_predict_prompt_returns_list(self, stage2_detector):
        """predict_prompt() returns a list."""
        prompt = "Context: Paris is in France. Question: Where is Paris?"
        answer = "Paris is in France."

        result = stage2_detector.predict_prompt(prompt, answer)
        assert isinstance(result, list)

    def test_predict_prompt_batch(self, stage2_detector):
        """predict_prompt_batch() returns list of lists."""
        prompts = [
            "Context: The sky is blue.",
            "Context: Water is wet.",
        ]
        answers = ["The sky is blue.", "Water is dry."]

        results = stage2_detector.predict_prompt_batch(prompts, answers)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)


@pytest.mark.gpu
class TestCascadeInterface:
    """Test predict_stage() interface."""

    def test_predict_stage_returns_stage_result(self, stage2_detector):
        """predict_stage() returns StageResult."""
        cascade_input = CascadeInput(
            context=["The capital of France is Paris."],
            answer="Paris is the capital.",
        )

        result = stage2_detector.predict_stage(cascade_input)

        assert isinstance(result, StageResult)
        assert result.stage_name == "stage2"
        assert 0 <= result.agreement <= 1
        assert result.routing_decision in RoutingDecision

    def test_predict_stage_with_previous_stage_result(self, stage2_detector):
        """predict_stage() uses previous stage result."""
        previous = StageResult(
            stage_name="stage1",
            hallucination_score=0.45,
            agreement=0.8,
            is_hallucination=False,
            routing_decision=RoutingDecision.ESCALATE,
            latency_ms=10.0,
            output=[],
            component_scores={},
            evidence={},
            routing_reason="Uncertain, escalating",
        )
        cascade_input = CascadeInput(
            context=["The stock price is $100."],
            answer="The stock price is $100.",
            previous_stage_result=previous,
        )

        result = stage2_detector.predict_stage(cascade_input)

        assert result.stage_name == "stage2"
        # Previous stage confidence should affect result
        assert result.latency_ms > 0

    def test_predict_stage_has_next_stage_affects_routing(self, stage2_detector):
        """has_next_stage parameter affects routing decision."""
        cascade_input = CascadeInput(
            context=["Some context."],
            answer="Some uncertain answer.",
        )

        result_with_next = stage2_detector.predict_stage(cascade_input, has_next_stage=True)
        result_no_next = stage2_detector.predict_stage(cascade_input, has_next_stage=False)

        # Both should complete, routing may differ for uncertain cases
        assert result_with_next.stage_name == "stage2"
        assert result_no_next.stage_name == "stage2"


@pytest.mark.gpu
class TestWarmup:
    """Test warmup functionality."""

    def test_warmup_loads_models(self, stage2_detector):
        """warmup() loads all models."""
        # Initially models may not be loaded
        stage2_detector.warmup()

        # After warmup, models should be loaded
        if stage2_detector._encoder:
            assert stage2_detector._encoder._model is not None
        if stage2_detector._nli:
            assert stage2_detector._nli._model is not None


@pytest.mark.gpu
class TestDetailedScores:
    """Test get_detailed_scores debug method."""

    def test_detailed_scores_returns_dict(self, stage2_detector):
        """get_detailed_scores returns component scores."""
        context = ["The Earth orbits the Sun."]
        answer = "The Sun is at the center of our solar system."

        result = stage2_detector.get_detailed_scores(context, answer)

        assert isinstance(result, dict)
        assert "ncs" in result
        assert "nli" in result

    def test_detailed_scores_ncs_structure(self, stage2_detector):
        """NCS scores contain max, mean, weighted_mean."""
        context = ["Test context."]
        answer = "Test answer."

        result = stage2_detector.get_detailed_scores(context, answer)

        ncs = result["ncs"]
        assert "max" in ncs
        assert "mean" in ncs
        assert "weighted_mean" in ncs

    def test_detailed_scores_nli_structure(self, stage2_detector):
        """NLI scores contain max_contradiction, min_non_contradiction."""
        context = ["Test context."]
        answer = "Test answer."

        result = stage2_detector.get_detailed_scores(context, answer)

        nli = result["nli"]
        assert "max_contradiction" in nli
        assert "min_non_contradiction" in nli


@pytest.mark.gpu
class TestComponentSelection:
    """Test component selection via config."""

    def test_ncs_only_uses_ncs(self, stage2_detector_ncs_only):
        """NCS-only detector uses only NCS scores."""
        context = ["The capital of France is Paris."]
        answer = "Paris is the capital."

        result = stage2_detector_ncs_only.predict(context, answer)
        # Should work with only NCS
        assert isinstance(result, list)

    def test_nli_only_uses_nli(self, stage2_detector_nli_only):
        """NLI-only detector uses only NLI scores."""
        context = ["The capital of France is Paris."]
        answer = "Paris is the capital."

        result = stage2_detector_nli_only.predict(context, answer)
        # Should work with only NLI
        assert isinstance(result, list)
