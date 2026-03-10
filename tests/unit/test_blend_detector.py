"""Unit tests for BlendDetector with mocked stages."""

from unittest.mock import MagicMock, patch

import pytest

from lettucedetect.cascade.types import CascadeInput, RoutingDecision, StageResult
from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage3Config


def _make_stage_result(stage_name, score, output=None):
    """Create a StageResult for testing."""
    return StageResult(
        stage_name=stage_name,
        hallucination_score=score,
        agreement=1.0,
        is_hallucination=score >= 0.5,
        routing_decision=RoutingDecision.RETURN_CONFIDENT,
        latency_ms=10.0,
        output=output or [],
        component_scores={f"{stage_name}_score": score},
        evidence={},
        routing_reason="test",
    )


def _make_blend_detector(alpha=0.5, threshold=0.5, mock_s1=None, mock_s3=None):
    """Create a BlendDetector with mocked stages."""
    from lettucedetect.detectors.blend import BlendDetector

    config = CascadeConfig(
        stages=[1, 3],
        strategy="blend",
        blend_alpha=alpha,
        blend_threshold=threshold,
    )

    with patch("lettucedetect.detectors.stage1.Stage1Detector") as s1_cls, \
         patch("lettucedetect.detectors.stage3.grounding_probe_detector.GroundingProbeDetector") as s3_cls:
        s1_cls.return_value = mock_s1 or MagicMock()
        s3_cls.return_value = mock_s3 or MagicMock()
        detector = BlendDetector(config=config)

    # Override with mocks directly
    if mock_s1:
        detector._stage1 = mock_s1
    if mock_s3:
        detector._stage3 = mock_s3

    return detector


class TestBlendScoreComputation:
    """Test that blended score = alpha * s1 + (1-alpha) * s3."""

    def test_blend_formula(self):
        """Blended score should follow alpha * s1 + (1-alpha) * s3."""
        mock_s1 = MagicMock()
        mock_s3 = MagicMock()

        s1_tokens = [{"token": "test", "pred": 1, "prob": 0.8}]
        mock_s1.predict_stage.return_value = _make_stage_result("stage1", 0.8, s1_tokens)
        mock_s3.predict_stage.return_value = _make_stage_result("stage3", 0.2)

        detector = _make_blend_detector(alpha=0.6, threshold=0.5, mock_s1=mock_s1, mock_s3=mock_s3)
        result = detector.predict(["ctx"], "test answer", output_format="detailed")

        expected_score = 0.6 * 0.8 + 0.4 * 0.2  # 0.56
        assert abs(result["blending"]["blended_score"] - expected_score) < 1e-6
        assert result["blending"]["is_hallucination"] is True  # 0.56 >= 0.5

    def test_blend_below_threshold(self):
        """Blended score below threshold should return not hallucinated."""
        mock_s1 = MagicMock()
        mock_s3 = MagicMock()

        mock_s1.predict_stage.return_value = _make_stage_result(
            "stage1", 0.3, [{"token": "ok", "pred": 0, "prob": 0.3}]
        )
        mock_s3.predict_stage.return_value = _make_stage_result("stage3", 0.2)

        detector = _make_blend_detector(alpha=0.5, threshold=0.5, mock_s1=mock_s1, mock_s3=mock_s3)
        result = detector.predict(["ctx"], "ok answer")
        assert result == []  # Not hallucinated -> empty output


class TestBlendThresholdBehavior:
    """Test threshold decision boundary."""

    def test_above_threshold_returns_s1_output(self):
        """When hallucinated, return Stage 1's token predictions."""
        mock_s1 = MagicMock()
        mock_s3 = MagicMock()

        s1_tokens = [
            {"token": "Paris", "pred": 1, "prob": 0.9, "start": 0, "end": 5},
            {"token": "is", "pred": 0, "prob": 0.1, "start": 6, "end": 8},
        ]
        mock_s1.predict_stage.return_value = _make_stage_result("stage1", 0.9, s1_tokens)
        mock_s3.predict_stage.return_value = _make_stage_result("stage3", 0.8)

        detector = _make_blend_detector(alpha=0.5, threshold=0.4, mock_s1=mock_s1, mock_s3=mock_s3)
        result = detector.predict(["ctx"], "Paris is great")
        assert result == s1_tokens

    def test_below_threshold_returns_empty(self):
        """When not hallucinated, return empty list."""
        mock_s1 = MagicMock()
        mock_s3 = MagicMock()

        mock_s1.predict_stage.return_value = _make_stage_result(
            "stage1", 0.1, [{"token": "ok", "pred": 0, "prob": 0.1}]
        )
        mock_s3.predict_stage.return_value = _make_stage_result("stage3", 0.1)

        detector = _make_blend_detector(alpha=0.5, threshold=0.5, mock_s1=mock_s1, mock_s3=mock_s3)
        result = detector.predict(["ctx"], "ok answer")
        assert result == []


class TestBothStagesCalled:
    """Test that both stages are always called (no early exit)."""

    def test_both_stages_called(self):
        """Both Stage 1 and Stage 3 should always be called."""
        mock_s1 = MagicMock()
        mock_s3 = MagicMock()

        mock_s1.predict_stage.return_value = _make_stage_result("stage1", 0.99)
        mock_s3.predict_stage.return_value = _make_stage_result("stage3", 0.99)

        detector = _make_blend_detector(mock_s1=mock_s1, mock_s3=mock_s3)
        detector.predict(["ctx"], "answer")

        mock_s1.predict_stage.assert_called_once()
        mock_s3.predict_stage.assert_called_once()


class TestStage3Fallback:
    """Test Stage 3 failure fallback."""

    def test_stage3_failure_falls_back_to_s1(self):
        """If Stage 3 fails, fall back to Stage 1 score only."""
        mock_s1 = MagicMock()
        mock_s3 = MagicMock()

        s1_tokens = [{"token": "bad", "pred": 1, "prob": 0.7}]
        mock_s1.predict_stage.return_value = _make_stage_result("stage1", 0.7, s1_tokens)
        mock_s3.predict_stage.side_effect = RuntimeError("OOM")

        detector = _make_blend_detector(alpha=0.5, threshold=0.5, mock_s1=mock_s1, mock_s3=mock_s3)
        result = detector.predict(["ctx"], "bad answer")
        assert result == s1_tokens  # S1 score 0.7 >= 0.5

    def test_stage3_failure_detailed_shows_degraded(self):
        """Detailed output should show stage3_degraded when S3 fails."""
        mock_s1 = MagicMock()
        mock_s3 = MagicMock()

        mock_s1.predict_stage.return_value = _make_stage_result("stage1", 0.3)
        mock_s3.predict_stage.side_effect = RuntimeError("OOM")

        detector = _make_blend_detector(alpha=0.5, threshold=0.5, mock_s1=mock_s1, mock_s3=mock_s3)
        result = detector.predict(["ctx"], "answer", output_format="detailed")
        assert result["blending"]["stage3_degraded"] is True
        assert result["blending"]["stage3_score"] is None
        assert result["routing"]["stages_executed"] == [1]
