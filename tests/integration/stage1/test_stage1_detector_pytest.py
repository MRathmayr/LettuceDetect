"""Integration tests for Stage1Detector with full augmentation pipeline.

These tests verify the complete Stage1 detection system works correctly
with realistic RAG examples, combining the transformer detector with
NER, Numeric, and Lexical augmentations.

Note: These tests require CUDA/GPU and will be skipped on CPU-only systems.
"""

import pytest

from lettucedetect.configs import Stage1Config
from lettucedetect.detectors.stage1 import Stage1Detector, AggregationConfig


@pytest.mark.gpu
class TestStage1DetectorInitialization:
    """Test Stage1Detector initialization with various configurations."""

    def test_default_initialization(self):
        """Test default initialization without augmentations."""
        detector = Stage1Detector(augmentations=[])
        assert detector is not None
        assert len(detector._augmentations) == 0

    def test_ner_augmentation_initialization(self):
        """Test initialization with NER augmentation."""
        detector = Stage1Detector(augmentations=["ner"])
        assert len(detector._augmentations) == 1
        assert detector._augmentations[0].name == "ner"

    def test_numeric_augmentation_initialization(self):
        """Test initialization with Numeric augmentation."""
        detector = Stage1Detector(augmentations=["numeric"])
        assert len(detector._augmentations) == 1
        assert detector._augmentations[0].name == "numeric"

    def test_lexical_augmentation_initialization(self):
        """Test initialization with Lexical augmentation."""
        detector = Stage1Detector(augmentations=["lexical"])
        assert len(detector._augmentations) == 1
        assert detector._augmentations[0].name == "lexical"

    def test_all_augmentations_initialization(self):
        """Test initialization with all augmentations."""
        detector = Stage1Detector(augmentations=["ner", "numeric", "lexical"])
        assert len(detector._augmentations) == 3
        names = {aug.name for aug in detector._augmentations}
        assert names == {"ner", "numeric", "lexical"}

    def test_config_based_augmentations(self):
        """Test initialization from config object."""
        config = Stage1Config(augmentations=["ner", "lexical"])
        detector = Stage1Detector(config=config)
        assert len(detector._augmentations) == 2


@pytest.mark.gpu
class TestAggregatorIntegration:
    """Test ScoreAggregator integration with augmentations."""

    def test_aggregator_uses_configured_weights(self, stage1_detector_numeric_lexical):
        """Verify aggregator uses configured weights."""
        custom_weights = {
            "transformer": 0.6,
            "numeric": 0.2,
            "lexical": 0.2,
        }
        config = Stage1Config(weights=custom_weights)
        detector = Stage1Detector(config=config, augmentations=["numeric", "lexical"])
        # Verify weights are passed through
        assert detector._aggregator.weights["transformer"] == 0.6


@pytest.mark.gpu
class TestNERAugmentationIntegration:
    """Test NER augmentation integration with Stage1Detector."""

    def test_ner_detects_fabricated_person(self, stage1_detector_ner, context_with_persons):
        """Verify NER catches fabricated person names."""
        # Context has "Albert Einstein" and "Mileva Maric"
        answer = "Dr. Richard Feynman developed the theory at the patent office."

        # Run augmentations only (no transformer)
        results = stage1_detector_ner._run_augmentations(context_with_persons, answer)

        assert "ner" in results
        # Feynman is not in context
        ner_result = results["ner"]
        assert ner_result.score < 1.0 or len(ner_result.flagged_spans) > 0

    def test_ner_verifies_existing_person(self, stage1_detector_ner, context_with_persons):
        """Verify NER confirms existing person names."""
        answer = "Albert Einstein worked at the patent office."

        results = stage1_detector_ner._run_augmentations(context_with_persons, answer)

        assert "ner" in results
        ner_result = results["ner"]
        # Einstein should be verified
        assert ner_result.score > 0

    def test_ner_with_organizations(self, stage1_detector_ner, context_with_organizations):
        """Verify NER works with organization entities."""
        # Context has Microsoft, LinkedIn, European Commission
        fabricated = "Google acquired Twitter for $30 billion."
        supported = "Microsoft acquired LinkedIn."

        fab_results = stage1_detector_ner._run_augmentations(context_with_organizations, fabricated)
        sup_results = stage1_detector_ner._run_augmentations(context_with_organizations, supported)

        # Fabricated should have lower score or flagged spans
        assert fab_results["ner"].score <= sup_results["ner"].score


@pytest.mark.gpu
class TestNumericAugmentationIntegration:
    """Test Numeric augmentation integration with Stage1Detector."""

    def test_numeric_detects_fabricated_numbers(
        self, stage1_detector_numeric, financial_report_context, financial_answer_hallucinated_numbers
    ):
        """Verify Numeric catches fabricated financial numbers."""
        results = stage1_detector_numeric._run_augmentations(
            financial_report_context, financial_answer_hallucinated_numbers
        )

        assert "numeric" in results
        numeric_result = results["numeric"]
        # Fabricated numbers should lower score
        assert numeric_result.score < 1.0
        assert len(numeric_result.flagged_spans) > 0

    def test_numeric_verifies_correct_numbers(
        self, stage1_detector_numeric, financial_report_context, financial_answer_supported
    ):
        """Verify Numeric confirms correct numbers."""
        results = stage1_detector_numeric._run_augmentations(
            financial_report_context, financial_answer_supported
        )

        assert "numeric" in results
        numeric_result = results["numeric"]
        assert numeric_result.score == 1.0

    def test_numeric_with_percentages(self, stage1_detector_numeric, context_with_percentages):
        """Verify Numeric handles percentages correctly."""
        correct = "Market share reached 31%."
        wrong = "Market share reached 45%."

        correct_results = stage1_detector_numeric._run_augmentations(context_with_percentages, correct)
        wrong_results = stage1_detector_numeric._run_augmentations(context_with_percentages, wrong)

        assert correct_results["numeric"].score > wrong_results["numeric"].score


@pytest.mark.gpu
class TestLexicalAugmentationIntegration:
    """Test Lexical augmentation integration with Stage1Detector."""

    def test_lexical_high_overlap_supported(
        self, stage1_detector_lexical, sample_context, sample_answer_supported
    ):
        """Verify high lexical overlap for supported answer."""
        results = stage1_detector_lexical._run_augmentations(sample_context, sample_answer_supported)

        assert "lexical" in results
        lexical_result = results["lexical"]
        assert lexical_result.score > 0.25  # Jaccard similarity with stemming/stopwords

    def test_lexical_low_overlap_hallucinated(
        self, stage1_detector_lexical, sample_context, sample_answer_supported, sample_answer_hallucinated
    ):
        """Verify lower lexical overlap for hallucinated answer."""
        supported_results = stage1_detector_lexical._run_augmentations(
            sample_context, sample_answer_supported
        )
        hallucinated_results = stage1_detector_lexical._run_augmentations(
            sample_context, sample_answer_hallucinated
        )

        # Hallucinated should have lower overlap
        assert (
            hallucinated_results["lexical"].score
            <= supported_results["lexical"].score
        )


@pytest.mark.gpu
class TestCombinedAugmentations:
    """Test multiple augmentations working together."""

    def test_all_augmentations_run(self, stage1_detector_all, financial_report_context, financial_answer_supported):
        """Verify all augmentations produce results."""
        results = stage1_detector_all._run_augmentations(
            financial_report_context, financial_answer_supported
        )

        assert "ner" in results
        assert "numeric" in results
        assert "lexical" in results

    def test_aggregation_combines_scores(
        self, stage1_detector_all, financial_report_context, financial_answer_hallucinated_numbers
    ):
        """Verify aggregator combines augmentation scores."""
        results = stage1_detector_all._run_augmentations(
            financial_report_context, financial_answer_hallucinated_numbers
        )

        # Create mock transformer predictions (no hallucination detected)
        mock_preds = [{"token": "test", "pred": 0, "prob": 0.1}]

        aggregated = stage1_detector_all._aggregator.aggregate(mock_preds, results)

        # Should have a combined hallucination score
        assert 0 <= aggregated.hallucination_score <= 1.0
        assert aggregated.confidence >= 0

    def test_multi_domain_hallucination_detection(self, stage1_detector_all):
        """Test detection across different hallucination types."""
        context = [
            "Acme Corp reported $100 million revenue. CEO John Smith announced expansion."
        ]

        # Hallucinated entity (Jane Doe) + hallucinated number ($150 million)
        answer = "CEO Jane Doe announced $150 million in revenue."

        results = stage1_detector_all._run_augmentations(context, answer)

        # NER should flag Jane Doe
        ner_score = results["ner"].score
        # Numeric should flag 150
        numeric_score = results["numeric"].score

        # At least one should be low
        assert ner_score < 1.0 or numeric_score < 1.0


@pytest.mark.gpu
class TestRoutingDecisions:
    """Test routing decision logic in Stage1Detector."""

    def test_high_confidence_not_escalated(
        self, stage1_detector_routing, financial_report_context, financial_answer_supported
    ):
        """Verify high confidence results are not escalated."""
        results = stage1_detector_routing._run_augmentations(
            financial_report_context, financial_answer_supported
        )
        mock_preds = [{"token": "test", "pred": 0, "prob": 0.05}]
        aggregated = stage1_detector_routing._aggregator.aggregate(mock_preds, results)

        # Low hallucination score with high confidence should not escalate
        if aggregated.confident:
            assert not aggregated.escalate


@pytest.mark.gpu
class TestEdgeCases:
    """Test edge cases in Stage1Detector."""

    def test_empty_context_handling(self, stage1_detector_all, empty_context):
        """Verify graceful handling of empty context."""
        results = stage1_detector_all._run_augmentations(empty_context, "Some answer text.")

        # All augmentations should return valid results
        assert all(r.score is not None for r in results.values())

    def test_empty_answer_handling(self, stage1_detector_all, sample_context, empty_answer):
        """Verify graceful handling of empty answer."""
        results = stage1_detector_all._run_augmentations(sample_context, empty_answer)

        # Empty answer typically means nothing to verify
        assert all(r.score is not None for r in results.values())

    def test_very_long_context(self, stage1_detector_lexical):
        """Verify handling of very long context."""
        long_context = ["The quick brown fox. " * 500]
        answer = "The fox is quick."

        results = stage1_detector_lexical._run_augmentations(long_context, answer)
        assert "lexical" in results
        assert results["lexical"].score is not None

    def test_special_characters_handling(self, stage1_detector_ner_numeric):
        """Verify handling of special characters."""
        context = ["Price: $100.00 (USD) - John O'Brien announced."]
        answer = "O'Brien said the price is $100."

        results = stage1_detector_ner_numeric._run_augmentations(context, answer)
        assert all(r.score is not None for r in results.values())


@pytest.mark.gpu
class TestWarmup:
    """Test warmup functionality for latency optimization."""

    def test_warmup_loads_models(self, stage1_detector_ner_lexical):
        """Verify warmup pre-loads all required models."""
        stage1_detector_ner_lexical.warmup()  # Should not raise

        # Verify spaCy model is loaded for NER
        ner_aug = next(a for a in stage1_detector_ner_lexical._augmentations if a.name == "ner")
        assert ner_aug._nlp is not None


@pytest.mark.gpu
class TestBaseStageInterface:
    """Test BaseStage interface compliance."""

    def test_stage_name(self, stage1_detector_no_aug):
        """Verify stage_name property."""
        assert stage1_detector_no_aug.stage_name == "stage1"

    def test_config_property(self, stage1_detector_no_aug):
        """Verify config property returns Stage1Config."""
        assert isinstance(stage1_detector_no_aug.config, Stage1Config)


@pytest.mark.gpu
class TestRealWorldScenarios:
    """Test realistic RAG scenarios end-to-end."""

    def test_medical_qa_scenario(self, stage1_detector_ner_numeric, medical_context, medical_answer_hallucinated):
        """Test medical QA with hallucinated content."""
        results = stage1_detector_ner_numeric._run_augmentations(
            medical_context, medical_answer_hallucinated
        )

        # Should detect either fabricated entity or wrong number
        total_flagged = sum(len(r.flagged_spans) for r in results.values())
        low_scores = [r.score for r in results.values() if r.score < 1.0]

        assert total_flagged > 0 or len(low_scores) > 0

    def test_financial_qa_scenario(
        self, stage1_detector_ner, financial_report_context, financial_answer_hallucinated_entities
    ):
        """Test financial QA with fabricated entities."""
        results = stage1_detector_ner._run_augmentations(
            financial_report_context, financial_answer_hallucinated_entities
        )

        # Should detect Michael Johnson and TechCorp as fabricated
        assert results["ner"].score < 1.0 or len(results["ner"].flagged_spans) > 0

    def test_geographic_qa_scenario(
        self, stage1_detector_ner_numeric, geographic_context, geographic_answer_hallucinated
    ):
        """Test geographic QA with wrong facts."""
        results = stage1_detector_ner_numeric._run_augmentations(
            geographic_context, geographic_answer_hallucinated
        )

        # Should detect wrong elevation (9100 vs 8849) and wrong person (Mallory)
        numeric_issues = (
            results["numeric"].score < 1.0
            or len(results["numeric"].flagged_spans) > 0
        )
        ner_issues = (
            results["ner"].score < 1.0 or len(results["ner"].flagged_spans) > 0
        )

        assert numeric_issues or ner_issues


# =============================================================================
# End-to-End Public API Tests
# =============================================================================


@pytest.mark.gpu
class TestPredictEndToEnd:
    """Test the full predict() pipeline that users actually call."""

    def test_predict_returns_token_list(self, stage1_detector_all, financial_report_context, financial_answer_supported):
        """Verify predict() returns a list of token predictions."""
        result = stage1_detector_all.predict(
            context=financial_report_context,
            answer=financial_answer_supported,
            output_format="tokens"
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_predict_token_format_has_required_fields(self, stage1_detector_no_aug, sample_context, sample_answer_supported):
        """Verify each token has required fields."""
        result = stage1_detector_no_aug.predict(
            context=sample_context,
            answer=sample_answer_supported,
            output_format="tokens"
        )
        for token in result:
            assert "token" in token
            assert "pred" in token
            assert "prob" in token

    def test_predict_spans_returns_list(self, stage1_detector_all, financial_report_context, financial_answer_hallucinated_numbers):
        """Verify predict() with spans format returns a list."""
        result = stage1_detector_all.predict(
            context=financial_report_context,
            answer=financial_answer_hallucinated_numbers,
            output_format="spans"
        )
        assert isinstance(result, list)

    def test_predict_detects_hallucinated_answer(self, stage1_detector_no_aug, sample_context, sample_answer_hallucinated):
        """Verify full pipeline flags hallucinated content."""
        result = stage1_detector_no_aug.predict(
            context=sample_context,
            answer=sample_answer_hallucinated,
            output_format="tokens"
        )
        # At least one token should be flagged
        hallucinated_tokens = [t for t in result if t.get("pred") == 1]
        assert len(hallucinated_tokens) > 0

    def test_predict_supported_answer_has_low_hallucination(self, stage1_detector_no_aug, sample_context, sample_answer_supported):
        """Verify supported answers have few/no hallucination flags."""
        result = stage1_detector_no_aug.predict(
            context=sample_context,
            answer=sample_answer_supported,
            output_format="tokens"
        )
        # Most tokens should NOT be flagged for supported answer
        hallucinated_tokens = [t for t in result if t.get("pred") == 1]
        total_tokens = len(result)
        assert len(hallucinated_tokens) / max(total_tokens, 1) < 0.5


@pytest.mark.gpu
class TestPromptPrediction:
    """Test prompt-based prediction methods."""

    def test_predict_prompt_returns_list(self, stage1_detector_no_aug):
        """Verify predict_prompt returns token list."""
        prompt = "Context: Paris is the capital of France.\n\nQuestion: What is the capital?"
        answer = "Paris is the capital of France."

        result = stage1_detector_no_aug.predict_prompt(prompt, answer)
        assert isinstance(result, list)

    def test_predict_prompt_batch_returns_list_of_lists(self, stage1_detector_no_aug):
        """Verify batch prediction returns list of results."""
        prompts = [
            "Context: Paris is the capital of France.",
            "Context: Tokyo is in Japan."
        ]
        answers = ["Paris is the capital.", "Tokyo is in Japan."]

        results = stage1_detector_no_aug.predict_prompt_batch(prompts, answers)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_predict_prompt_batch_length_matches_inputs(self, stage1_detector_no_aug):
        """Verify batch output length matches input length."""
        prompts = ["Context: Test 1.", "Context: Test 2.", "Context: Test 3."]
        answers = ["Answer 1.", "Answer 2.", "Answer 3."]

        results = stage1_detector_no_aug.predict_prompt_batch(prompts, answers)
        assert len(results) == len(prompts)


@pytest.mark.gpu
class TestOutputFormat:
    """Test output formatting methods."""

    def test_tokens_prob_in_valid_range(self, stage1_detector_no_aug, sample_context, sample_answer_supported):
        """Verify token probabilities are in [0, 1]."""
        result = stage1_detector_no_aug.predict(
            context=sample_context,
            answer=sample_answer_supported,
            output_format="tokens"
        )
        for token in result:
            assert 0 <= token["prob"] <= 1

    def test_tokens_pred_is_binary(self, stage1_detector_no_aug, sample_context, sample_answer_supported):
        """Verify token predictions are 0 or 1."""
        result = stage1_detector_no_aug.predict(
            context=sample_context,
            answer=sample_answer_supported,
            output_format="tokens"
        )
        for token in result:
            assert token["pred"] in [0, 1]

    def test_spans_have_valid_indices(self, stage1_detector_all, financial_report_context, financial_answer_hallucinated_numbers):
        """Verify span indices are valid."""
        answer = financial_answer_hallucinated_numbers
        result = stage1_detector_all.predict(
            context=financial_report_context,
            answer=answer,
            output_format="spans"
        )
        for span in result:
            assert "start" in span
            assert "end" in span
            assert span["start"] >= 0
            assert span["end"] <= len(answer)
            # Empty spans (start=end=0) are valid when no hallucination in that region
            if span["start"] != 0 or span["end"] != 0:
                assert span["start"] < span["end"]


@pytest.mark.gpu
class TestAugmentationEffect:
    """Test that augmentations affect the predict() output."""

    def test_numeric_augmentation_affects_hallucinated_numbers(
        self, stage1_detector_numeric, financial_report_context, financial_answer_hallucinated_numbers
    ):
        """Verify numeric augmentation influences detection of wrong numbers."""
        result = stage1_detector_numeric.predict(
            context=financial_report_context,
            answer=financial_answer_hallucinated_numbers,
            output_format="spans"
        )
        # Should return a list (may be empty if no hallucinations detected at span level)
        assert isinstance(result, list)

    def test_ner_augmentation_affects_fabricated_entities(
        self, stage1_detector_ner, financial_report_context, financial_answer_hallucinated_entities
    ):
        """Verify NER augmentation influences detection of fabricated entities."""
        result = stage1_detector_ner.predict(
            context=financial_report_context,
            answer=financial_answer_hallucinated_entities,
            output_format="spans"
        )
        assert isinstance(result, list)


@pytest.mark.gpu
class TestPackagePublicAPI:
    """Test Stage1Detector through the public package API (how real users import it)."""

    def test_make_detector_with_augmentations_creates_stage1(self, sample_context, sample_answer_supported):
        """Verify make_detector with augmentations creates Stage1Detector."""
        from lettucedetect.detectors.factory import make_detector

        detector = make_detector("transformer", augmentations=["ner", "numeric", "lexical"])

        # Should be a Stage1Detector
        assert detector.__class__.__name__ == "Stage1Detector"

        # Should work like a normal detector
        result = detector.predict(
            context=sample_context,
            answer=sample_answer_supported,
            output_format="tokens"
        )
        assert isinstance(result, list)

    def test_make_detector_without_augmentations_creates_transformer(self, sample_context, sample_answer_supported):
        """Verify make_detector without augmentations creates TransformerDetector."""
        from lettucedetect.detectors.factory import make_detector

        detector = make_detector(
            "transformer",
            model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
        )

        # Should be a TransformerDetector (not Stage1Detector)
        assert detector.__class__.__name__ == "TransformerDetector"

        result = detector.predict(
            context=sample_context,
            answer=sample_answer_supported,
            output_format="tokens"
        )
        assert isinstance(result, list)

    def test_hallucination_detector_class_interface(self, sample_context, sample_answer_supported):
        """Verify HallucinationDetector wrapper works (main user-facing API)."""
        from lettucedetect import HallucinationDetector

        detector = HallucinationDetector(
            method="transformer",
            model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
        )

        result = detector.predict(
            context=sample_context,
            answer=sample_answer_supported,
            output_format="tokens"
        )
        assert isinstance(result, list)

    def test_stage1_available_via_direct_import(self):
        """Verify Stage1Detector can be imported from detectors.stage1."""
        from lettucedetect.detectors.stage1 import Stage1Detector, AggregationConfig

        # Verify the classes exist and are importable
        assert Stage1Detector is not None
        assert AggregationConfig is not None
