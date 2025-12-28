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

    def setup_method(self):
        """Set up detector with augmentations."""
        self.detector = Stage1Detector(augmentations=["numeric", "lexical"])

    def test_aggregator_uses_configured_weights(self):
        """Verify aggregator uses configured weights."""
        custom_weights = {
            "transformer": 0.6,
            "numeric": 0.2,
            "lexical": 0.2,
        }
        config = Stage1Config(weights=custom_weights)
        detector = Stage1Detector(config=config, augmentations=["numeric", "lexical"])
        # Verify weights are passed through
        assert detector._aggregator._weights["transformer"] == 0.6


@pytest.mark.gpu
class TestNERAugmentationIntegration:
    """Test NER augmentation integration with Stage1Detector."""

    def setup_method(self):
        """Set up detector with NER augmentation."""
        self.detector = Stage1Detector(augmentations=["ner"])

    def test_ner_detects_fabricated_person(self, context_with_persons):
        """Verify NER catches fabricated person names."""
        # Context has "Albert Einstein" and "Mileva Maric"
        answer = "Dr. Richard Feynman developed the theory at the patent office."

        # Run augmentations only (no transformer)
        results = self.detector._run_augmentations(context_with_persons, answer)

        assert "ner" in results
        # Feynman is not in context
        ner_result = results["ner"]
        assert ner_result.score < 1.0 or len(ner_result.flagged_spans) > 0

    def test_ner_verifies_existing_person(self, context_with_persons):
        """Verify NER confirms existing person names."""
        answer = "Albert Einstein worked at the patent office."

        results = self.detector._run_augmentations(context_with_persons, answer)

        assert "ner" in results
        ner_result = results["ner"]
        # Einstein should be verified
        assert ner_result.score > 0

    def test_ner_with_organizations(self, context_with_organizations):
        """Verify NER works with organization entities."""
        # Context has Microsoft, LinkedIn, European Commission
        fabricated = "Google acquired Twitter for $30 billion."
        supported = "Microsoft acquired LinkedIn."

        fab_results = self.detector._run_augmentations(context_with_organizations, fabricated)
        sup_results = self.detector._run_augmentations(context_with_organizations, supported)

        # Fabricated should have lower score or flagged spans
        assert fab_results["ner"].score <= sup_results["ner"].score


@pytest.mark.gpu
class TestNumericAugmentationIntegration:
    """Test Numeric augmentation integration with Stage1Detector."""

    def setup_method(self):
        """Set up detector with Numeric augmentation."""
        self.detector = Stage1Detector(augmentations=["numeric"])

    def test_numeric_detects_fabricated_numbers(
        self, financial_report_context, financial_answer_hallucinated_numbers
    ):
        """Verify Numeric catches fabricated financial numbers."""
        results = self.detector._run_augmentations(
            financial_report_context, financial_answer_hallucinated_numbers
        )

        assert "numeric" in results
        numeric_result = results["numeric"]
        # Fabricated numbers should lower score
        assert numeric_result.score < 1.0
        assert len(numeric_result.flagged_spans) > 0

    def test_numeric_verifies_correct_numbers(
        self, financial_report_context, financial_answer_supported
    ):
        """Verify Numeric confirms correct numbers."""
        results = self.detector._run_augmentations(
            financial_report_context, financial_answer_supported
        )

        assert "numeric" in results
        numeric_result = results["numeric"]
        assert numeric_result.score == 1.0

    def test_numeric_with_percentages(self, context_with_percentages):
        """Verify Numeric handles percentages correctly."""
        correct = "Market share reached 31%."
        wrong = "Market share reached 45%."

        correct_results = self.detector._run_augmentations(context_with_percentages, correct)
        wrong_results = self.detector._run_augmentations(context_with_percentages, wrong)

        assert correct_results["numeric"].score > wrong_results["numeric"].score


@pytest.mark.gpu
class TestLexicalAugmentationIntegration:
    """Test Lexical augmentation integration with Stage1Detector."""

    def setup_method(self):
        """Set up detector with Lexical augmentation."""
        self.detector = Stage1Detector(augmentations=["lexical"])

    def test_lexical_high_overlap_supported(
        self, sample_context, sample_answer_supported
    ):
        """Verify high lexical overlap for supported answer."""
        results = self.detector._run_augmentations(sample_context, sample_answer_supported)

        assert "lexical" in results
        lexical_result = results["lexical"]
        assert lexical_result.score > 0.3

    def test_lexical_low_overlap_hallucinated(
        self, sample_context, sample_answer_hallucinated
    ):
        """Verify lower lexical overlap for hallucinated answer."""
        supported_results = self.detector._run_augmentations(
            sample_context, sample_answer_supported
        )
        hallucinated_results = self.detector._run_augmentations(
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

    def setup_method(self):
        """Set up detector with all augmentations."""
        self.detector = Stage1Detector(augmentations=["ner", "numeric", "lexical"])

    def test_all_augmentations_run(self, financial_report_context, financial_answer_supported):
        """Verify all augmentations produce results."""
        results = self.detector._run_augmentations(
            financial_report_context, financial_answer_supported
        )

        assert "ner" in results
        assert "numeric" in results
        assert "lexical" in results

    def test_aggregation_combines_scores(
        self, financial_report_context, financial_answer_hallucinated_numbers
    ):
        """Verify aggregator combines augmentation scores."""
        results = self.detector._run_augmentations(
            financial_report_context, financial_answer_hallucinated_numbers
        )

        # Create mock transformer predictions (no hallucination detected)
        mock_preds = [{"token": "test", "pred": 0, "prob": 0.1}]

        aggregated = self.detector._aggregator.aggregate(mock_preds, results)

        # Should have a combined hallucination score
        assert 0 <= aggregated.hallucination_score <= 1.0
        assert aggregated.confidence >= 0

    def test_multi_domain_hallucination_detection(self):
        """Test detection across different hallucination types."""
        context = [
            "Acme Corp reported $100 million revenue. CEO John Smith announced expansion."
        ]

        # Hallucinated entity (Jane Doe) + hallucinated number ($150 million)
        answer = "CEO Jane Doe announced $150 million in revenue."

        results = self.detector._run_augmentations(context, answer)

        # NER should flag Jane Doe
        ner_score = results["ner"].score
        # Numeric should flag 150
        numeric_score = results["numeric"].score

        # At least one should be low
        assert ner_score < 1.0 or numeric_score < 1.0


@pytest.mark.gpu
class TestRoutingDecisions:
    """Test routing decision logic in Stage1Detector."""

    def setup_method(self):
        """Set up detector with routing configuration."""
        agg_config = AggregationConfig(
            confidence_threshold_high=0.7,
            confidence_threshold_low=0.4,
        )
        self.detector = Stage1Detector(
            augmentations=["numeric", "lexical"],
            aggregation_config=agg_config,
        )

    def test_high_confidence_not_escalated(
        self, financial_report_context, financial_answer_supported
    ):
        """Verify high confidence results are not escalated."""
        results = self.detector._run_augmentations(
            financial_report_context, financial_answer_supported
        )
        mock_preds = [{"token": "test", "pred": 0, "prob": 0.05}]
        aggregated = self.detector._aggregator.aggregate(mock_preds, results)

        # Low hallucination score with high confidence should not escalate
        if aggregated.confident:
            assert not aggregated.escalate


@pytest.mark.gpu
class TestEdgeCases:
    """Test edge cases in Stage1Detector."""

    def test_empty_context_handling(self, empty_context):
        """Verify graceful handling of empty context."""
        detector = Stage1Detector(augmentations=["ner", "numeric", "lexical"])
        results = detector._run_augmentations(empty_context, "Some answer text.")

        # All augmentations should return valid results
        assert all(r.score is not None for r in results.values())

    def test_empty_answer_handling(self, sample_context, empty_answer):
        """Verify graceful handling of empty answer."""
        detector = Stage1Detector(augmentations=["ner", "numeric", "lexical"])
        results = detector._run_augmentations(sample_context, empty_answer)

        # Empty answer typically means nothing to verify
        assert all(r.score is not None for r in results.values())

    def test_very_long_context(self):
        """Verify handling of very long context."""
        detector = Stage1Detector(augmentations=["lexical"])
        long_context = ["The quick brown fox. " * 500]
        answer = "The fox is quick."

        results = detector._run_augmentations(long_context, answer)
        assert "lexical" in results
        assert results["lexical"].score is not None

    def test_special_characters_handling(self):
        """Verify handling of special characters."""
        detector = Stage1Detector(augmentations=["ner", "numeric"])
        context = ["Price: $100.00 (USD) - John O'Brien announced."]
        answer = "O'Brien said the price is $100."

        results = detector._run_augmentations(context, answer)
        assert all(r.score is not None for r in results.values())


@pytest.mark.gpu
class TestWarmup:
    """Test warmup functionality for latency optimization."""

    def test_warmup_loads_models(self):
        """Verify warmup pre-loads all required models."""
        detector = Stage1Detector(augmentations=["ner", "lexical"])
        detector.warmup()  # Should not raise

        # Verify spaCy model is loaded for NER
        ner_aug = next(a for a in detector._augmentations if a.name == "ner")
        assert ner_aug._nlp is not None


@pytest.mark.gpu
class TestBaseStageInterface:
    """Test BaseStage interface compliance."""

    def test_stage_name(self):
        """Verify stage_name property."""
        detector = Stage1Detector(augmentations=[])
        assert detector.stage_name == "stage1"

    def test_config_property(self):
        """Verify config property returns Stage1Config."""
        detector = Stage1Detector(augmentations=[])
        assert isinstance(detector.config, Stage1Config)


@pytest.mark.gpu
class TestRealWorldScenarios:
    """Test realistic RAG scenarios end-to-end."""

    def test_medical_qa_scenario(self, medical_context, medical_answer_hallucinated):
        """Test medical QA with hallucinated content."""
        detector = Stage1Detector(augmentations=["ner", "numeric"])
        results = detector._run_augmentations(
            medical_context, medical_answer_hallucinated
        )

        # Should detect either fabricated entity or wrong number
        total_flagged = sum(len(r.flagged_spans) for r in results.values())
        low_scores = [r.score for r in results.values() if r.score < 1.0]

        assert total_flagged > 0 or len(low_scores) > 0

    def test_financial_qa_scenario(
        self, financial_report_context, financial_answer_hallucinated_entities
    ):
        """Test financial QA with fabricated entities."""
        detector = Stage1Detector(augmentations=["ner"])
        results = detector._run_augmentations(
            financial_report_context, financial_answer_hallucinated_entities
        )

        # Should detect Michael Johnson and TechCorp as fabricated
        assert results["ner"].score < 1.0 or len(results["ner"].flagged_spans) > 0

    def test_geographic_qa_scenario(
        self, geographic_context, geographic_answer_hallucinated
    ):
        """Test geographic QA with wrong facts."""
        detector = Stage1Detector(augmentations=["ner", "numeric"])
        results = detector._run_augmentations(
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
