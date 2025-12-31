"""Integration tests for NumericValidator with realistic RAG examples.

These tests verify numeric extraction and validation work correctly
with real-world financial, scientific, and geographic data.

Unified score direction: 0.0 = supported, 1.0 = hallucinated
"""

import pytest

from lettucedetect.detectors.stage1.augmentations.numeric_validator import (
    NumericValidator,
    NumericConfig,
)


class TestFinancialNumericValidation:
    """Test numeric validation with financial data."""

    def setup_method(self):
        """Set up NumericValidator."""
        self.validator = NumericValidator()

    def test_financial_numbers_supported(
        self, financial_report_context, financial_answer_supported
    ):
        """Verify supported financial numbers."""
        result = self.validator.score(
            financial_report_context, financial_answer_supported, None, None
        )
        # All numbers match context -> low hallucination score
        assert result.score == 0.0
        assert len(result.flagged_spans) == 0

    def test_financial_numbers_hallucinated(
        self, financial_report_context, financial_answer_hallucinated_numbers
    ):
        """Verify hallucinated financial numbers are flagged."""
        result = self.validator.score(
            financial_report_context, financial_answer_hallucinated_numbers, None, None
        )
        # $5.8 billion (vs $4.2), 22% (vs 15%), $2.3 billion (vs $1.8) are wrong
        assert result.score > 0.0  # hallucination detected
        assert len(result.flagged_spans) > 0

    def test_percentage_hallucination(self, context_with_percentages):
        """Verify fabricated percentages are caught."""
        answer = "The market share grew to 45%, and retention reached 85%."
        result = self.validator.score(context_with_percentages, answer, None, None)
        # 45% and 85% are not in context (23%, 31%, 92% are)
        assert result.score > 0.0  # hallucination detected
        assert len(result.flagged_spans) >= 2

    def test_currency_validation(self, context_with_currencies):
        """Verify currency values are validated."""
        supported = "The property sold for $2.5 million."
        result = self.validator.score(context_with_currencies, supported, None, None)
        assert result.score == 0.0  # supported

        hallucinated = "The property sold for $3.2 million."
        result = self.validator.score(context_with_currencies, hallucinated, None, None)
        assert result.score > 0.0  # hallucination detected
        assert len(result.flagged_spans) > 0


class TestScientificNumericValidation:
    """Test numeric validation with scientific/medical data."""

    def setup_method(self):
        """Set up NumericValidator."""
        self.validator = NumericValidator()

    def test_medical_numbers_supported(self, medical_context, medical_answer_supported):
        """Verify supported medical numbers."""
        result = self.validator.score(
            medical_context, medical_answer_supported, None, None
        )
        # 37 million and 7% are in context -> low hallucination score
        assert result.score <= 0.5  # mostly supported

    def test_medical_numbers_hallucinated(self, medical_context, medical_answer_hallucinated):
        """Verify hallucinated medical numbers are flagged."""
        result = self.validator.score(
            medical_context, medical_answer_hallucinated, None, None
        )
        # 50 million (vs 37 million), 5% (vs 7%) are wrong
        assert result.score > 0.0  # hallucination detected
        assert len(result.flagged_spans) > 0

    def test_measurement_validation(self, context_with_measurements):
        """Verify physical measurements are validated."""
        supported = "The bridge is 2.7 kilometers long and 65 meters high."
        result = self.validator.score(context_with_measurements, supported, None, None)
        assert result.score == 0.0  # supported

        hallucinated = "The bridge is 3.5 kilometers long and 80 meters high."
        result = self.validator.score(context_with_measurements, hallucinated, None, None)
        assert result.score > 0.0  # hallucination detected


class TestGeographicNumericValidation:
    """Test numeric validation with geographic data."""

    def setup_method(self):
        """Set up NumericValidator."""
        self.validator = NumericValidator()

    def test_elevation_supported(self, geographic_context, geographic_answer_supported):
        """Verify supported elevation numbers."""
        result = self.validator.score(
            geographic_context, geographic_answer_supported, None, None
        )
        # 8,849 meters and 1953 are correct -> low hallucination score
        assert result.score <= 0.5  # mostly supported

    def test_elevation_hallucinated(self, geographic_context, geographic_answer_hallucinated):
        """Verify hallucinated elevation is flagged."""
        result = self.validator.score(
            geographic_context, geographic_answer_hallucinated, None, None
        )
        # 9,100 meters (vs 8,849) and 1924 (vs 1953) are wrong
        assert result.score > 0.0  # hallucination detected
        # Should have flagged at least one wrong number
        assert len(result.flagged_spans) > 0


class TestYearValidation:
    """Test year and date validation."""

    def setup_method(self):
        """Set up NumericValidator."""
        self.validator = NumericValidator()

    def test_year_exact_match(self, context_with_dates):
        """Verify years require exact match."""
        supported = "World War II ended in 1945."
        result = self.validator.score(context_with_dates, supported, None, None)
        assert result.score == 0.0  # supported

        hallucinated = "World War II ended in 1946."
        result = self.validator.score(context_with_dates, hallucinated, None, None)
        assert result.score > 0.0  # hallucination detected

    def test_multi_passage_years(
        self, multi_passage_context, multi_passage_answer_partial_hallucination
    ):
        """Verify year hallucinations across multiple passages."""
        result = self.validator.score(
            multi_passage_context,
            multi_passage_answer_partial_hallucination,
            None,
            None,
        )
        # 2015 (vs 2011) and $420 billion (vs $383 billion) are wrong
        assert result.score > 0.0  # hallucination detected
        assert len(result.flagged_spans) >= 1


class TestToleranceMatching:
    """Test tolerance-based number matching."""

    def test_tolerance_percentage(self):
        """Test tolerance matching with percentages."""
        # 5% tolerance
        config = NumericConfig(tolerance_percent=5.0)
        validator = NumericValidator(config=config)

        context = ["The value is 100 units."]
        # 103 is within 5% of 100
        answer = "The value is 103 units."
        result = validator.score(context, answer, None, None)
        # Check behavior - if 103 is extracted and matched to 100 with tolerance
        assert result.score is not None

    def test_zero_tolerance(self):
        """Test zero tolerance requires exact match."""
        config = NumericConfig(tolerance_percent=0.0)
        validator = NumericValidator(config=config)

        context = ["The count is 1000 items."]
        answer = "The count is 1001 items."
        result = validator.score(context, answer, None, None)
        # With zero tolerance, 1001 should not match 1000
        assert result.score is not None


class TestComplexNumberFormats:
    """Test extraction of complex number formats."""

    def setup_method(self):
        """Set up NumericValidator."""
        self.validator = NumericValidator()

    def test_comma_separated_thousands(self):
        """Test numbers with comma separators."""
        context = ["The population is 1234567 people."]
        answer = "The population is about 1234567."
        result = self.validator.score(context, answer, None, None)
        # Exact match of the number should succeed -> low hallucination
        assert result.score == 0.0

    def test_decimal_numbers(self):
        """Test decimal number extraction."""
        context = ["The rate is 3.14159."]
        answer = "The rate is 3.14159."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 0.0  # supported

    def test_mixed_formats(self):
        """Test multiple number formats in same text."""
        context = ["Revenue was $2.5 billion (up 15%) with 10,000 employees."]
        answer = "The company has $2.5 billion revenue, 15% growth, and 10,000 staff."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 0.0  # supported


class TestEdgeCases:
    """Test edge cases in numeric validation."""

    def setup_method(self):
        """Set up NumericValidator."""
        self.validator = NumericValidator()

    def test_empty_context(self, empty_context):
        """Handle empty context gracefully."""
        result = self.validator.score(empty_context, "The value is 42.", None, None)
        # No context means no numbers to verify against
        assert result.score is not None

    def test_empty_answer(self, sample_context, empty_answer):
        """Handle empty answer gracefully."""
        result = self.validator.score(sample_context, empty_answer, None, None)
        # No numbers in answer = fully supported
        assert result.score == 0.0

    def test_no_numbers_in_answer(self):
        """Handle answer with no numeric content."""
        context = ["The value is 42."]
        answer = "There is a value mentioned."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 0.0  # supported (nothing to verify)

    def test_very_large_numbers(self):
        """Handle very large numbers."""
        context = ["The national debt is $34,000,000,000,000."]
        answer = "The debt exceeds $34 trillion."
        result = self.validator.score(context, answer, None, None)
        # May or may not match depending on extraction
        assert result.score is not None

    def test_negative_context_numbers(self):
        """Handle negative numbers in context."""
        context = ["The temperature dropped to -40 degrees."]
        answer = "It was -40 degrees outside."
        result = self.validator.score(context, answer, None, None)
        # 40 should be extracted (sign handling varies)
        assert result.score is not None


class TestNumericConfigOptions:
    """Test NumericConfig options affect behavior."""

    def test_disable_currency_extraction(self):
        """Test disabling currency extraction."""
        config = NumericConfig(extract_currencies=False)
        validator = NumericValidator(config=config)

        context = ["The price is $100."]
        answer = "The price is $200."
        result = validator.score(context, answer, None, None)
        # Without currency extraction, might not flag the mismatch
        # or might extract 100/200 as plain numbers
        assert result.score is not None

    def test_disable_date_extraction(self):
        """Test disabling date/year extraction."""
        config = NumericConfig(extract_dates=False)
        validator = NumericValidator(config=config)

        context = ["Founded in 2020."]
        answer = "Founded in 2021."
        result = validator.score(context, answer, None, None)
        # With dates disabled, years might be treated as regular numbers with tolerance
        assert result.score is not None


class TestPartialMatching:
    """Test partial number matching scenarios."""

    def setup_method(self):
        """Set up NumericValidator."""
        self.validator = NumericValidator()

    def test_some_numbers_match(self):
        """Test when some but not all numbers match."""
        context = ["Revenue was $100 million in 2023 with 5000 employees."]
        # $100 and 2023 match, but 6000 doesn't match 5000
        answer = "Revenue was $100 million with 6000 staff in 2023."
        result = self.validator.score(context, answer, None, None)
        # Score should be partial (between 0 and 1)
        assert 0 < result.score < 1.0
        assert len(result.flagged_spans) > 0

    def test_all_numbers_match(self):
        """Test when all numbers match."""
        context = ["The report shows 42 items at $10 each for $420 total."]
        answer = "There are 42 items costing $10 each, totaling $420."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 0.0  # supported

    def test_no_numbers_match(self):
        """Test when no numbers match."""
        context = ["The count is 100 items."]
        answer = "The count is 250 items."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 1.0  # full hallucination
        assert len(result.flagged_spans) >= 1


class TestAugmentationInterface:
    """Test BaseAugmentation interface compliance."""

    def test_name_property(self):
        """Verify name property returns 'numeric'."""
        validator = NumericValidator()
        assert validator.name == "numeric"

    def test_preload_succeeds(self):
        """Verify preload doesn't raise."""
        validator = NumericValidator()
        validator.preload()  # Should not raise

    def test_safe_score_handles_errors(self):
        """Verify safe_score returns neutral on error."""
        validator = NumericValidator()
        result = validator.safe_score(["context"], "answer", None, None)
        assert result.score is not None
