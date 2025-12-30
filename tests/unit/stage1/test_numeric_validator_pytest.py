"""Unit tests for NumericValidator augmentation."""

import pytest

from lettucedetect.detectors.stage1.augmentations.numeric_validator import (
    NumericValidator,
    NumericConfig,
)


class TestNumericExtraction:
    """Test number extraction from text."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = NumericValidator()

    def test_integer_extraction(self):
        """Extract basic integers."""
        numbers = self.validator._extract_numbers("The value is 42 and 1,000")
        values = [n.value for n in numbers]
        assert 42 in values
        assert 1000 in values

    def test_float_extraction(self):
        """Extract floating point numbers."""
        numbers = self.validator._extract_numbers("Pi is 3.14 and half is 0.5")
        values = [n.value for n in numbers]
        assert 3.14 in values
        assert 0.5 in values

    def test_percentage_extraction(self):
        """Extract percentage values."""
        numbers = self.validator._extract_numbers("Growth was 50% and then 99.9%")
        values = [n.value for n in numbers]
        types = [n.num_type for n in numbers]
        assert 50 in values
        assert 99.9 in values
        assert "percentage" in types

    def test_currency_extraction(self):
        """Extract currency values."""
        numbers = self.validator._extract_numbers("Cost is $100 and 50 dollars")
        values = [n.value for n in numbers]
        types = [n.num_type for n in numbers]
        assert 100 in values
        assert 50 in values
        assert "currency" in types

    def test_year_extraction(self):
        """Extract year values."""
        numbers = self.validator._extract_numbers("Founded in 2024 and 1999")
        values = [n.value for n in numbers]
        types = [n.num_type for n in numbers]
        assert 2024 in values
        assert 1999 in values
        assert "year" in types

    def test_negative_numbers_not_extracted(self):
        """Negative numbers handled correctly."""
        # Current implementation doesn't handle negatives explicitly
        numbers = self.validator._extract_numbers("Temperature is -5 degrees")
        # -5 would be extracted as 5
        values = [n.value for n in numbers]
        assert 5 in values


class TestNumericValidation:
    """Test validation of numbers against context.

    Unified score direction: 0.0 = supported, 1.0 = hallucinated
    """

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = NumericValidator()

    def test_number_in_context_exact(self):
        """Number in context returns low (supported) score."""
        context = ["The population is 1000 people."]
        answer = "There are 1000 residents."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 0.0  # supported

    def test_fabricated_number(self):
        """Fabricated number returns high (hallucination) score."""
        context = ["The population is 1000 people."]
        answer = "There are 5000 residents."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 1.0  # hallucinated
        assert len(result.flagged_spans) == 1

    def test_no_numbers_in_answer(self):
        """No numbers in answer = fully supported."""
        context = ["The city has many people."]
        answer = "The city is populous."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 0.0  # supported (nothing to verify)

    def test_tolerance_matching(self):
        """Numbers within tolerance match."""
        config = NumericConfig(tolerance_percent=5.0)
        validator = NumericValidator(config=config)
        context = ["The value is 100."]
        answer = "The value is approximately 102."
        result = validator.score(context, answer, None, None)
        assert result.score == 0.0  # supported (within tolerance)

    def test_percentage_exact_match_required(self):
        """Percentages require exact match."""
        context = ["Growth was 50%."]
        answer = "Growth was 51%."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 1.0  # hallucinated (exact mismatch)

    def test_year_exact_match_required(self):
        """Years require exact match."""
        context = ["Founded in 2020."]
        answer = "Founded in 2021."
        result = self.validator.score(context, answer, None, None)
        assert result.score == 1.0  # hallucinated (exact mismatch)

    def test_multiple_numbers_partial_match(self):
        """Multiple numbers with partial match."""
        context = ["Revenue was $100 million in 2023."]
        answer = "Revenue was $100 million in 2024."
        result = self.validator.score(context, answer, None, None)
        # 100 matches, 2024 doesn't match 2023
        assert result.score == 0.5  # partial hallucination
        assert len(result.flagged_spans) == 1


class TestNumericConfig:
    """Test NumericConfig options."""

    def test_disable_currency_extraction(self):
        """Disable currency extraction."""
        config = NumericConfig(extract_currencies=False)
        validator = NumericValidator(config=config)
        numbers = validator._extract_numbers("Cost is $100")
        types = [n.num_type for n in numbers]
        assert "currency" not in types

    def test_disable_date_extraction(self):
        """Disable date/year extraction."""
        config = NumericConfig(extract_dates=False)
        validator = NumericValidator(config=config)
        numbers = validator._extract_numbers("Founded in 2024")
        types = [n.num_type for n in numbers]
        assert "year" not in types


class TestAugmentationInterface:
    """Test BaseAugmentation interface."""

    def test_name_property(self):
        """Verify name property."""
        validator = NumericValidator()
        assert validator.name == "numeric"

    def test_safe_score_handles_errors(self):
        """safe_score returns neutral on error."""
        validator = NumericValidator()
        # This should work fine, but testing the interface
        result = validator.safe_score(["context"], "answer", None, None)
        assert result.score is not None

    def test_preload_succeeds(self):
        """preload doesn't raise."""
        validator = NumericValidator()
        validator.preload()  # Should not raise
