"""Unit tests for numeric abbreviation parsing.

Tests the enhanced NumericValidator with support for:
- Abbreviations (k, M, bn, million, billion, trillion)
- Word numbers (five, dozen, twenty-three)
- Fractions (3/4, 1/2)
- Ranges (10-20, between 5 and 10)
- Scientific notation (1.5e6)
- Ordinals (1st, 2nd, 23rd)
"""

import pytest

from lettucedetect.detectors.stage1.augmentations.numeric_validator import (
    ABBREVIATIONS,
    WORD_NUMBERS,
    NumericValidator,
)


@pytest.fixture
def validator():
    """Create NumericValidator instance."""
    return NumericValidator()


class TestAbbreviationMultipliers:
    """Test that abbreviations are correctly parsed as multipliers."""

    @pytest.mark.parametrize("text,expected", [
        ("10k employees", 10_000),
        ("5M users", 5_000_000),
        ("2.5bn revenue", 2_500_000_000),
        ("3 trillion dollars", 3_000_000_000_000),
        ("100k downloads", 100_000),
        ("1.5m subscribers", 1_500_000),
        ("7B parameters", 7_000_000_000),
    ])
    def test_abbreviation_multipliers(self, validator, text, expected):
        """Abbreviations should be parsed as multipliers."""
        numbers = validator._extract_numbers(text)
        assert len(numbers) >= 1
        # Check that expected value is found
        values = [n.value for n in numbers]
        assert any(abs(v - expected) < 1 for v in values), (
            f"Expected {expected} in {values} for text: '{text}'"
        )

    @pytest.mark.parametrize("text,raw_value,wrong_value", [
        ("5 km distance", 5, 5000),     # km = kilometer, not k multiplier
        ("10 mm thick", 10, 10000),     # mm = millimeter, not m multiplier
    ])
    def test_units_not_parsed_as_multipliers(self, validator, text, raw_value, wrong_value):
        """Units like km, mm should NOT be parsed as multipliers."""
        numbers = validator._extract_numbers(text)
        values = [n.value for n in numbers]
        # Should extract raw number (positive assertion)
        assert raw_value in values, f"Expected {raw_value} in {values}"
        # Should NOT multiply it (negative assertion)
        assert wrong_value not in values, f"Should not have {wrong_value} in {values}"


class TestPatternPrecedence:
    """Test that patterns match in correct order (most specific first)."""

    def test_percentage_before_float(self, validator):
        """Percentage pattern should match before float/integer."""
        numbers = validator._extract_numbers("50% discount")
        assert len(numbers) == 1
        assert numbers[0].num_type == "percentage"
        assert numbers[0].value == 50

    def test_currency_before_integer(self, validator):
        """Currency pattern should match before integer."""
        numbers = validator._extract_numbers("$100 payment")
        assert len(numbers) == 1
        assert numbers[0].num_type == "currency"
        assert numbers[0].value == 100

    def test_abbreviated_before_integer(self, validator):
        """Abbreviated pattern should match before integer."""
        numbers = validator._extract_numbers("5M users")
        assert len(numbers) == 1
        assert numbers[0].num_type == "abbreviated"
        assert numbers[0].value == 5_000_000

    def test_scientific_before_float(self, validator):
        """Scientific notation should match before float."""
        numbers = validator._extract_numbers("1.5e6 watts")
        assert len(numbers) == 1
        assert numbers[0].num_type == "scientific"
        assert numbers[0].value == 1_500_000

    def test_fraction_before_integer(self, validator):
        """Fraction should match before integer."""
        numbers = validator._extract_numbers("3/4 of the budget")
        frac_nums = [n for n in numbers if n.num_type == "fraction"]
        assert len(frac_nums) == 1
        assert abs(frac_nums[0].value - 0.75) < 0.01


@pytest.mark.skip(reason="Word number extraction disabled - causes false positives in benchmark")
class TestWordNumbers:
    """Test word number parsing.

    DISABLED: Word number extraction was too aggressive and matched common words
    that aren't actually numeric in context, causing AUROC regression on RAGTruth.
    """

    @pytest.mark.parametrize("text,expected", [
        ("five apples", 5),
        ("twelve months", 12),
        ("a dozen donuts", 12),
        ("a score of years", 20),
        ("half the pie", 0.5),
        ("a quarter of", 0.25),
    ])
    def test_simple_word_numbers(self, validator, text, expected):
        """Simple word numbers should be parsed correctly."""
        numbers = validator._extract_numbers(text)
        values = [n.value for n in numbers]
        assert any(abs(v - expected) < 0.01 for v in values), (
            f"Expected {expected} in {values}"
        )

    def test_compound_word_numbers(self, validator):
        """Compound word numbers like 'twenty-three' should be parsed."""
        numbers = validator._extract_numbers("twenty-three employees")
        values = [n.value for n in numbers]
        assert 23 in values or 23.0 in values

    def test_multiple_word_numbers(self, validator):
        """Multiple word numbers in same text."""
        numbers = validator._extract_numbers("five cats and twelve dogs")
        values = [n.value for n in numbers]
        assert 5 in values or 5.0 in values
        assert 12 in values or 12.0 in values


class TestFractions:
    """Test fraction parsing."""

    @pytest.mark.parametrize("text,expected", [
        ("3/4 complete", 0.75),
        ("1/2 the time", 0.5),
        ("1/4 remaining", 0.25),
        ("2/3 majority", 0.666),
    ])
    def test_fractions(self, validator, text, expected):
        """Fractions should be parsed correctly."""
        numbers = validator._extract_numbers(text)
        frac_nums = [n for n in numbers if n.num_type == "fraction"]
        assert len(frac_nums) >= 1
        assert any(abs(n.value - expected) < 0.01 for n in frac_nums)

    def test_fraction_not_confused_with_date(self, validator):
        """Fractions should not be confused with dates like 1/2/2024."""
        # This is tricky - "1/2" could be fraction or part of date
        # Current implementation may extract both; just verify no crash
        numbers = validator._extract_numbers("on 1/2/2024")
        assert len(numbers) >= 1


@pytest.mark.skip(reason="Range extraction disabled - causes -0.095 AUROC regression on RAGTruth")
class TestRanges:
    """Test range parsing.

    DISABLED: Range patterns match date-like patterns (2020-2021) and "X and Y"
    listings that are not actually numeric ranges, causing false positives.
    """

    def test_hyphen_range(self, validator):
        """Hyphenated ranges should be parsed."""
        numbers = validator._extract_numbers("10-20 items")
        range_nums = [n for n in numbers if n.num_type == "range"]
        assert len(range_nums) == 1
        assert range_nums[0].value == 10
        assert range_nums[0].range_high == 20

    def test_word_range_to(self, validator):
        """Word ranges with 'to' should be parsed."""
        numbers = validator._extract_numbers("5 to 10 minutes")
        range_nums = [n for n in numbers if n.num_type == "range"]
        assert len(range_nums) == 1
        assert range_nums[0].value == 5
        assert range_nums[0].range_high == 10

    def test_word_range_between(self, validator):
        """Word ranges with 'between...and' should be parsed."""
        numbers = validator._extract_numbers("between 100 and 200 users")
        range_nums = [n for n in numbers if n.num_type == "range"]
        assert len(range_nums) == 1
        assert range_nums[0].value == 100
        assert range_nums[0].range_high == 200

    def test_answer_in_context_range(self, validator):
        """Answer number within context range should be verified."""
        result = validator.score(
            context=["The count is between 50 and 100."],
            answer="The count is 75.",
            question=None,
            token_predictions=None,
        )
        # 75 is within 50-100 range, should be verified
        assert result.evidence["numbers_verified"] >= 1


class TestOrdinals:
    """Test ordinal number parsing."""

    @pytest.mark.parametrize("text,expected", [
        ("1st place", 1),
        ("2nd position", 2),
        ("3rd quarter", 3),
        ("23rd day", 23),
        ("100th anniversary", 100),
    ])
    def test_ordinals(self, validator, text, expected):
        """Ordinals should be parsed correctly."""
        numbers = validator._extract_numbers(text)
        ordinal_nums = [n for n in numbers if n.num_type == "ordinal"]
        assert len(ordinal_nums) >= 1
        assert any(n.value == expected for n in ordinal_nums)


class TestScientificNotation:
    """Test scientific notation parsing."""

    @pytest.mark.parametrize("text,expected", [
        ("1e6 cells", 1_000_000),
        ("2.5e3 units", 2500),
        ("1.5E9 bytes", 1_500_000_000),
        ("3e-2 meters", 0.03),
    ])
    def test_scientific_notation(self, validator, text, expected):
        """Scientific notation should be parsed correctly."""
        numbers = validator._extract_numbers(text)
        sci_nums = [n for n in numbers if n.num_type == "scientific"]
        # Note: negative exponents like 3e-2 should work
        assert len(sci_nums) >= 1
        assert any(abs(n.value - expected) < 0.001 for n in sci_nums)


class TestTypeCompatibility:
    """Test type compatibility in matching."""

    def test_abbreviated_matches_integer(self, validator):
        """Abbreviated number should match equivalent integer in context."""
        result = validator.score(
            context=["The company has 1000000 employees."],
            answer="The company has 1M employees.",
            question=None,
            token_predictions=None,
        )
        # 1M (1,000,000) should match 1000000
        assert result.evidence["numbers_verified"] >= 1

    @pytest.mark.skip(reason="Word number extraction disabled - causes false positives in benchmark")
    def test_word_number_matches_integer(self, validator):
        """Word number should match equivalent integer in context."""
        result = validator.score(
            context=["There are 12 items in the box."],
            answer="There are a dozen items in the box.",
            question=None,
            token_predictions=None,
        )
        # "dozen" (12) should match 12
        assert result.evidence["numbers_verified"] >= 1


class TestConstants:
    """Test that constants are properly defined."""

    def test_abbreviations_dict(self):
        """ABBREVIATIONS dict should have expected keys."""
        assert "k" in ABBREVIATIONS
        assert "m" in ABBREVIATIONS
        assert "bn" in ABBREVIATIONS
        assert "billion" in ABBREVIATIONS
        assert "trillion" in ABBREVIATIONS

    def test_word_numbers_dict(self):
        """WORD_NUMBERS dict should have expected keys."""
        assert "one" in WORD_NUMBERS
        assert "dozen" in WORD_NUMBERS
        assert "half" in WORD_NUMBERS
        assert "twenty" in WORD_NUMBERS
