"""Numeric validation augmentation using regex patterns."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from lettucedetect.cascade.types import AugmentationResult
from lettucedetect.detectors.stage1.augmentations.base import BaseAugmentation
from lettucedetect.detectors.stage1.augmentations.config import NumericConfig

logger = logging.getLogger(__name__)

# Abbreviation multipliers (case-insensitive)
ABBREVIATIONS = {
    "k": 1_000,
    "m": 1_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}

# Word numbers
WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "half": 0.5, "quarter": 0.25, "third": 0.333,
    "dozen": 12, "score": 20, "gross": 144,
}


@dataclass
class ExtractedNumber:
    """Represents an extracted numeric value."""

    text: str
    value: float
    num_type: str  # integer, float, percentage, currency, year, range, etc.
    start: int
    end: int
    range_high: float | None = None  # Only for ranges


class NumericValidator(BaseAugmentation):
    """Validate numeric values in answer exist in context.

    Extracts and validates:
    - Integers: 42, 1,000, -5
    - Floats: 3.14, 0.5, .25
    - Percentages: 50%, 99.9%
    - Currencies: $100, 100 dollars (normalized)
    - Years: 2024, 1999
    - Abbreviated: 10k, 5M, 2.5bn, 3 million
    - Word numbers: five, dozen, twenty-three
    - Fractions: 3/4, 1/2
    - Ranges: 10-20, 5 to 10, between 10 and 20

    Tolerance:
    - Integers: exact match
    - Floats: configurable tolerance (default 1%)
    - Years: exact match
    """

    # Regex patterns for number extraction - ORDER MATTERS!
    # More specific patterns must come before general ones
    PATTERNS = {
        # 1. Percentages (before float consumes "50" from "50%")
        "percentage": re.compile(r"(\d+(?:\.\d+)?)\s*%"),

        # 2. Currency (before integer consumes "$100")
        "currency": re.compile(
            r"(?:[$\u20ac\u00a3\u00a5])\s*(\d+(?:,\d{3})*(?:\.\d+)?)|"
            r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|euros?|pounds?)"
        ),

        # 3. Scientific notation (before float consumes "1.5" from "1.5e6")
        "scientific": re.compile(r"(\d+(?:\.\d+)?)\s*[eE]\s*([+-]?\d+)"),

        # 4. Abbreviated numbers (before integer consumes "5" from "5M")
        # Uses negative lookahead to avoid matching units like "km", "mm"
        "abbreviated": re.compile(
            r"(\d+(?:\.\d+)?)\s*(k|m|bn?|million|billion|trillion)(?![a-z])",
            re.IGNORECASE
        ),

        # 5. Fractions (before integer consumes "3" from "3/4")
        "fraction": re.compile(r"\b(\d+)\s*/\s*(\d+)\b"),

        # 5b. Ratios (1:10, 3:1)
        "ratio": re.compile(r"\b(\d+)\s*:\s*(\d+)\b"),

        # 6. Ordinals (1st, 2nd, 23rd)
        "ordinal": re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE),

        # 7. Ranges with hyphen (10-20) - must have digits on both sides
        "range_hyphen": re.compile(r"\b(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\b"),

        # 8. Ranges with words (5 to 10, between 10 and 20)
        "range_words": re.compile(
            r"\b(?:between\s+)?(\d+(?:\.\d+)?)\s+(?:to|and)\s+(\d+(?:\.\d+)?)\b",
            re.IGNORECASE
        ),

        # 9. Compound word numbers (twenty-three)
        "compound_word": re.compile(
            r"\b((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
            r"[-\s]?(?:one|two|three|four|five|six|seven|eight|nine))\b",
            re.IGNORECASE
        ),

        # 10. Word numbers (five, dozen)
        "word_number": re.compile(
            r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety|hundred|thousand|half|quarter|third|dozen|score|gross)\b",
            re.IGNORECASE
        ),

        # 11. Years (1600-2099 range)
        "year": re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b"),

        # 12. Floats
        "float": re.compile(r"(?<![,\d])(\d+\.\d+)(?![,\d])"),

        # 13. Integers (catch-all, last)
        # Allow period at end of sentence (followed by space or end of string)
        # Supports negative numbers with optional minus sign
        "integer": re.compile(r"(?<![.\d])(-?\d{1,3}(?:,\d{3})+|-?\d+)(?![\d]|\.(?=\d))"),
    }

    def __init__(self, config: NumericConfig | None = None) -> None:
        """Initialize numeric validator.

        Args:
            config: NumericConfig with tolerance settings
        """
        self.config = config or NumericConfig()

    @property
    def name(self) -> str:
        """Return augmentation name."""
        return "numeric"

    def _parse_number(self, text: str) -> float:
        """Parse number string to float, handling commas."""
        cleaned = text.replace(",", "").replace(" ", "")
        return float(cleaned)

    def _parse_compound_word(self, text: str) -> float:
        """Parse compound word number like 'twenty-three'."""
        text_lower = text.lower().replace("-", " ").replace("  ", " ")
        parts = text_lower.split()
        total = 0.0
        for part in parts:
            if part in WORD_NUMBERS:
                total += WORD_NUMBERS[part]
        return total

    def _normalize_word_numbers(self, text: str) -> str:
        """Convert written numbers to digits using word2number.

        Examples:
        - "twenty-three" -> "23"
        - "one million" -> "1000000"
        - "two hundred and fifty" -> "250"
        """
        if not self.config.normalize_word_numbers:
            return text

        try:
            from word2number import w2n
        except ImportError:
            logger.debug("word2number not installed, skipping normalization")
            return text

        # Pattern to find potential number phrases
        # Matches sequences of number words
        number_words = (
            r"\b((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety|hundred|thousand|million|billion|trillion|and)"
            r"(?:[-\s]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
            r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
            r"eighty|ninety|hundred|thousand|million|billion|trillion|and))*)\b"
        )
        pattern = re.compile(number_words, re.IGNORECASE)

        def replace_word_number(match: re.Match) -> str:
            phrase = match.group(1)
            try:
                num = w2n.word_to_num(phrase)
                return str(num)
            except ValueError:
                return phrase  # Keep original if parsing fails

        return pattern.sub(replace_word_number, text)

    def _extract_numbers(self, text: str) -> list[ExtractedNumber]:
        """Extract all numeric values from text."""
        numbers = []
        used_positions = set()

        # Process patterns in order (most specific first)

        # 1. Percentages
        for match in self.PATTERNS["percentage"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                value = self._parse_number(match.group(1))
                numbers.append(
                    ExtractedNumber(
                        text=match.group(),
                        value=value,
                        num_type="percentage",
                        start=start,
                        end=end,
                    )
                )
                used_positions.update(range(start, end))

        # 2. Currencies
        if self.config.extract_currencies:
            for match in self.PATTERNS["currency"].finditer(text):
                start, end = match.span()
                if not any(p in range(start, end) for p in used_positions):
                    value_str = match.group(1) or match.group(2)
                    if value_str:
                        value = self._parse_number(value_str)
                        numbers.append(
                            ExtractedNumber(
                                text=match.group(),
                                value=value,
                                num_type="currency",
                                start=start,
                                end=end,
                            )
                        )
                        used_positions.update(range(start, end))

        # 3. Scientific notation
        for match in self.PATTERNS["scientific"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                base = self._parse_number(match.group(1))
                exp = int(match.group(2))
                value = base * (10 ** exp)
                numbers.append(
                    ExtractedNumber(
                        text=match.group(),
                        value=value,
                        num_type="scientific",
                        start=start,
                        end=end,
                    )
                )
                used_positions.update(range(start, end))

        # 4. Abbreviated numbers
        for match in self.PATTERNS["abbreviated"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                base = self._parse_number(match.group(1))
                abbrev = match.group(2).lower()
                multiplier = ABBREVIATIONS.get(abbrev, 1)
                value = base * multiplier
                numbers.append(
                    ExtractedNumber(
                        text=match.group(),
                        value=value,
                        num_type="abbreviated",
                        start=start,
                        end=end,
                    )
                )
                used_positions.update(range(start, end))

        # 5. Fractions
        for match in self.PATTERNS["fraction"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                numerator = int(match.group(1))
                denominator = int(match.group(2))
                if denominator != 0:
                    value = numerator / denominator
                    numbers.append(
                        ExtractedNumber(
                            text=match.group(),
                            value=value,
                            num_type="fraction",
                            start=start,
                            end=end,
                        )
                    )
                    used_positions.update(range(start, end))

        # 5b. Ratios (1:10, 3:1) - stored as decimal ratio
        for match in self.PATTERNS["ratio"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                left = int(match.group(1))
                right = int(match.group(2))
                if right != 0:
                    value = left / right
                    numbers.append(
                        ExtractedNumber(
                            text=match.group(),
                            value=value,
                            num_type="ratio",
                            start=start,
                            end=end,
                        )
                    )
                    used_positions.update(range(start, end))

        # 6. Ordinals
        for match in self.PATTERNS["ordinal"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                value = int(match.group(1))
                numbers.append(
                    ExtractedNumber(
                        text=match.group(),
                        value=value,
                        num_type="ordinal",
                        start=start,
                        end=end,
                    )
                )
                used_positions.update(range(start, end))

        # 7. Ranges with hyphen - DISABLED (causes -0.095 AUROC regression on RAGTruth)
        # The range patterns match date-like patterns (2020-2021) and "X and Y" listings
        # that are not actually numeric ranges, causing false positives.

        # 8. Ranges with words - DISABLED (causes AUROC regression)
        # Same issue as hyphen ranges.

        # 9. Compound word numbers - DISABLED (causes false positives)
        # Word number extraction is too aggressive and matches common words
        # that aren't actually numeric in context

        # 10. Simple word numbers - DISABLED (causes false positives)
        # Word number extraction is too aggressive and matches common words
        # that aren't actually numeric in context

        # 11. Years
        if self.config.extract_dates:
            for match in self.PATTERNS["year"].finditer(text):
                start, end = match.span()
                if not any(p in range(start, end) for p in used_positions):
                    value = self._parse_number(match.group(1))
                    numbers.append(
                        ExtractedNumber(
                            text=match.group(),
                            value=value,
                            num_type="year",
                            start=start,
                            end=end,
                        )
                    )
                    used_positions.update(range(start, end))

        # 12. Floats
        for match in self.PATTERNS["float"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                value = self._parse_number(match.group(1))
                numbers.append(
                    ExtractedNumber(
                        text=match.group(),
                        value=value,
                        num_type="float",
                        start=start,
                        end=end,
                    )
                )
                used_positions.update(range(start, end))

        # 13. Integers (catch-all)
        for match in self.PATTERNS["integer"].finditer(text):
            start, end = match.span()
            if not any(p in range(start, end) for p in used_positions):
                value = self._parse_number(match.group(1))
                numbers.append(
                    ExtractedNumber(
                        text=match.group(),
                        value=value,
                        num_type="integer",
                        start=start,
                        end=end,
                    )
                )
                used_positions.update(range(start, end))

        return numbers

    def _numbers_match(self, answer_num: ExtractedNumber, context_num: ExtractedNumber) -> bool:
        """Check if two numbers match within tolerance."""
        # Different types don't match (except compatible types)
        type_compatible = (
            answer_num.num_type == context_num.num_type
            or {answer_num.num_type, context_num.num_type} <= {"currency", "integer", "float"}
            or {answer_num.num_type, context_num.num_type} <= {"abbreviated", "integer"}
            or {answer_num.num_type, context_num.num_type} <= {"fraction", "ratio", "float"}
        )
        if not type_compatible:
            return False

        # Years and percentages require exact match
        if answer_num.num_type in ("year", "percentage"):
            return answer_num.value == context_num.value

        # Ordinals require exact match
        if answer_num.num_type == "ordinal":
            return answer_num.value == context_num.value

        # Integer↔float compatibility: allow "100" to match "100.0"
        if {answer_num.num_type, context_num.num_type} == {"integer", "float"}:
            # Check if values are exactly equal (handles 100 == 100.0)
            if answer_num.value == context_num.value:
                return True
            # For whole number floats, also check integer equivalence
            int_val = answer_num.value if answer_num.num_type == "integer" else context_num.value
            float_val = context_num.value if answer_num.num_type == "integer" else answer_num.value
            if float_val == int(float_val) and int(float_val) == int_val:
                return True
            return False

        # Integers require exact match (when both are integers)
        if answer_num.num_type == "integer" and context_num.num_type == "integer":
            return answer_num.value == context_num.value

        # Floats, currencies, fractions, scientific use per-type tolerance
        if context_num.value == 0:
            return answer_num.value == 0

        diff_percent = abs(answer_num.value - context_num.value) / abs(context_num.value) * 100

        # Select tolerance based on answer type
        if answer_num.num_type == "percentage":
            tolerance = self.config.tolerance_percentage
        elif answer_num.num_type == "currency":
            tolerance = self.config.tolerance_currency
        elif answer_num.num_type == "float":
            tolerance = self.config.tolerance_float
        else:
            tolerance = self.config.tolerance_percent

        return diff_percent <= tolerance

    def _number_in_context(
        self, answer_num: ExtractedNumber, context_numbers: list[ExtractedNumber]
    ) -> bool:
        """Check if answer number exists in context numbers."""
        for ctx_num in context_numbers:
            if self._numbers_match(answer_num, ctx_num):
                return True
        return False

    def score(
        self,
        context: list[str],
        answer: str,
        question: str | None,
        token_predictions: list[dict] | None,
    ) -> AugmentationResult:
        """Score answer numbers against context.

        Returns:
            AugmentationResult with hallucination score (0 = supported, 1 = hallucinated).
            Score is ratio of unverified numbers.
        """
        # Normalize text numbers if enabled (e.g., "twenty-three" -> "23")
        context_text = " ".join(context)
        answer_text = answer
        if self.config.normalize_word_numbers:
            context_text = self._normalize_word_numbers(context_text)
            answer_text = self._normalize_word_numbers(answer_text)

        # Extract numbers from context
        context_numbers = self._extract_numbers(context_text)

        # Extract numbers from answer
        answer_numbers = self._extract_numbers(answer_text)

        # No numbers in answer = nothing to verify = no evidence of hallucination
        if not answer_numbers:
            return AugmentationResult(
                score=0.0,  # 0.0 = supported/no hallucination (absence of numbers is not hallucination)
                evidence={
                    "numbers_checked": 0,
                    "numbers_verified": 0,
                    "context_numbers": len(context_numbers),
                },
                details={
                    "answer_numbers": 0,
                    "context_numbers": len(context_numbers),
                },
                flagged_spans=[],
                is_active=False,  # Nothing to verify
            )

        # Verify each answer number
        verified_count = 0
        flagged_spans = []
        number_details = []

        for num in answer_numbers:
            found = self._number_in_context(num, context_numbers)
            number_details.append(
                {
                    "text": num.text,
                    "value": num.value,
                    "type": num.num_type,
                    "found": found,
                }
            )

            if found:
                verified_count += 1
            else:
                # Flag unverified number
                flagged_spans.append(
                    {
                        "start": num.start,
                        "end": num.end,
                        "text": num.text,
                        "confidence": 0.9,
                        "reason": f"Number '{num.text}' ({num.num_type}) not found in context",
                    }
                )

        # Calculate hallucination score (ratio of unverified numbers)
        # 0 = all verified (supported), 1 = none verified (hallucinated)
        unverified_ratio = 1.0 - (verified_count / len(answer_numbers))

        return AugmentationResult(
            score=unverified_ratio,  # 0 = supported, 1 = hallucinated
            evidence={
                "numbers_checked": len(answer_numbers),
                "numbers_verified": verified_count,
                "context_numbers": len(context_numbers),
            },
            details={
                "answer_numbers": len(answer_numbers),
                "verified_numbers": verified_count,
                "context_numbers": len(context_numbers),
                "numbers": number_details,
            },
            flagged_spans=flagged_spans,
            is_active=True,  # Numbers found and verified
        )

    def preload(self) -> None:
        """No preloading needed for regex-based extraction."""
        pass
