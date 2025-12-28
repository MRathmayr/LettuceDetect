"""Numeric validation augmentation using regex patterns."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from lettucedetect.cascade.types import AugmentationResult
from lettucedetect.detectors.stage1.augmentations.base import BaseAugmentation
from lettucedetect.detectors.stage1.augmentations.config import NumericConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractedNumber:
    """Represents an extracted numeric value."""

    text: str
    value: float
    num_type: str  # integer, float, percentage, currency, year
    start: int
    end: int


class NumericValidator(BaseAugmentation):
    """Validate numeric values in answer exist in context.

    Extracts and validates:
    - Integers: 42, 1,000, -5
    - Floats: 3.14, 0.5, .25
    - Percentages: 50%, 99.9%
    - Currencies: $100, 100 dollars (normalized)
    - Years: 2024, 1999

    Tolerance:
    - Integers: exact match
    - Floats: configurable tolerance (default 1%)
    - Years: exact match
    """

    # Regex patterns for number extraction
    PATTERNS = {
        "percentage": re.compile(r"(\d+(?:\.\d+)?)\s*%"),
        "currency": re.compile(
            r"(?:[$€£¥])\s*(\d+(?:,\d{3})*(?:\.\d+)?)|"
            r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|euros?|pounds?)"
        ),
        "float": re.compile(r"(?<![,\d])(\d+\.\d+)(?![,\d])"),
        "integer": re.compile(r"(?<![.\d])(\d{1,3}(?:,\d{3})+|\d+)(?![.\d])"),
        "year": re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b"),
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

    def _extract_numbers(self, text: str) -> list[ExtractedNumber]:
        """Extract all numeric values from text."""
        numbers = []
        used_positions = set()

        # Extract percentages first (before floats consume them)
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

        # Extract currencies
        if self.config.extract_currencies:
            for match in self.PATTERNS["currency"].finditer(text):
                start, end = match.span()
                if not any(p in range(start, end) for p in used_positions):
                    # Get the captured group (either group 1 or 2)
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

        # Extract years
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

        # Extract floats
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

        # Extract integers
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
        # Different types don't match (except currency/integer can match)
        type_compatible = (
            answer_num.num_type == context_num.num_type
            or {answer_num.num_type, context_num.num_type} <= {"currency", "integer", "float"}
        )
        if not type_compatible:
            return False

        # Years and percentages require exact match
        if answer_num.num_type in ("year", "percentage"):
            return answer_num.value == context_num.value

        # Integers require exact match
        if answer_num.num_type == "integer" and context_num.num_type == "integer":
            return answer_num.value == context_num.value

        # Floats and currencies use tolerance
        if context_num.value == 0:
            return answer_num.value == 0

        diff_percent = abs(answer_num.value - context_num.value) / abs(context_num.value) * 100
        return diff_percent <= self.config.tolerance_percent

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
            AugmentationResult with score = ratio of verified numbers
        """
        # Extract numbers from context
        context_text = " ".join(context)
        context_numbers = self._extract_numbers(context_text)

        # Extract numbers from answer
        answer_numbers = self._extract_numbers(answer)

        # No numbers in answer = nothing to verify = fully supported
        if not answer_numbers:
            return AugmentationResult(
                score=1.0,
                confidence=0.5,  # Lower confidence when no numbers to check
                details={
                    "answer_numbers": 0,
                    "context_numbers": len(context_numbers),
                },
                flagged_spans=[],
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

        # Calculate support score
        support_score = verified_count / len(answer_numbers)

        return AugmentationResult(
            score=support_score,
            confidence=0.9,  # High confidence for numeric checks
            details={
                "answer_numbers": len(answer_numbers),
                "verified_numbers": verified_count,
                "context_numbers": len(context_numbers),
                "numbers": number_details,
            },
            flagged_spans=flagged_spans,
        )

    def preload(self) -> None:
        """No preloading needed for regex-based extraction."""
        pass
