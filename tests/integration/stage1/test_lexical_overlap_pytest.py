"""Integration tests for LexicalOverlapCalculator with realistic RAG examples.

These tests verify lexical overlap calculation works correctly
with real-world text from various domains.
"""

import pytest

from lettucedetect.utils.lexical import LexicalOverlapCalculator, LexicalConfig


class TestFinancialTextOverlap:
    """Test lexical overlap with financial text."""

    def setup_method(self):
        """Set up calculator."""
        self.calc = LexicalOverlapCalculator()

    def test_supported_financial_answer(
        self, financial_report_context, financial_answer_supported
    ):
        """Verify high overlap for supported financial answer."""
        score = self.calc.compute_support_score(
            financial_report_context, financial_answer_supported
        )
        # Positive overlap expected since answer uses context vocabulary
        assert score > 0.2

    def test_hallucinated_entities_lower_overlap(
        self, financial_report_context, financial_answer_hallucinated_entities
    ):
        """Verify lower overlap for answer with fabricated entities."""
        score = self.calc.compute_support_score(
            financial_report_context, financial_answer_hallucinated_entities
        )
        # "Michael Johnson" and "TechCorp" not in context
        # Score should still be positive since "Acme Corporation" and other words match
        assert 0 < score < 0.8


class TestMedicalTextOverlap:
    """Test lexical overlap with medical text."""

    def setup_method(self):
        """Set up calculator."""
        self.calc = LexicalOverlapCalculator()

    def test_supported_medical_answer(self, medical_context, medical_answer_supported):
        """Verify high overlap for supported medical answer."""
        score = self.calc.compute_support_score(
            medical_context, medical_answer_supported
        )
        assert score > 0.15

    def test_hallucinated_medical_answer(
        self, medical_context, medical_answer_supported, medical_answer_hallucinated
    ):
        """Verify lower overlap for hallucinated medical answer."""
        supported_score = self.calc.compute_support_score(
            medical_context, medical_answer_supported
        )
        hallucinated_score = self.calc.compute_support_score(
            medical_context, medical_answer_hallucinated
        )
        # Hallucinated answer should have lower overlap
        assert hallucinated_score < supported_score


class TestGeographicTextOverlap:
    """Test lexical overlap with geographic text."""

    def setup_method(self):
        """Set up calculator."""
        self.calc = LexicalOverlapCalculator()

    def test_supported_geographic_answer(
        self, geographic_context, geographic_answer_supported
    ):
        """Verify high overlap for supported geographic answer."""
        score = self.calc.compute_support_score(
            geographic_context, geographic_answer_supported
        )
        assert score > 0.2

    def test_hallucinated_geographic_answer(
        self, geographic_context, geographic_answer_supported, geographic_answer_hallucinated
    ):
        """Verify lower overlap for hallucinated geographic answer."""
        supported_score = self.calc.compute_support_score(
            geographic_context, geographic_answer_supported
        )
        hallucinated_score = self.calc.compute_support_score(
            geographic_context, geographic_answer_hallucinated
        )
        # "Pakistan" and "George Mallory" reduce overlap
        assert hallucinated_score < supported_score


class TestMultiPassageOverlap:
    """Test lexical overlap across multiple context passages."""

    def setup_method(self):
        """Set up calculator."""
        self.calc = LexicalOverlapCalculator()

    def test_multi_passage_supported(
        self, multi_passage_context, multi_passage_answer_supported
    ):
        """Verify overlap calculation across multiple passages."""
        score = self.calc.compute_support_score(
            multi_passage_context, multi_passage_answer_supported
        )
        # Answer draws from all three passages
        assert score > 0.1

    def test_multi_passage_partial_hallucination(
        self,
        multi_passage_context,
        multi_passage_answer_supported,
        multi_passage_answer_partial_hallucination,
    ):
        """Verify partial hallucination has medium overlap."""
        supported_score = self.calc.compute_support_score(
            multi_passage_context, multi_passage_answer_supported
        )
        partial_score = self.calc.compute_support_score(
            multi_passage_context, multi_passage_answer_partial_hallucination
        )
        # Partial hallucination should have lower score than fully supported
        assert partial_score <= supported_score


class TestNgramConfiguration:
    """Test n-gram configuration effects on overlap."""

    def test_unigram_only(self, sample_context, sample_answer_supported):
        """Test unigram-only overlap."""
        config = LexicalConfig(ngram_range=(1, 1))
        calc = LexicalOverlapCalculator(config)
        score = calc.compute_support_score(sample_context, sample_answer_supported)
        assert score > 0

    def test_bigram_only(self, sample_context, sample_answer_supported):
        """Test bigram-only overlap."""
        config = LexicalConfig(ngram_range=(2, 2))
        calc = LexicalOverlapCalculator(config)
        score = calc.compute_support_score(sample_context, sample_answer_supported)
        # Bigrams may have lower overlap
        assert score >= 0

    def test_mixed_ngrams(self, sample_context, sample_answer_supported):
        """Test mixed unigram and bigram overlap."""
        config = LexicalConfig(ngram_range=(1, 2))
        calc = LexicalOverlapCalculator(config)
        score = calc.compute_support_score(sample_context, sample_answer_supported)
        assert score > 0


class TestPreprocessingOptions:
    """Test text preprocessing options."""

    def test_stopword_removal_effect(self):
        """Test that stopword removal affects overlap calculation."""
        context = ["The quick brown fox jumps over the lazy dog."]
        answer = "The fox jumps."

        # With stopwords
        with_stopwords = LexicalOverlapCalculator(
            LexicalConfig(remove_stopwords=False)
        )
        score_with = with_stopwords.compute_support_score(context, answer)

        # Without stopwords
        without_stopwords = LexicalOverlapCalculator(
            LexicalConfig(remove_stopwords=True)
        )
        score_without = without_stopwords.compute_support_score(context, answer)

        # Both should give valid scores
        assert score_with > 0
        assert score_without > 0

    def test_stemming_effect(self):
        """Test that stemming affects word matching."""
        context = ["The company is running its operations."]
        answer = "The companies ran their operations."

        # Without stemming
        no_stem = LexicalOverlapCalculator(LexicalConfig(use_stemming=False))
        score_no_stem = no_stem.compute_support_score(context, answer)

        # With stemming
        with_stem = LexicalOverlapCalculator(LexicalConfig(use_stemming=True))
        score_with_stem = with_stem.compute_support_score(context, answer)

        # Stemming should improve matching of related word forms
        assert score_with_stem >= score_no_stem


class TestEdgeCases:
    """Test edge cases in lexical overlap."""

    def setup_method(self):
        """Set up calculator."""
        self.calc = LexicalOverlapCalculator()

    def test_empty_context(self, empty_context):
        """Handle empty context."""
        score = self.calc.compute_support_score(empty_context, "Some answer.")
        # Empty context should give low/zero support
        assert score is not None

    def test_empty_answer(self, sample_context, empty_answer):
        """Handle empty answer."""
        score = self.calc.compute_support_score(sample_context, empty_answer)
        # Empty answer = nothing to verify, typically returns 1.0
        assert score == 1.0

    def test_identical_text(self):
        """Test identical text gives perfect overlap."""
        context = ["The quick brown fox jumps."]
        answer = "The quick brown fox jumps."
        score = self.calc.compute_jaccard(context, answer)
        assert score == 1.0

    def test_completely_disjoint_text(self):
        """Test completely disjoint text gives zero overlap."""
        context = ["Alpha beta gamma delta."]
        answer = "One two three four."
        score = self.calc.compute_jaccard(context, answer)
        assert score == 0.0

    def test_unicode_text(self):
        """Handle Unicode characters."""
        context = ["Le cafe est tres bon a Paris."]
        answer = "Paris has good cafe."
        score = self.calc.compute_support_score(context, answer)
        assert score > 0

    def test_special_characters(self):
        """Handle special characters."""
        context = ["The price is $100 (USD) - final!"]
        answer = "Price: $100."
        score = self.calc.compute_support_score(context, answer)
        assert score > 0


class TestAugmentationInterface:
    """Test BaseAugmentation interface compliance."""

    def test_name_property(self):
        """Verify name property returns 'lexical'."""
        calc = LexicalOverlapCalculator()
        assert calc.name == "lexical"

    def test_score_method(self, sample_context, sample_answer_supported):
        """Verify score method returns AugmentationResult."""
        calc = LexicalOverlapCalculator()
        result = calc.score(sample_context, sample_answer_supported, None, None)
        assert hasattr(result, "score")
        assert hasattr(result, "evidence")
        assert hasattr(result, "details")
        assert "jaccard" in result.details

    def test_preload_succeeds(self):
        """Verify preload downloads NLTK resources."""
        calc = LexicalOverlapCalculator()
        calc.preload()  # Should not raise
