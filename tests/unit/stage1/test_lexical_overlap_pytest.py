"""Unit tests for LexicalOverlapCalculator."""

import pytest

from lettucedetect.utils.lexical import LexicalOverlapCalculator, LexicalConfig


class TestTokenization:
    """Test tokenization logic."""

    def test_basic_tokenization(self):
        """Basic word tokenization."""
        calc = LexicalOverlapCalculator(LexicalConfig(use_stemming=False, remove_stopwords=False))
        tokens = calc._tokenize("Hello world test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stopword_removal(self):
        """Stopwords are removed."""
        calc = LexicalOverlapCalculator(LexicalConfig(use_stemming=False, remove_stopwords=True))
        tokens = calc._tokenize("The quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens

    def test_stemming(self):
        """Words are stemmed."""
        calc = LexicalOverlapCalculator(LexicalConfig(use_stemming=True, remove_stopwords=False))
        tokens = calc._tokenize("running runs")
        # Porter stemmer stems these to "run"
        assert all(t == "run" for t in tokens)


class TestNgramExtraction:
    """Test n-gram extraction."""

    def test_unigrams(self):
        """Extract unigrams only."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=False, ngram_range=(1, 1))
        )
        tokens = ["a", "b", "c"]
        ngrams = calc._get_ngrams(tokens)
        assert ("a",) in ngrams
        assert ("b",) in ngrams
        assert ("c",) in ngrams
        assert len(ngrams) == 3

    def test_bigrams(self):
        """Extract bigrams."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=False, ngram_range=(2, 2))
        )
        tokens = ["a", "b", "c"]
        ngrams = calc._get_ngrams(tokens)
        assert ("a", "b") in ngrams
        assert ("b", "c") in ngrams
        assert len(ngrams) == 2

    def test_mixed_ngrams(self):
        """Extract unigrams and bigrams."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=False, ngram_range=(1, 2))
        )
        tokens = ["a", "b", "c"]
        ngrams = calc._get_ngrams(tokens)
        # 3 unigrams + 2 bigrams = 5
        assert len(ngrams) == 5


class TestJaccardCalculation:
    """Test Jaccard similarity calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=False, ngram_range=(1, 1))
        )

    def test_full_overlap(self):
        """Answer is subset of context."""
        context = ["The quick brown fox jumps"]
        answer = "quick brown"
        score = self.calc.compute_jaccard(context, answer)
        # answer words are subset of context, so intersection/union > 0
        assert score > 0

    def test_no_overlap(self):
        """No common words."""
        context = ["The quick brown fox"]
        answer = "xyz abc"
        score = self.calc.compute_jaccard(context, answer)
        assert score == 0.0

    def test_empty_answer(self):
        """Empty answer returns 1.0 (no hallucination possible)."""
        context = ["The quick brown fox"]
        answer = ""
        score = self.calc.compute_jaccard(context, answer)
        assert score == 1.0

    def test_identical_texts(self):
        """Identical texts have score close to 1."""
        context = ["hello world"]
        answer = "hello world"
        score = self.calc.compute_jaccard(context, answer)
        assert score == 1.0


class TestSupportScore:
    """Test support score calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calc = LexicalOverlapCalculator()

    def test_high_support(self):
        """High overlap = high support."""
        context = ["The company reported revenue of 100 million dollars"]
        answer = "The company reported revenue"
        score = self.calc.compute_support_score(context, answer)
        # Jaccard depends on n-gram overlap; with stemming/stopwords, expect > 0.3
        assert score > 0.3

    def test_low_support(self):
        """Low overlap = low support."""
        context = ["The company reported revenue"]
        answer = "Sales declined significantly last quarter"
        score = self.calc.compute_support_score(context, answer)
        assert score < 0.5


class TestAugmentationInterface:
    """Test BaseAugmentation interface compliance."""

    def test_name_property(self):
        """Verify name property."""
        calc = LexicalOverlapCalculator()
        assert calc.name == "lexical"

    def test_score_method(self):
        """score method returns AugmentationResult."""
        calc = LexicalOverlapCalculator()
        result = calc.score(["context text"], "answer text", None, None)
        assert hasattr(result, "score")
        assert hasattr(result, "evidence")
        assert hasattr(result, "details")
        assert hasattr(result, "flagged_spans")

    def test_score_details(self):
        """score method includes jaccard in details."""
        calc = LexicalOverlapCalculator()
        result = calc.score(["context text"], "answer text", None, None)
        assert "jaccard" in result.details

    def test_preload_downloads_nltk(self):
        """preload downloads NLTK resources."""
        calc = LexicalOverlapCalculator()
        calc.preload()  # Should not raise


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_context(self):
        """Empty context list."""
        calc = LexicalOverlapCalculator()
        result = calc.score([], "answer", None, None)
        # Empty context means nothing to compare against
        assert result.score is not None

    def test_multiple_context_passages(self):
        """Multiple context passages are joined."""
        calc = LexicalOverlapCalculator()
        context = ["First passage about cats", "Second passage about dogs"]
        answer = "cats and dogs"
        result = calc.score(context, answer, None, None)
        assert result.score > 0


class TestNegationPreservation:
    """Test that negation words are preserved during stopword removal."""

    def test_negation_words_preserved(self):
        """Negation words 'not', 'no', 'never' are not removed as stopwords."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=True)
        )
        tokens = calc._tokenize("This is not true")
        assert "not" in tokens

    def test_negation_no_preserved(self):
        """'no' is preserved."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=True)
        )
        tokens = calc._tokenize("There is no evidence")
        assert "no" in tokens

    def test_negation_never_preserved(self):
        """'never' is preserved."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=True)
        )
        tokens = calc._tokenize("This has never happened")
        assert "never" in tokens

    def test_negation_contractions_preserved(self):
        """Contracted negations like 'don't' -> 'don' are preserved."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=True)
        )
        # "don't" tokenizes to ["don", "t"]
        tokens = calc._tokenize("I don't know")
        assert "don" in tokens

    def test_negation_affects_overlap_score(self):
        """Negation preservation affects overlap score."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=True, ngram_range=(1, 1))
        )
        # Context says "This is true", answer says "This is not true"
        # Without negation preservation, "not" would be removed as stopword
        # and both would have same tokens -> high overlap
        # With negation preservation, "not" is kept -> different tokens -> lower overlap
        context = ["This statement is true"]
        answer_without_negation = "This statement is true"
        answer_with_negation = "This statement is not true"

        score_without = calc.compute_jaccard(context, answer_without_negation)
        score_with = calc.compute_jaccard(context, answer_with_negation)

        # Answer with negation should have lower Jaccard due to "not" being preserved
        assert score_with < score_without

    def test_other_stopwords_still_removed(self):
        """Other stopwords (not negations) are still removed."""
        calc = LexicalOverlapCalculator(
            LexicalConfig(use_stemming=False, remove_stopwords=True)
        )
        tokens = calc._tokenize("The cat is on the mat")
        # "the", "is", "on" are stopwords and should be removed
        assert "the" not in tokens
        assert "is" not in tokens
        assert "on" not in tokens
        # "cat", "mat" are content words and should be kept
        assert "cat" in tokens
        assert "mat" in tokens
