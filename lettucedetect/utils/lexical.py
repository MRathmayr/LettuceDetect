"""Shared lexical overlap calculator for Stage 1 and Stage 2."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from lettucedetect.cascade.types import AugmentationResult
from lettucedetect.detectors.stage1.augmentations.base import BaseAugmentation


@dataclass
class LexicalConfig:
    """Configuration for lexical overlap calculation."""

    use_stemming: bool = True
    remove_stopwords: bool = True
    ngram_range: tuple[int, int] = (1, 2)
    stopwords_lang: str = "english"


class LexicalOverlapCalculator(BaseAugmentation):
    """Calculate lexical overlap between context and answer.

    Used by both Stage 1 (augmentation) and Stage 2 (component).
    """

    def __init__(self, config: LexicalConfig | None = None) -> None:
        """Initialize calculator with optional config."""
        self.config = config or LexicalConfig()
        self._stemmer = None
        self._stopwords: set[str] | None = None

        if self.config.use_stemming:
            from nltk.stem import PorterStemmer

            self._stemmer = PorterStemmer()

        if self.config.remove_stopwords:
            import nltk
            import nltk.data
            from nltk.corpus import stopwords

            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)

            self._stopwords = set(stopwords.words(self.config.stopwords_lang))

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize and optionally stem/filter text."""
        tokens = re.findall(r"\b\w+\b", text.lower())

        if self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens

    def _get_ngrams(self, tokens: list[str]) -> Counter:
        """Extract n-grams from token list."""
        ngrams: Counter = Counter()
        min_n, max_n = self.config.ngram_range

        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i : i + n])
                ngrams[ngram] += 1

        return ngrams

    def compute_jaccard(self, context: list[str], answer: str) -> float:
        """Compute Jaccard similarity between context and answer n-grams."""
        context_text = " ".join(context)
        context_tokens = self._tokenize(context_text)
        answer_tokens = self._tokenize(answer)

        context_ngrams = set(self._get_ngrams(context_tokens).keys())
        answer_ngrams = set(self._get_ngrams(answer_tokens).keys())

        if not answer_ngrams:
            return 1.0  # No answer tokens = no hallucination possible

        intersection = context_ngrams & answer_ngrams
        union = context_ngrams | answer_ngrams

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def compute_support_score(self, context: list[str], answer: str) -> float:
        """Compute support score (0=unsupported, 1=fully supported)."""
        return self.compute_jaccard(context, answer)

    def score(
        self,
        context: list[str],
        answer: str,
        question: str | None = None,
        token_predictions: list[dict] | None = None,
    ) -> AugmentationResult:
        """Score method for Stage 1 augmentation interface.

        Returns hallucination score (0 = supported, 1 = hallucinated).
        Low overlap suggests hallucination, high overlap suggests support.
        """
        context_text = " ".join(context)
        context_tokens = self._tokenize(context_text)
        answer_tokens = self._tokenize(answer)

        overlap_score = self.compute_support_score(context, answer)
        # Convert support score to hallucination score
        # High overlap = supported = low hallucination score
        hallucination_score = 1.0 - overlap_score

        return AugmentationResult(
            score=hallucination_score,  # 0 = supported, 1 = hallucinated
            evidence={
                "tokens_analyzed": len(answer_tokens),
                "context_tokens": len(context_tokens),
                "overlap_ratio": overlap_score,
            },
            details={"jaccard": overlap_score, "hallucination_score": hallucination_score},
            flagged_spans=[],
        )

    @property
    def name(self) -> str:
        """Return augmentation name."""
        return "lexical"

    def preload(self) -> None:
        """Preload NLTK resources."""
        import nltk

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
