"""NER-based verification augmentation using spaCy."""

from __future__ import annotations

import logging

from lettucedetect.cascade.types import AugmentationResult
from lettucedetect.detectors.stage1.augmentations.base import BaseAugmentation
from lettucedetect.detectors.stage1.augmentations.config import NERConfig

logger = logging.getLogger(__name__)


class NERVerifier(BaseAugmentation):
    """Verify named entities in answer exist in context.

    Uses spaCy for entity extraction and rapidfuzz for fuzzy matching.
    Returns ratio of verified entities as support score.

    Limitations:
    - Poor on domain-specific entities
    - Reliable for common entity types: PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PERCENT
    """

    def __init__(self, config: NERConfig | None = None) -> None:
        """Initialize NER verifier.

        Args:
            config: NERConfig with spacy_model, entity_types, fuzzy_threshold
        """
        self.config = config or NERConfig()
        self._nlp = None

    def _load_model(self) -> None:
        """Lazy load spaCy model, downloading if necessary."""
        if self._nlp is None:
            import spacy
            from spacy.util import is_package

            model_name = self.config.spacy_model
            if not is_package(model_name):
                logger.info(f"Downloading spaCy model: {model_name}")
                from spacy.cli import download

                download(model_name)

            self._nlp = spacy.load(model_name)

    @property
    def name(self) -> str:
        """Return augmentation name."""
        return "ner"

    def preload(self) -> None:
        """Preload spaCy model."""
        self._load_model()

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text.

        Returns:
            List of dicts with text, label, start, end
        """
        self._load_model()
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.config.entity_types:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )
        return entities

    def _entity_in_context(
        self, entity_text: str, context_entities: list[dict]
    ) -> tuple[bool, float]:
        """Check if entity exists in context entities.

        Returns:
            (found, similarity_score) tuple
        """
        from rapidfuzz import fuzz

        entity_lower = entity_text.lower()

        for ctx_ent in context_entities:
            ctx_text_lower = ctx_ent["text"].lower()

            # Exact match
            if entity_lower == ctx_text_lower:
                return True, 1.0

            # Fuzzy match
            similarity = fuzz.ratio(entity_lower, ctx_text_lower) / 100.0
            if similarity >= self.config.fuzzy_threshold:
                return True, similarity

        return False, 0.0

    def score(
        self,
        context: list[str],
        answer: str,
        question: str | None,
        token_predictions: list[dict] | None,
    ) -> AugmentationResult:
        """Score answer entities against context.

        Returns:
            AugmentationResult with score = ratio of verified entities
        """
        # Extract entities from context
        context_text = " ".join(context)
        context_entities = self._extract_entities(context_text)

        # Extract entities from answer
        answer_entities = self._extract_entities(answer)

        # No entities in answer = nothing to verify = fully supported
        if not answer_entities:
            return AugmentationResult(
                score=1.0,
                confidence=0.5,  # Lower confidence when no entities to check
                details={"answer_entities": 0, "context_entities": len(context_entities)},
                flagged_spans=[],
            )

        # Verify each answer entity
        verified_count = 0
        flagged_spans = []
        entity_details = []

        for ent in answer_entities:
            found, similarity = self._entity_in_context(ent["text"], context_entities)
            entity_details.append(
                {
                    "text": ent["text"],
                    "label": ent["label"],
                    "found": found,
                    "similarity": similarity,
                }
            )

            if found:
                verified_count += 1
            else:
                # Flag unverified entity
                flagged_spans.append(
                    {
                        "start": ent["start"],
                        "end": ent["end"],
                        "text": ent["text"],
                        "confidence": 0.8,
                        "reason": f"Entity '{ent['text']}' ({ent['label']}) not found in context",
                    }
                )

        # Calculate support score
        support_score = verified_count / len(answer_entities)

        return AugmentationResult(
            score=support_score,
            confidence=0.8,
            details={
                "answer_entities": len(answer_entities),
                "verified_entities": verified_count,
                "context_entities": len(context_entities),
                "entities": entity_details,
            },
            flagged_spans=flagged_spans,
        )
