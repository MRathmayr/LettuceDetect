"""NER-based verification augmentation using spaCy or GLiNER."""

from __future__ import annotations

import logging

from rapidfuzz import fuzz

from lettucedetect.cascade.types import AugmentationResult
from lettucedetect.detectors.stage1.augmentations.base import BaseAugmentation
from lettucedetect.detectors.stage1.augmentations.config import NERConfig

logger = logging.getLogger(__name__)


class NERVerifier(BaseAugmentation):
    """Verify named entities in answer exist in context.

    Supports two NER backends:
    - spaCy (default): Traditional NER, fast, good for common entity types
    - GLiNER: Zero-shot NER, higher recall for domain-specific entities

    Returns ratio of verified entities as support score.

    Limitations:
    - spaCy: Poor on domain-specific entities
    - GLiNER: Requires GPU for reasonable latency (~500ms on GPU, several seconds on CPU)
    """

    def __init__(self, config: NERConfig | None = None) -> None:
        """Initialize NER verifier.

        Args:
            config: NERConfig with model selection and thresholds
        """
        self.config = config or NERConfig()
        self._nlp = None  # spaCy model
        self._gliner = None  # GLiNER model

    def _load_spacy_model(self) -> None:
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

    def _load_gliner_model(self) -> None:
        """Lazy load GLiNER model."""
        if self._gliner is None:
            try:
                from gliner import GLiNER

                logger.info(f"Loading GLiNER model: {self.config.gliner_model}")
                try:
                    self._gliner = GLiNER.from_pretrained(self.config.gliner_model)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load GLiNER model '{self.config.gliner_model}': {e}. "
                        "Ensure the model exists on HuggingFace and you have network access."
                    ) from e

                # Move to GPU if available and enabled
                if self.config.use_gpu:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            self._gliner = self._gliner.to("cuda")
                            logger.info("GLiNER model loaded on GPU")
                        else:
                            logger.warning("GPU not available, GLiNER running on CPU (slow)")
                    except ImportError:
                        logger.warning("torch not available, GLiNER running on CPU")
                else:
                    logger.info("GLiNER model loaded on CPU (use_gpu=False)")
            except ImportError:
                raise ImportError(
                    "GLiNER not installed. Install with: pip install gliner"
                )

    @property
    def name(self) -> str:
        """Return augmentation name."""
        return "ner"

    def preload(self) -> None:
        """Preload NER model based on configuration."""
        if self.config.model == "gliner":
            self._load_gliner_model()
        else:
            self._load_spacy_model()

    def _extract_entities_spacy(self, text: str) -> list[dict]:
        """Extract named entities using spaCy.

        Returns:
            List of dicts with text, label, start, end
        """
        self._load_spacy_model()
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

    def _extract_entities_gliner(self, text: str) -> list[dict]:
        """Extract named entities using GLiNER.

        Returns:
            List of dicts with text, label, start, end
        """
        self._load_gliner_model()

        # GLiNER predict_entities expects labels as list of strings
        predictions = self._gliner.predict_entities(
            text,
            self.config.gliner_entity_labels,
            threshold=self.config.gliner_threshold,
        )

        entities = []
        for pred in predictions:
            entities.append(
                {
                    "text": pred["text"],
                    "label": pred["label"].upper(),  # Normalize to uppercase
                    "start": pred["start"],
                    "end": pred["end"],
                }
            )
        return entities

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text using configured backend."""
        if self.config.model == "gliner":
            return self._extract_entities_gliner(text)
        return self._extract_entities_spacy(text)

    def _entity_in_context(
        self, entity_text: str, context_entities: list[dict]
    ) -> tuple[bool, float]:
        """Check if entity exists in context entities.

        Returns:
            (found, similarity_score) tuple
        """
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
            AugmentationResult with hallucination score (0 = supported, 1 = hallucinated).
            Score is ratio of unverified entities.
        """
        # Extract entities from context
        context_text = " ".join(context)
        context_entities = self._extract_entities(context_text)

        # Extract entities from answer
        answer_entities = self._extract_entities(answer)

        # No entities in answer = nothing to verify = no evidence of hallucination
        # Score 0.0 = supported (no hallucination), is_active=False excludes from aggregation
        if not answer_entities:
            return AugmentationResult(
                score=0.0,  # 0.0 = supported/no hallucination (absence of entities is not hallucination)
                evidence={
                    "entities_checked": 0,
                    "entities_verified": 0,
                    "context_entities": len(context_entities),
                },
                details={"answer_entities": 0, "context_entities": len(context_entities)},
                flagged_spans=[],
                is_active=False,  # Excluded from weighted average - nothing to verify
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

        # Calculate hallucination score (ratio of unverified entities)
        # 0 = all verified (supported), 1 = none verified (hallucinated)
        unverified_ratio = 1.0 - (verified_count / len(answer_entities))

        return AugmentationResult(
            score=unverified_ratio,  # 0 = supported, 1 = hallucinated
            evidence={
                "entities_checked": len(answer_entities),
                "entities_verified": verified_count,
                "context_entities": len(context_entities),
            },
            details={
                "answer_entities": len(answer_entities),
                "verified_entities": verified_count,
                "context_entities": len(context_entities),
                "entities": entity_details,
            },
            flagged_spans=flagged_spans,
            is_active=True,  # Entities found and verified
        )
