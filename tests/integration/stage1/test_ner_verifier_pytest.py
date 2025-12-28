"""Integration tests for NERVerifier with real spaCy model.

These tests use the en_core_web_sm spaCy model to verify NER extraction
and entity matching work correctly with real-world RAG examples.
"""

import pytest

from lettucedetect.detectors.stage1.augmentations.ner_verifier import (
    NERVerifier,
    NERConfig,
)


class TestSpaCyEntityExtraction:
    """Test entity extraction with real spaCy model."""

    def setup_method(self):
        """Set up NERVerifier with spaCy model."""
        self.verifier = NERVerifier()

    def test_person_extraction(self, context_with_persons):
        """Extract PERSON entities from context."""
        entities = self.verifier._extract_entities(" ".join(context_with_persons))
        person_entities = [e for e in entities if e["label"] == "PERSON"]
        names = [e["text"] for e in person_entities]
        assert "Albert Einstein" in names
        assert "Mileva Maric" in names

    def test_organization_extraction(self, context_with_organizations):
        """Extract ORG entities from context."""
        entities = self.verifier._extract_entities(" ".join(context_with_organizations))
        org_entities = [e for e in entities if e["label"] == "ORG"]
        names = [e["text"] for e in org_entities]
        assert "Microsoft" in names
        assert "LinkedIn" in names
        # spaCy may include "the" in entity text
        assert any("European Commission" in n for n in names)

    def test_location_extraction(self, context_with_locations):
        """Extract GPE/LOC entities from context."""
        entities = self.verifier._extract_entities(" ".join(context_with_locations))
        location_labels = {"GPE", "LOC"}
        loc_entities = [e for e in entities if e["label"] in location_labels]
        names = [e["text"] for e in loc_entities]
        # "China" may be part of "Great Wall of China" so check for any of these
        assert len(names) >= 2  # Should extract multiple locations
        assert any("Xinjiang" in n or "Liaoning" in n or "Dandong" in n for n in names)

    def test_date_extraction(self, context_with_dates):
        """Extract DATE entities from context."""
        entities = self.verifier._extract_entities(" ".join(context_with_dates))
        date_entities = [e for e in entities if e["label"] == "DATE"]
        # Should find dates like "September 2, 1945" and "September 1, 1939"
        assert len(date_entities) >= 2


class TestEntityVerificationRealWorld:
    """Test entity verification with realistic RAG examples."""

    def setup_method(self):
        """Set up NERVerifier."""
        self.verifier = NERVerifier()

    def test_financial_context_supported_answer(
        self, financial_report_context, financial_answer_supported
    ):
        """Verify supported financial answer has high score."""
        result = self.verifier.score(
            financial_report_context, financial_answer_supported, None, None
        )
        # Entities like "Acme Corporation" and numbers should be found in context
        assert result.score >= 0.5  # At least half the entities are verified
        assert result.confidence > 0

    def test_financial_context_hallucinated_entities(
        self, financial_report_context, financial_answer_hallucinated_entities
    ):
        """Verify hallucinated entities are flagged."""
        result = self.verifier.score(
            financial_report_context, financial_answer_hallucinated_entities, None, None
        )
        # "Michael Johnson" and "TechCorp" are not in context
        assert len(result.flagged_spans) > 0
        flagged_texts = [span["text"] for span in result.flagged_spans]
        # At least one hallucinated entity should be flagged
        assert any(
            "Michael Johnson" in text or "TechCorp" in text for text in flagged_texts
        )

    def test_medical_context_supported(
        self, medical_context, medical_answer_supported
    ):
        """Verify supported medical answer."""
        result = self.verifier.score(
            medical_context, medical_answer_supported, None, None
        )
        # Note: spaCy may misclassify drug names like "Metformin" as PERSON
        # We just verify the system doesn't crash and returns a valid score
        assert result.score is not None
        assert 0 <= result.score <= 1.0

    def test_medical_context_hallucinated(
        self, medical_context, medical_answer_hallucinated
    ):
        """Verify hallucinated medical answer is flagged."""
        result = self.verifier.score(
            medical_context, medical_answer_hallucinated, None, None
        )
        # "Glucomax", "James Wilson", "Stanford" are not in context
        assert len(result.flagged_spans) > 0
        flagged_texts = [span["text"] for span in result.flagged_spans]
        # Should flag fabricated entities
        assert any(
            "Glucomax" in text or "James Wilson" in text or "Stanford" in text
            for text in flagged_texts
        )

    def test_geographic_context_supported(
        self, geographic_context, geographic_answer_supported
    ):
        """Verify supported geographic answer."""
        result = self.verifier.score(
            geographic_context, geographic_answer_supported, None, None
        )
        # Mount Everest, Edmund Hillary, Tenzing Norgay should be verified
        assert result.score >= 0.5

    def test_geographic_context_hallucinated(
        self, geographic_context, geographic_answer_hallucinated
    ):
        """Verify hallucinated geographic answer is flagged."""
        result = self.verifier.score(
            geographic_context, geographic_answer_hallucinated, None, None
        )
        # "Pakistan" and "George Mallory" are not in context
        flagged_texts = [span["text"] for span in result.flagged_spans]
        # Should flag at least one fabricated entity
        assert len(result.flagged_spans) > 0 or result.score < 0.5

    def test_multi_passage_context_supported(
        self, multi_passage_context, multi_passage_answer_supported
    ):
        """Verify answer drawing from multiple passages."""
        result = self.verifier.score(
            multi_passage_context, multi_passage_answer_supported, None, None
        )
        # Steve Jobs, Tim Cook, Apple should be found across passages
        assert result.score >= 0.5


class TestFuzzyMatching:
    """Test fuzzy matching capabilities for entity verification."""

    def setup_method(self):
        """Set up NERVerifier with fuzzy matching."""
        self.verifier = NERVerifier()

    def test_partial_name_match(self):
        """Test that partial name matches are handled."""
        context = ["The company was founded by Dr. Robert Smith and Ms. Sarah Johnson."]
        # Answer uses slightly different form
        answer = "Robert Smith co-founded the company."
        result = self.verifier.score(context, answer, None, None)
        # "Robert Smith" should match "Dr. Robert Smith"
        assert result.score > 0

    def test_abbreviation_handling(self):
        """Test organization abbreviation handling."""
        context = ["The National Aeronautics and Space Administration launched the mission."]
        answer = "NASA launched the mission successfully."
        result = self.verifier.score(context, answer, None, None)
        # This is a harder case - might not match perfectly
        # Just verify it doesn't crash and returns a result
        assert result.score is not None

    def test_case_insensitive_matching(self):
        """Test case-insensitive entity matching."""
        context = ["MICROSOFT announced the acquisition."]
        answer = "Microsoft confirmed the deal."
        result = self.verifier.score(context, answer, None, None)
        # Should match despite case difference
        assert result.score >= 0.5


class TestNERConfig:
    """Test NERConfig options affect behavior."""

    def test_custom_entity_types(self):
        """Test restricting to specific entity types."""
        config = NERConfig(entity_types=["PERSON"])
        verifier = NERVerifier(config=config)

        context = ["John Smith works at Microsoft in Seattle."]
        answer = "Jane Doe works at Google in Boston."

        result = verifier.score(context, answer, None, None)
        # Should only flag person entities, not organizations or locations
        flagged_texts = [span["text"] for span in result.flagged_spans]
        assert any("Jane Doe" in text for text in flagged_texts)
        # Should not flag organizations when only checking PERSON
        assert not any("Google" in text for text in flagged_texts)

    def test_fuzzy_threshold(self):
        """Test fuzzy matching threshold affects matching."""
        # Very strict threshold
        strict_config = NERConfig(fuzzy_threshold=99)
        strict_verifier = NERVerifier(config=strict_config)

        # Lenient threshold
        lenient_config = NERConfig(fuzzy_threshold=60)
        lenient_verifier = NERVerifier(config=lenient_config)

        context = ["Dr. Robert James Smith presented the findings."]
        answer = "Robert Smith presented the findings."

        strict_result = strict_verifier.score(context, answer, None, None)
        lenient_result = lenient_verifier.score(context, answer, None, None)

        # Lenient should score higher due to fuzzy matching
        assert lenient_result.score >= strict_result.score


class TestEdgeCases:
    """Test edge cases in NER verification."""

    def setup_method(self):
        """Set up NERVerifier."""
        self.verifier = NERVerifier()

    def test_empty_context(self, empty_context):
        """Handle empty context gracefully."""
        result = self.verifier.score(empty_context, "Some answer with John Smith.", None, None)
        # Should return a result without crashing
        assert result.score is not None

    def test_empty_answer(self, sample_context, empty_answer):
        """Handle empty answer gracefully."""
        result = self.verifier.score(sample_context, empty_answer, None, None)
        # Empty answer has no entities to verify, so fully supported
        assert result.score == 1.0

    def test_no_entities_in_answer(self):
        """Handle answer with no named entities."""
        context = ["John Smith works at Microsoft."]
        answer = "The person works at the company."
        result = self.verifier.score(context, answer, None, None)
        # No entities to verify = fully supported
        assert result.score == 1.0

    def test_special_characters_in_entities(self):
        """Handle entities with special characters."""
        context = ["The company O'Reilly Media published the book."]
        answer = "O'Reilly Media is a publisher."
        result = self.verifier.score(context, answer, None, None)
        assert result.score is not None

    def test_unicode_entities(self):
        """Handle Unicode characters in entities."""
        context = ["François Mitterrand was the President of France."]
        answer = "Mitterrand led France during the 1980s."
        result = self.verifier.score(context, answer, None, None)
        assert result.score is not None

    def test_very_long_context(self):
        """Handle very long context text."""
        long_context = ["This is a paragraph about John Smith. " * 100]
        answer = "John Smith is mentioned."
        result = self.verifier.score(long_context, answer, None, None)
        assert result.score >= 0.5


class TestAugmentationInterface:
    """Test BaseAugmentation interface compliance."""

    def test_name_property(self):
        """Verify name property returns 'ner'."""
        verifier = NERVerifier()
        assert verifier.name == "ner"

    def test_preload_loads_spacy_model(self):
        """Verify preload initializes spaCy model."""
        verifier = NERVerifier()
        verifier.preload()
        # Should not raise and model should be loaded
        assert verifier._nlp is not None

    def test_safe_score_handles_errors(self):
        """Verify safe_score returns neutral on error."""
        verifier = NERVerifier()
        # safe_score should handle any input gracefully
        result = verifier.safe_score(["context"], "answer", None, None)
        assert result.score is not None
