"""Shared fixtures for pytest tests."""

import gc
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo

# Enable fallback to eager mode when torch.compile fails (e.g., on GPUs without Triton support)
torch._dynamo.config.suppress_errors = True


@pytest.fixture(autouse=True)
def clear_cuda_cache():
    """Clear CUDA cache after each test to prevent OOM errors."""
    yield
    if torch.cuda.is_available():
        gc.collect()                    # FIRST: release Python refs to GPU tensors
        torch.cuda.synchronize()        # SECOND: wait for GPU ops to complete
        torch.cuda.empty_cache()        # THIRD: free unreferenced GPU memory


@pytest.fixture(scope="class", autouse=True)
def clear_cuda_between_classes():
    """Aggressively clear CUDA memory between test classes."""
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# =============================================================================
# Real-world RAG Fixtures
# =============================================================================


@pytest.fixture
def sample_context():
    """Basic sample context for testing."""
    return ["The capital of France is Paris. It has a population of 2.1 million."]


@pytest.fixture
def sample_answer_supported():
    """Answer that is fully supported by context."""
    return "Paris is the capital of France."


@pytest.fixture
def sample_answer_hallucinated():
    """Answer containing hallucinated information."""
    return "The capital of France is Lyon."


@pytest.fixture
def financial_report_context():
    """Realistic financial report context for RAG testing."""
    return [
        "Acme Corporation reported Q3 2024 earnings of $4.2 billion, representing a "
        "15% increase year-over-year. The company's revenue was driven by strong "
        "performance in its cloud services division, which generated $1.8 billion. "
        "CEO John Smith announced plans to expand operations to Asia in 2025. "
        "The stock price closed at $142.50 on December 15, 2024."
    ]


@pytest.fixture
def financial_answer_supported():
    """Financially accurate answer based on context."""
    return (
        "Acme Corporation had Q3 2024 earnings of $4.2 billion, up 15% from the "
        "previous year. Their cloud services division was the main driver with "
        "$1.8 billion in revenue."
    )


@pytest.fixture
def financial_answer_hallucinated_numbers():
    """Answer with fabricated financial numbers."""
    return (
        "Acme Corporation reported Q3 2024 earnings of $5.8 billion, which was a "
        "22% increase year-over-year. The cloud division generated $2.3 billion."
    )


@pytest.fixture
def financial_answer_hallucinated_entities():
    """Answer with fabricated entity names."""
    return (
        "Acme Corporation reported strong earnings. CEO Michael Johnson announced "
        "the acquisition of TechCorp for $500 million."
    )


@pytest.fixture
def medical_context():
    """Medical/scientific context for testing entity accuracy."""
    return [
        "Type 2 diabetes affects approximately 37 million Americans. The condition "
        "is characterized by insulin resistance and is often managed with metformin "
        "as a first-line treatment. According to the American Diabetes Association, "
        "patients should maintain an HbA1c level below 7%. Dr. Sarah Chen at Johns "
        "Hopkins published a 2023 study showing that combined diet and exercise "
        "reduced HbA1c by 1.2% on average."
    ]


@pytest.fixture
def medical_answer_supported():
    """Medically accurate answer."""
    return (
        "Type 2 diabetes affects about 37 million Americans. Metformin is commonly "
        "used as a first-line treatment, and patients should aim for an HbA1c below 7%."
    )


@pytest.fixture
def medical_answer_hallucinated():
    """Answer with fabricated medical claims."""
    return (
        "Type 2 diabetes affects 50 million Americans. The new drug Glucomax "
        "has replaced metformin as the standard treatment. Dr. James Wilson at "
        "Stanford found that HbA1c should be below 5%."
    )


@pytest.fixture
def geographic_context():
    """Geographic context for testing location accuracy."""
    return [
        "Mount Everest is located in the Himalayas on the border between Nepal and "
        "Tibet. It stands at 8,849 meters (29,032 feet), making it the highest peak "
        "on Earth. The first confirmed summit was achieved by Edmund Hillary and "
        "Tenzing Norgay on May 29, 1953. The mountain is known locally as Sagarmatha "
        "in Nepal and Chomolungma in Tibet."
    ]


@pytest.fixture
def geographic_answer_supported():
    """Geographically accurate answer."""
    return (
        "Mount Everest, at 8,849 meters, is the world's highest peak. It was first "
        "climbed by Edmund Hillary and Tenzing Norgay in 1953."
    )


@pytest.fixture
def geographic_answer_hallucinated():
    """Answer with geographic errors."""
    return (
        "Mount Everest stands at 9,100 meters and is located in Pakistan. It was "
        "first climbed by George Mallory in 1924."
    )


@pytest.fixture
def multi_passage_context():
    """Multiple context passages typical of RAG systems."""
    return [
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in "
        "April 1976. The company's headquarters is located in Cupertino, California.",
        "In 2023, Apple reported annual revenue of $383 billion. The iPhone remains "
        "the company's most profitable product line, generating over 50% of revenue.",
        "Tim Cook has served as CEO since August 2011, following Steve Jobs' resignation "
        "due to health issues. Cook previously served as Apple's Chief Operating Officer.",
    ]


@pytest.fixture
def multi_passage_answer_supported():
    """Answer accurately drawing from multiple passages."""
    return (
        "Apple was founded in 1976 by Steve Jobs and others. Tim Cook became CEO in "
        "2011. The company reported $383 billion in revenue in 2023, with iPhone "
        "generating over half of it."
    )


@pytest.fixture
def multi_passage_answer_partial_hallucination():
    """Answer with some accurate and some hallucinated information."""
    return (
        "Apple was founded in 1976 by Steve Jobs. Tim Cook became CEO in 2015 after "
        "Jobs retired. The company reported $420 billion in revenue in 2023."
    )


@pytest.fixture
def empty_context():
    """Empty context list edge case."""
    return []


@pytest.fixture
def empty_answer():
    """Empty answer edge case."""
    return ""


# =============================================================================
# Entity-specific fixtures for NER testing
# =============================================================================


@pytest.fixture
def context_with_persons():
    """Context with person entities for NER testing."""
    return [
        "Albert Einstein developed the theory of relativity while working at the "
        "Swiss Patent Office in Bern. His wife, Mileva Maric, was also a physicist."
    ]


@pytest.fixture
def context_with_organizations():
    """Context with organization entities."""
    return [
        "Microsoft acquired LinkedIn for $26.2 billion in 2016. The deal was approved "
        "by the European Commission after a thorough antitrust review."
    ]


@pytest.fixture
def context_with_locations():
    """Context with location entities."""
    return [
        "The Great Wall of China extends from Dandong in Liaoning Province to Lop Lake "
        "in Xinjiang. Its total length is approximately 21,196 kilometers."
    ]


@pytest.fixture
def context_with_dates():
    """Context with temporal entities."""
    return [
        "World War II ended on September 2, 1945, when Japan formally surrendered. "
        "The war had begun with Germany's invasion of Poland on September 1, 1939."
    ]


# =============================================================================
# Numeric-specific fixtures
# =============================================================================


@pytest.fixture
def context_with_percentages():
    """Context with percentage values."""
    return [
        "The company's market share grew from 23% to 31% in 2024. Customer retention "
        "rate improved by 8 percentage points to reach 92%."
    ]


@pytest.fixture
def context_with_currencies():
    """Context with monetary values."""
    return [
        "The property sold for $2.5 million, which was $300,000 above the asking price. "
        "Annual maintenance costs are estimated at $45,000."
    ]


@pytest.fixture
def context_with_measurements():
    """Context with physical measurements."""
    return [
        "The new bridge spans 2.7 kilometers and stands 65 meters above the water. "
        "It can support loads up to 50,000 tonnes."
    ]


# =============================================================================
# Mock fixtures
# =============================================================================


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [101, 102, 103, 104, 105]

    # Mock tokenizer call to return encoding
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102, 103, 104, 105, 106, 107, 108]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]]),
        "offset_mapping": torch.tensor(
            [
                [0, 0],  # [CLS]
                [0, 4],  # "This"
                [5, 7],  # "is"
                [8, 9],  # "a"
                [10, 16],  # "prompt"
                [0, 0],  # [SEP]
                [0, 4],  # "This"
                [5, 12],  # "answer"
            ]
        ),
    }

    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]])
    model.return_value = mock_output
    return model


# =============================================================================
# Stage1Detector fixtures with automatic GPU cleanup
# =============================================================================


@pytest.fixture
def stage1_detector_no_aug():
    """Stage1Detector without augmentations - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=[])
    yield detector
    del detector


@pytest.fixture
def stage1_detector_ner():
    """Stage1Detector with NER augmentation - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=["ner"])
    yield detector
    del detector


@pytest.fixture
def stage1_detector_numeric():
    """Stage1Detector with numeric augmentation - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=["numeric"])
    yield detector
    del detector


@pytest.fixture
def stage1_detector_lexical():
    """Stage1Detector with lexical augmentation - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=["lexical"])
    yield detector
    del detector


@pytest.fixture
def stage1_detector_numeric_lexical():
    """Stage1Detector with numeric and lexical augmentations - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=["numeric", "lexical"])
    yield detector
    del detector


@pytest.fixture
def stage1_detector_all():
    """Stage1Detector with all augmentations - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=["ner", "numeric", "lexical"])
    yield detector
    del detector


@pytest.fixture
def stage1_detector_routing():
    """Stage1Detector with routing config for escalation tests - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector, AggregationConfig

    agg_config = AggregationConfig(
        threshold_high=0.7,
        threshold_low=0.3,
    )
    detector = Stage1Detector(
        augmentations=["numeric", "lexical"],
        aggregation_config=agg_config,
    )
    yield detector
    del detector


@pytest.fixture
def stage1_detector_ner_numeric():
    """Stage1Detector with NER and numeric augmentations - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=["ner", "numeric"])
    yield detector
    del detector


@pytest.fixture
def stage1_detector_ner_lexical():
    """Stage1Detector with NER and lexical augmentations - auto cleanup."""
    from lettucedetect.detectors.stage1 import Stage1Detector

    detector = Stage1Detector(augmentations=["ner", "lexical"])
    yield detector
    del detector
