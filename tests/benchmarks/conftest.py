"""Pytest configuration for benchmarks."""

import pytest

from tests.benchmarks.config import BenchmarkConfig, DatasetConfig


def pytest_addoption(parser):
    """Add benchmark-specific command line options."""
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Quick mode: 100 samples, no CI computation",
    )
    parser.addoption(
        "--full",
        action="store_true",
        default=False,
        help="Full mode: all samples, full metrics",
    )
    parser.addoption(
        "--dataset",
        action="store",
        default="ragtruth",
        help="Dataset to benchmark (ragtruth, halueval_qa, etc.)",
    )
    parser.addoption(
        "--limit",
        action="store",
        type=int,
        default=None,
        help="Limit samples per dataset",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")


@pytest.fixture(scope="session")
def benchmark_config(request) -> BenchmarkConfig:
    """Create benchmark configuration from command line options."""
    quick = request.config.getoption("--quick")
    full = request.config.getoption("--full")
    dataset = request.config.getoption("--dataset")
    limit = request.config.getoption("--limit")

    # Determine limit based on mode
    if quick:
        limit = limit or 100
    elif not full and limit is None:
        limit = 100  # Default to quick mode if neither specified

    config = BenchmarkConfig(
        quick=quick,
        full=full,
        datasets=[DatasetConfig(name=dataset, limit=limit)],
    )

    if quick:
        config = config.apply_quick_mode()

    return config


@pytest.fixture(scope="session")
def sample_limit(request) -> int | None:
    """Get sample limit from config."""
    quick = request.config.getoption("--quick")
    limit = request.config.getoption("--limit")

    if quick:
        return limit or 100
    return limit


@pytest.fixture(scope="module")
def ragtruth_samples(sample_limit):
    """Load RAGTruth samples for benchmarking."""
    from tests.benchmarks.data_adapters import RAGTruthAdapter

    adapter = RAGTruthAdapter()
    return adapter.load(limit=sample_limit)


@pytest.fixture(scope="module")
def halueval_qa_samples(sample_limit):
    """Load HaluEval QA samples for benchmarking."""
    from tests.benchmarks.data_adapters import HaluEvalAdapter

    adapter = HaluEvalAdapter(subset="qa_samples")
    return adapter.load(limit=sample_limit)


@pytest.fixture(scope="module")
def all_samples(sample_limit):
    """Load all benchmark samples."""
    from tests.benchmarks.data_adapters.base import load_all_datasets

    return load_all_datasets(limit=sample_limit)


# Component fixtures
@pytest.fixture(scope="module")
def lexical_calculator():
    """Create lexical overlap calculator."""
    from lettucedetect.utils.lexical import LexicalOverlapCalculator

    calc = LexicalOverlapCalculator()
    calc.preload()
    return calc


@pytest.fixture(scope="module")
def model2vec_encoder():
    """Create Model2Vec encoder."""
    from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder

    encoder = Model2VecEncoder()
    encoder.preload()
    return encoder


@pytest.fixture(scope="module")
def stage1_detector():
    """Create Stage 1 detector with all augmentations."""
    from lettucedetect.detectors.stage1.detector import Stage1Detector

    detector = Stage1Detector(augmentations=["ner", "numeric", "lexical"])
    detector.warmup()
    return detector


@pytest.fixture(scope="module")
def stage2_detector():
    """Create Stage 2 detector."""
    from lettucedetect.detectors.stage2.detector import Stage2Detector

    detector = Stage2Detector()
    detector.warmup()
    return detector


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clear CUDA cache between tests."""
    yield
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
