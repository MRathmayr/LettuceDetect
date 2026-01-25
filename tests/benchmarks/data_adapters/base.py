"""Base dataset adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class BenchmarkSample:
    """Unified sample format for all benchmarks.

    Adapted from read-training/benchmark/dataset_loaders.py.
    """

    id: str
    benchmark: str  # "ragtruth", "halueval_qa", etc.
    question: str
    context: list[str] | None  # RAG context documents
    response: str | None  # Pre-generated response (None = needs generation)
    ground_truth: int  # 0=factual, 1=hallucination

    # Optional fields for span-level evaluation
    hallucination_spans: list[dict] | None = None  # [{"start": 0, "end": 10, "type": "..."}]

    # TruthfulQA-specific
    correct_answers: list[str] | None = None
    incorrect_answers: list[str] | None = None


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return dataset name."""
        ...

    @abstractmethod
    def load(self, limit: int | None = None) -> list[BenchmarkSample]:
        """Load dataset samples.

        Args:
            limit: Maximum samples to load (None = all)

        Returns:
            List of BenchmarkSample objects
        """
        ...

    def iterate(self, limit: int | None = None) -> Iterator[BenchmarkSample]:
        """Iterate over samples.

        Default implementation loads all into memory. Override for streaming.
        """
        yield from self.load(limit=limit)

    def __len__(self) -> int:
        """Return dataset size (loads full dataset)."""
        return len(self.load())


def load_dataset_adapter(name: str) -> DatasetAdapter:
    """Factory function to get dataset adapter by name.

    Args:
        name: Dataset name ("ragtruth", "halueval_qa", etc.)

    Returns:
        DatasetAdapter instance
    """
    from tests.benchmarks.data_adapters.halueval import HaluEvalAdapter
    from tests.benchmarks.data_adapters.ragtruth import RAGTruthAdapter

    adapters = {
        "ragtruth": RAGTruthAdapter,
        "halueval_qa": lambda: HaluEvalAdapter(subset="qa_samples"),
    }

    if name not in adapters:
        valid = ", ".join(adapters.keys())
        raise ValueError(f"Unknown dataset: {name}. Valid options: {valid}")

    factory = adapters[name]
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory()


def load_all_datasets(limit: int | None = None) -> list[BenchmarkSample]:
    """Load all benchmark datasets.

    Args:
        limit: Maximum samples per dataset (None = all)

    Returns:
        Combined list of samples from all datasets
    """
    samples = []
    for name in ["ragtruth", "halueval_qa"]:
        adapter = load_dataset_adapter(name)
        samples.extend(adapter.load(limit=limit))
    return samples
