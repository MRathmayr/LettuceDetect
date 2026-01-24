"""RAGTruth dataset adapter.

RAGTruth is the primary benchmark for RAG hallucination detection.
Uses wandb/RAGTruth-processed which has response-level labels.
"""

from tests.benchmarks.data_adapters.base import BenchmarkSample, DatasetAdapter

try:
    from datasets import load_dataset as hf_load_dataset

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    hf_load_dataset = None


class RAGTruthAdapter(DatasetAdapter):
    """Adapter for RAGTruth benchmark dataset."""

    @property
    def name(self) -> str:
        return "ragtruth"

    def load(self, limit: int | None = None) -> list[BenchmarkSample]:
        """Load RAGTruth samples.

        RAGTruth-processed has hallucination annotations (evident_conflict, baseless_info).
        We convert to binary: any hallucination type = 1, none = 0.

        HuggingFace: wandb/RAGTruth-processed
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library required for RAGTruth. Install with: pip install datasets"
            )

        ds = hf_load_dataset("wandb/RAGTruth-processed", split="test")

        samples = []
        for idx, item in enumerate(ds):
            if limit is not None and idx >= limit:
                break

            question = item.get("query", "")
            response = item.get("output", "")
            context_str = item.get("context", "")

            # Hallucination labels - check both types
            labels = item.get("hallucination_labels_processed", {})
            evident_conflict = labels.get("evident_conflict", 0) if labels else 0
            baseless_info = labels.get("baseless_info", 0) if labels else 0
            ground_truth = 1 if (evident_conflict or baseless_info) else 0

            # Context is a single string in this dataset
            context = [context_str] if context_str else None

            samples.append(
                BenchmarkSample(
                    id=f"ragtruth_{idx}",
                    benchmark="ragtruth",
                    question=question,
                    context=context,
                    response=response,
                    ground_truth=ground_truth,
                    hallucination_spans=None,  # RAGTruth-processed is response-level
                )
            )

        return samples
