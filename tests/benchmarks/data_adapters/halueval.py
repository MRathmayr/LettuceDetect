"""HaluEval dataset adapter.

HaluEval provides pre-generated responses with hallucination labels
across three domains: QA, Dialogue, and Summarization.
"""

from typing import Literal

from tests.benchmarks.data_adapters.base import BenchmarkSample, DatasetAdapter

try:
    from datasets import load_dataset as hf_load_dataset

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    hf_load_dataset = None


class HaluEvalAdapter(DatasetAdapter):
    """Adapter for HaluEval benchmark dataset."""

    def __init__(
        self,
        subset: Literal["qa_samples", "dialogue_samples", "summarization_samples"] = "qa_samples",
    ):
        """Initialize HaluEval adapter.

        Args:
            subset: Which HaluEval subset to load
        """
        self.subset = subset

    @property
    def name(self) -> str:
        return f"halueval_{self.subset.replace('_samples', '')}"

    def load(self, limit: int | None = None) -> list[BenchmarkSample]:
        """Load HaluEval samples.

        HaluEval provides pre-generated responses with hallucination labels.
        Supports all 3 subsets:
        - qa_samples: knowledge, question, answer, hallucination
        - dialogue_samples: knowledge, dialogue_history, response, hallucination
        - summarization_samples: document, summary, hallucination

        HuggingFace: pminervini/HaluEval
        """
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library required for HaluEval. Install with: pip install datasets"
            )

        ds = hf_load_dataset("pminervini/HaluEval", self.subset, split="data")

        samples = []
        for idx, item in enumerate(ds):
            if limit is not None and idx >= limit:
                break

            # Extract fields based on subset type
            if self.subset == "qa_samples":
                question = item.get("question", "")
                knowledge = item.get("knowledge", "")
                response = item.get("answer", "")
                context = [knowledge] if knowledge else None
                task_type = "qa"

            elif self.subset == "dialogue_samples":
                dialogue_history = item.get("dialogue_history") or ""
                knowledge = item.get("knowledge", "")
                response = item.get("response", "")
                # Dialogue history becomes the "question"
                if isinstance(dialogue_history, str):
                    question = dialogue_history
                elif dialogue_history:
                    question = "\n".join(dialogue_history)
                else:
                    question = ""
                context = [knowledge] if knowledge else None
                task_type = "dialogue"

            elif self.subset == "summarization_samples":
                document = item.get("document", "")
                response = item.get("summary", "")
                question = "Summarize the following document."
                context = [document] if document else None
                task_type = "summarization"

            else:
                raise ValueError(f"Unknown HaluEval subset: {self.subset}")

            hallucination = item.get("hallucination", "").lower()
            ground_truth = 1 if hallucination == "yes" else 0

            samples.append(
                BenchmarkSample(
                    id=f"halueval_{self.subset}_{idx}",
                    benchmark=self.name,
                    question=question,
                    context=context,
                    response=response,
                    ground_truth=ground_truth,
                    task_type=task_type,
                    hallucination_spans=None,
                )
            )

        return samples
