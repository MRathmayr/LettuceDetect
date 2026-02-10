"""Benchmark tests for Stage 3: Hallu Probe detector (Qwen 3B).

Stage 3 uses a causal LM + trained sklearn probe on hidden states
to predict P(hallucinated). Requires GPU for quantized inference.

Run:
    # Quick (100 samples, HaluEval QA only)
    pytest tests/benchmarks/stages/test_stage3_3b.py --quick -v -s

    # Full (all samples, both datasets)
    pytest tests/benchmarks/stages/test_stage3_3b.py --full -v -s
"""

import pytest

from tests.benchmarks.core import (
    BenchmarkTimer,
)
from tests.benchmarks.stages._stage3_helpers import _run_stage3_accuracy

_3B_CONFIG = {"model": "Qwen/Qwen2.5-3B-Instruct", "layer": -15, "position": "mean"}


@pytest.mark.benchmark
@pytest.mark.gpu
class TestStage3BenchmarkQwen3B:
    """Benchmarks for Stage 3 with Qwen 2.5 3B hallu probe."""

    def test_stage3_latency(self, stage3_detector_3b, halueval_qa_samples, benchmark_config):
        """Benchmark Stage 3 (3B) latency on HaluEval QA."""
        timer = BenchmarkTimer(sync_cuda=True)
        samples = halueval_qa_samples[:50] if len(halueval_qa_samples) > 50 else halueval_qa_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context and sample.response:
                stage3_detector_3b.predict(
                    context=sample.context,
                    answer=sample.response,
                    question=sample.question,
                    output_format="spans",
                )

        for sample in samples:
            if not sample.context or not sample.response:
                continue
            with timer.measure():
                stage3_detector_3b.predict(
                    context=sample.context,
                    answer=sample.response,
                    question=sample.question,
                    output_format="spans",
                )

        stats = timer.get_stats()
        print(f"\n{'='*60}")
        print(f"Stage 3 (Qwen 3B) Latency - HaluEval QA")
        print(f"{'='*60}")
        print(f"Mean: {stats.mean_ms:.2f}ms, P95: {stats.p95_ms:.2f}ms")
        print(f"Min: {stats.min_ms:.2f}ms, Max: {stats.max_ms:.2f}ms")
        # Stage 3 target: 50-200ms per sample
        assert stats.mean_ms < 500, f"Stage 3 too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 750, f"Stage 3 P95 too slow: {stats.p95_ms:.2f}ms"

    def test_stage3_accuracy_halueval_qa(
        self, stage3_detector_3b, halueval_qa_samples, benchmark_config
    ):
        """Benchmark Stage 3 (3B) accuracy on HaluEval QA."""
        _run_stage3_accuracy(
            stage3_detector_3b, halueval_qa_samples,
            "stage3_hallu_probe_3b", "halueval_qa", _3B_CONFIG, benchmark_config,
        )

    def test_stage3_accuracy_ragtruth(
        self, stage3_detector_3b, ragtruth_samples, benchmark_config
    ):
        """Benchmark Stage 3 (3B) accuracy on RAGTruth."""
        _run_stage3_accuracy(
            stage3_detector_3b, ragtruth_samples,
            "stage3_hallu_probe_3b", "ragtruth", _3B_CONFIG, benchmark_config,
        )
