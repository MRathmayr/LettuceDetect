"""Benchmark tests for LexicalOverlapCalculator."""

import pytest

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)
from tests.benchmarks.core.memory import MemoryTracker


@pytest.mark.benchmark
class TestLexicalOverlapBenchmark:
    """Benchmarks for LexicalOverlapCalculator."""

    def test_lexical_latency(self, lexical_calculator, ragtruth_samples, benchmark_config):
        """Benchmark lexical overlap calculation latency."""
        timer = BenchmarkTimer(sync_cuda=False)  # CPU only

        # Take subset for latency measurement
        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                lexical_calculator.score(
                    context=sample.context,
                    answer=sample.response or "",
                    question=sample.question,
                    token_predictions=None,
                )

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    lexical_calculator.score(
                        context=sample.context,
                        answer=sample.response or "",
                        question=sample.question,
                        token_predictions=None,
                    )

        stats = timer.get_stats()
        assert stats.mean_ms < 10, f"Lexical overlap too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 20, f"Lexical overlap P95 too slow: {stats.p95_ms:.2f}ms"

    def test_lexical_accuracy(self, lexical_calculator, ragtruth_samples, benchmark_config):
        """Benchmark lexical overlap accuracy on RAGTruth."""
        predictions = []

        timer = BenchmarkTimer(sync_cuda=False)

        for sample in ragtruth_samples:
            if not sample.context or not sample.response:
                continue

            with timer.measure():
                result = lexical_calculator.score(
                    context=sample.context,
                    answer=sample.response,
                    question=sample.question,
                    token_predictions=None,
                )

            predictions.append(
                PredictionResult(
                    sample_id=sample.id,
                    ground_truth=sample.ground_truth,
                    predicted_score=result.score,
                    predicted_label=1 if result.score >= 0.5 else 0,
                    latency_ms=timer.last_ms,
                    component_name="lexical",
                )
            )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()

        results = BenchmarkResults(
            component="lexical",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            config={"use_stemming": True, "ngram_range": (1, 2)},
        )

        # Assertions
        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 10, f"Too slow: {timing.mean_ms:.2f}ms"

        # Print summary
        print(f"\n{'='*60}")
        print(f"Lexical Overlap Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 else "F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
