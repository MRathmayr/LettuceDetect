"""Benchmark tests for NumericValidator."""

import pytest

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)


@pytest.fixture(scope="module")
def numeric_validator():
    """Create NumericValidator instance."""
    from lettucedetect.detectors.stage1.augmentations.numeric_validator import (
        NumericValidator,
    )

    validator = NumericValidator()
    validator.preload()
    return validator


@pytest.mark.benchmark
class TestNumericValidatorBenchmark:
    """Benchmarks for NumericValidator."""

    def test_numeric_latency(self, numeric_validator, ragtruth_samples, benchmark_config):
        """Benchmark numeric validation latency."""
        timer = BenchmarkTimer(sync_cuda=False)  # CPU only

        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                numeric_validator.score(
                    context=sample.context,
                    answer=sample.response or "",
                    question=sample.question,
                    token_predictions=None,
                )

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    numeric_validator.score(
                        context=sample.context,
                        answer=sample.response or "",
                        question=sample.question,
                        token_predictions=None,
                    )

        stats = timer.get_stats()
        assert stats.mean_ms < 5, f"Numeric validation too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 10, f"Numeric validation P95 too slow: {stats.p95_ms:.2f}ms"

    def test_numeric_accuracy(self, numeric_validator, ragtruth_samples, benchmark_config):
        """Benchmark numeric validator accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=False)

        for sample in ragtruth_samples:
            if not sample.context or not sample.response:
                continue

            with timer.measure():
                result = numeric_validator.score(
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
                    component_name="numeric",
                )
            )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()

        results = BenchmarkResults(
            component="numeric",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            config={},
        )

        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 5, f"Too slow: {timing.mean_ms:.2f}ms"

        print(f"\n{'='*60}")
        print(f"Numeric Validator Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc is not None else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 is not None else "F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
