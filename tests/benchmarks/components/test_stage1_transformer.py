"""Benchmark tests for TransformerDetector."""

import pytest

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)
from tests.benchmarks.core.memory import MemoryTracker


@pytest.fixture(scope="module")
def transformer_detector():
    """Create TransformerDetector instance."""
    from lettucedetect.detectors.transformer import TransformerDetector

    detector = TransformerDetector(
        model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
    )
    detector.warmup()
    return detector


@pytest.mark.benchmark
@pytest.mark.gpu
class TestTransformerDetectorBenchmark:
    """Benchmarks for TransformerDetector (ModernBERT)."""

    def test_transformer_latency(self, transformer_detector, ragtruth_samples, benchmark_config):
        """Benchmark TransformerDetector latency."""
        timer = BenchmarkTimer(sync_cuda=True)

        samples = ragtruth_samples[:50] if len(ragtruth_samples) > 50 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                transformer_detector.predict(
                    context=sample.context,
                    answer=sample.response or "",
                    question=sample.question,
                    output_format="spans",
                )

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    transformer_detector.predict(
                        context=sample.context,
                        answer=sample.response or "",
                        question=sample.question,
                        output_format="spans",
                    )

        stats = timer.get_stats()
        assert stats.mean_ms < 150, f"Transformer too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 200, f"Transformer P95 too slow: {stats.p95_ms:.2f}ms"

    def test_transformer_accuracy(self, transformer_detector, ragtruth_samples, benchmark_config):
        """Benchmark TransformerDetector accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=True)
        memory_tracker = MemoryTracker()

        with memory_tracker.track():
            for sample in ragtruth_samples:
                if not sample.context or not sample.response:
                    continue

                with timer.measure():
                    spans = transformer_detector.predict(
                        context=sample.context,
                        answer=sample.response,
                        question=sample.question,
                        output_format="spans",
                    )

                # Convert span predictions to response-level score
                # If any hallucinated spans, use max confidence
                if spans:
                    hallucination_score = max(s.get("confidence", 0.5) for s in spans)
                else:
                    hallucination_score = 0.0

                predictions.append(
                    PredictionResult(
                        sample_id=sample.id,
                        ground_truth=sample.ground_truth,
                        predicted_score=hallucination_score,
                        predicted_label=1 if spans else 0,
                        latency_ms=timer.last_ms,
                        component_name="transformer",
                    )
                )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()
        memory = memory_tracker.get_stats()

        results = BenchmarkResults(
            component="transformer",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            memory=memory,
            config={"model": "KRLabsOrg/lettucedect-base-modernbert-en-v1"},
        )

        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 150, f"Too slow: {timing.mean_ms:.2f}ms"

        print(f"\n{'='*60}")
        print(f"TransformerDetector Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc is not None else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 is not None else "F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
        if memory.gpu_peak_mb:
            print(f"GPU Peak: {memory.gpu_peak_mb:.1f}MB")
