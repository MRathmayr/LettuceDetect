"""Benchmark tests for combined Stages 1+2 cascade."""

import pytest

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)
from tests.benchmarks.core.memory import MemoryTracker


@pytest.fixture(scope="module")
def cascade_detector():
    """Create cascade detector with Stages 1 and 2."""
    from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage2Config
    from lettucedetect.detectors.cascade import CascadeDetector

    config = CascadeConfig(
        stages=[1, 2],
        stage1=Stage1Config(augmentations=["ner", "numeric", "lexical"]),
        stage2=Stage2Config(components=["ncs", "nli"]),
    )

    detector = CascadeDetector(config)
    # Warmup both stages
    detector._stage1.warmup()
    detector._stage2.warmup()
    return detector


@pytest.mark.benchmark
@pytest.mark.gpu
class TestCascadeStages12Benchmark:
    """Benchmarks for combined Stages 1+2 cascade."""

    def test_cascade_latency(self, cascade_detector, ragtruth_samples, benchmark_config):
        """Benchmark cascade pipeline latency."""
        timer = BenchmarkTimer(sync_cuda=True)

        samples = ragtruth_samples[:50] if len(ragtruth_samples) > 50 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                cascade_detector.predict(
                    context=sample.context,
                    answer=sample.response or "",
                    question=sample.question,
                    output_format="spans",
                )

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    cascade_detector.predict(
                        context=sample.context,
                        answer=sample.response or "",
                        question=sample.question,
                        output_format="spans",
                    )

        stats = timer.get_stats()
        # Cascade should average between Stage 1 and Stage 1+2
        # Most samples should be resolved at Stage 1
        assert stats.mean_ms < 300, f"Cascade too slow: {stats.mean_ms:.2f}ms"

    def test_cascade_accuracy(self, cascade_detector, ragtruth_samples, benchmark_config):
        """Benchmark cascade accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=True)
        memory_tracker = MemoryTracker()

        with memory_tracker.track():
            for sample in ragtruth_samples:
                if not sample.context or not sample.response:
                    continue

                with timer.measure():
                    result = cascade_detector.predict(
                        context=sample.context,
                        answer=sample.response,
                        question=sample.question,
                        output_format="detailed",
                    )

                # Extract final score from detailed result
                if isinstance(result, dict):
                    final_score = result.get("scores", {}).get("final_score", 0.0)
                    spans = result.get("spans", [])
                else:
                    spans = result
                    final_score = max(s.get("confidence", 0.5) for s in spans) if spans else 0.0

                predictions.append(
                    PredictionResult(
                        sample_id=sample.id,
                        ground_truth=sample.ground_truth,
                        predicted_score=final_score,
                        predicted_label=1 if final_score >= 0.5 else 0,
                        latency_ms=timer.last_ms,
                        component_name="cascade_12",
                    )
                )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()
        memory = memory_tracker.get_stats()

        results = BenchmarkResults(
            component="cascade_12",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            memory=memory,
            config={"stages": [1, 2]},
        )

        assert metrics.n_samples > 0, "No predictions made"

        print(f"\n{'='*60}")
        print(f"Cascade (Stages 1+2) Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 else "F1: N/A")
        print(f"Optimal F1: {metrics.optimal_f1:.3f}" if metrics.optimal_f1 else "Optimal F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
        if memory.gpu_peak_mb:
            print(f"GPU Peak: {memory.gpu_peak_mb:.1f}MB")
