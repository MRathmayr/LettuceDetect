"""Benchmark tests for full Stage 2 pipeline."""

import pytest

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)
from tests.benchmarks.core.memory import MemoryTracker


@pytest.mark.benchmark
@pytest.mark.gpu
class TestStage2Benchmark:
    """Benchmarks for full Stage 2 detector."""

    def test_stage2_latency(self, stage2_detector, ragtruth_samples, benchmark_config):
        """Benchmark Stage 2 pipeline latency."""
        timer = BenchmarkTimer(sync_cuda=True)

        samples = ragtruth_samples[:50] if len(ragtruth_samples) > 50 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                stage2_detector.predict(
                    context=sample.context,
                    answer=sample.response or "",
                    question=sample.question,
                    output_format="spans",
                )

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    stage2_detector.predict(
                        context=sample.context,
                        answer=sample.response or "",
                        question=sample.question,
                        output_format="spans",
                    )

        stats = timer.get_stats()
        # Stage 2 target: 100-200ms
        assert stats.mean_ms < 250, f"Stage 2 too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 350, f"Stage 2 P95 too slow: {stats.p95_ms:.2f}ms"

    def test_stage2_accuracy(self, stage2_detector, ragtruth_samples, benchmark_config):
        """Benchmark Stage 2 accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=True)
        memory_tracker = MemoryTracker()

        with memory_tracker.track():
            for sample in ragtruth_samples:
                if not sample.context or not sample.response:
                    continue

                with timer.measure():
                    spans = stage2_detector.predict(
                        context=sample.context,
                        answer=sample.response,
                        question=sample.question,
                        output_format="spans",
                    )

                # Stage 2 returns response-level spans
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
                        component_name="stage2",
                    )
                )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()
        memory = memory_tracker.get_stats()

        results = BenchmarkResults(
            component="stage2",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            memory=memory,
            config={"components": ["ncs", "nli"]},
        )

        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 250, f"Too slow: {timing.mean_ms:.2f}ms"

        print(f"\n{'='*60}")
        print(f"Stage 2 Full Pipeline Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc is not None else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 is not None else "F1: N/A")
        print(f"Optimal F1: {metrics.optimal_f1:.3f}" if metrics.optimal_f1 is not None else "Optimal F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
        if memory.gpu_peak_mb:
            print(f"GPU Peak: {memory.gpu_peak_mb:.1f}MB")

    def test_stage2_detailed_scores(self, stage2_detector, ragtruth_samples, benchmark_config):
        """Test Stage 2 with detailed score breakdown."""
        samples = ragtruth_samples[:10] if len(ragtruth_samples) > 10 else ragtruth_samples

        ncs_scores = []
        nli_scores = []

        for sample in samples:
            if not sample.context or not sample.response:
                continue

            scores = stage2_detector.get_detailed_scores(
                context=sample.context,
                answer=sample.response,
            )

            ncs_scores.append(scores.get("ncs_score", 0.5))
            nli_scores.append(scores.get("nli_score", 0.5))

        if ncs_scores:
            import numpy as np

            print(f"\n{'='*60}")
            print(f"Stage 2 Component Scores")
            print(f"{'='*60}")
            print(f"NCS: mean={np.mean(ncs_scores):.3f}, std={np.std(ncs_scores):.3f}")
            print(f"NLI: mean={np.mean(nli_scores):.3f}, std={np.std(nli_scores):.3f}")
