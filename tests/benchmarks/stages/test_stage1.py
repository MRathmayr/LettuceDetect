"""Benchmark tests for full Stage 1 pipeline."""

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
class TestStage1Benchmark:
    """Benchmarks for full Stage 1 detector with all augmentations."""

    def test_stage1_latency(self, stage1_detector, ragtruth_samples, benchmark_config):
        """Benchmark Stage 1 pipeline latency."""
        timer = BenchmarkTimer(sync_cuda=True)

        samples = ragtruth_samples[:50] if len(ragtruth_samples) > 50 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                stage1_detector.predict(
                    context=sample.context,
                    answer=sample.response or "",
                    question=sample.question,
                    output_format="spans",
                )

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    stage1_detector.predict(
                        context=sample.context,
                        answer=sample.response or "",
                        question=sample.question,
                        output_format="spans",
                    )

        stats = timer.get_stats()
        # Stage 1 target: 50-150ms
        assert stats.mean_ms < 200, f"Stage 1 too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 300, f"Stage 1 P95 too slow: {stats.p95_ms:.2f}ms"

    def test_stage1_accuracy(self, stage1_detector, ragtruth_samples, benchmark_config):
        """Benchmark Stage 1 accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=True)
        memory_tracker = MemoryTracker()

        with memory_tracker.track():
            for sample in ragtruth_samples:
                if not sample.context or not sample.response:
                    continue

                with timer.measure():
                    spans = stage1_detector.predict(
                        context=sample.context,
                        answer=sample.response,
                        question=sample.question,
                        output_format="spans",
                    )

                # Aggregate span predictions to response-level
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
                        component_name="stage1",
                    )
                )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()
        memory = memory_tracker.get_stats()

        results = BenchmarkResults(
            component="stage1",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            memory=memory,
            config={"augmentations": ["ner", "numeric", "lexical"]},
        )

        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 200, f"Too slow: {timing.mean_ms:.2f}ms"

        print(f"\n{'='*60}")
        print(f"Stage 1 Full Pipeline Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 else "F1: N/A")
        print(f"Optimal F1: {metrics.optimal_f1:.3f}" if metrics.optimal_f1 else "Optimal F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
        if memory.gpu_peak_mb:
            print(f"GPU Peak: {memory.gpu_peak_mb:.1f}MB")

    def test_stage1_component_breakdown(self, stage1_detector, ragtruth_samples, benchmark_config):
        """Measure latency contribution of each Stage 1 component."""
        from lettucedetect.cascade.types import CascadeInput

        samples = ragtruth_samples[:20] if len(ragtruth_samples) > 20 else ragtruth_samples

        transformer_times = []
        augmentation_times = []
        total_times = []

        for sample in samples:
            if not sample.context or not sample.response:
                continue

            cascade_input = CascadeInput(
                context=sample.context,
                answer=sample.response,
                question=sample.question,
                prompt=None,
                previous_stage_result=None,
            )

            result = stage1_detector.predict_stage(cascade_input, has_next_stage=True)

            # StageResult includes latency breakdown in component_scores
            total_times.append(result.latency_ms)

        if total_times:
            import numpy as np

            print(f"\n{'='*60}")
            print(f"Stage 1 Component Breakdown")
            print(f"{'='*60}")
            print(f"Total latency: {np.mean(total_times):.2f}ms (P95: {np.percentile(total_times, 95):.2f}ms)")
