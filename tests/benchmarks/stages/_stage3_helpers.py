"""Shared helpers for Stage 3 benchmark tests."""

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)
from tests.benchmarks.core.memory import MemoryTracker


def _run_stage3_accuracy(detector, samples, component_name, dataset_name, config, benchmark_config):
    """Shared accuracy benchmark logic for Stage 3 detectors.

    Args:
        detector: ReadingProbeDetector instance.
        samples: List of BenchmarkSample.
        component_name: Component identifier (e.g., "stage3_reading_probe_3b").
        dataset_name: Dataset name (e.g., "halueval_qa", "ragtruth").
        config: Config dict for BenchmarkResults metadata.
        benchmark_config: Pytest benchmark config fixture.

    Returns:
        Tuple of (BenchmarkResults, AccuracyMetrics, TimingStats).
    """
    predictions = []
    timer = BenchmarkTimer(sync_cuda=True)
    memory_tracker = MemoryTracker()

    with memory_tracker.track():
        for sample in samples:
            if not sample.context or not sample.response:
                continue

            with timer.measure():
                spans = detector.predict(
                    context=sample.context,
                    answer=sample.response,
                    question=sample.question,
                    output_format="spans",
                )

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
                    component_name=component_name,
                )
            )

    metrics = compute_accuracy_metrics(
        predictions, compute_ci=benchmark_config.component.compute_ci
    )
    timing = timer.get_stats()
    memory = memory_tracker.get_stats()

    results = BenchmarkResults(
        component=component_name,
        dataset=dataset_name,
        predictions=predictions,
        metrics=metrics,
        timing=timing,
        memory=memory,
        config=config,
    )

    assert metrics.n_samples > 0, "No predictions made"

    label = config.get("model", component_name).split("/")[-1]
    print(f"\n{'='*60}")
    print(f"Stage 3 ({label}) - {dataset_name}")
    print(f"{'='*60}")
    print(f"Samples: {metrics.n_samples}")
    print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc is not None else "AUROC: N/A")
    print(f"F1: {metrics.f1:.3f}" if metrics.f1 is not None else "F1: N/A")
    print(f"Optimal F1: {metrics.optimal_f1:.3f} @ {metrics.optimal_threshold:.3f}" if metrics.optimal_f1 is not None else "Optimal F1: N/A")
    print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
    if memory.gpu_peak_mb:
        print(f"GPU Peak: {memory.gpu_peak_mb:.1f}MB")

    return results, metrics, timing
