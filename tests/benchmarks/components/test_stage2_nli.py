"""Benchmark tests for NLI detector component."""

import time

import pytest

from tests.benchmarks.core import (
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)
from tests.benchmarks.core.memory import MemoryTracker
from tests.benchmarks.core.verdict_export import (
    compact_metrics,
    compact_predictions,
    write_compact,
    write_native,
)


@pytest.fixture(scope="module")
def nli_detector():
    """Create NLI detector instance."""
    from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector

    detector = NLIContradictionDetector()
    detector.preload()
    return detector


@pytest.mark.benchmark
@pytest.mark.gpu
class TestNLIDetectorBenchmark:
    """Benchmarks for NLI detector."""

    def test_nli_latency(self, nli_detector, ragtruth_samples, benchmark_config):
        """Benchmark NLI detection latency."""
        timer = BenchmarkTimer(sync_cuda=True)  # GPU timing

        samples = ragtruth_samples[:50] if len(ragtruth_samples) > 50 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                nli_detector.compute_context_nli(sample.context, sample.response or "")

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    nli_detector.compute_context_nli(sample.context, sample.response or "")

        stats = timer.get_stats()
        assert stats.mean_ms < 200, f"NLI too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 300, f"NLI P95 too slow: {stats.p95_ms:.2f}ms"

    def test_nli_accuracy(self, nli_detector, ragtruth_samples, benchmark_config):
        """Benchmark NLI detector accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=True)
        memory_tracker = MemoryTracker()
        started = time.time()

        with memory_tracker.track():
            for sample in ragtruth_samples:
                if not sample.context or not sample.response:
                    continue

                with timer.measure():
                    result = nli_detector.compute_context_nli(sample.context, sample.response)

                # NLI hallucination_score uses weighted combined (higher = hallucination)
                hallucination_score = result.get("hallucination_score", 0.5)

                predictions.append(
                    PredictionResult(
                        sample_id=sample.id,
                        ground_truth=sample.ground_truth,
                        predicted_score=hallucination_score,
                        predicted_label=1 if hallucination_score >= 0.5 else 0,
                        latency_ms=timer.last_ms,
                        component_name="nli",
                    )
                )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()
        memory = memory_tracker.get_stats()
        elapsed = time.time() - started

        # Per-sample export for the paper's verdict-style baselines. Written
        # before the assertions below so a latency regression on a loaded
        # machine does not discard 2700 rows of scoring.
        task_map = {s.id: s.task_type for s in ragtruth_samples if s.task_type}
        meta = {
            "model": nli_detector.MODEL_NAME,
            "precision": "fp32",
            "n_samples": metrics.n_samples,
            "dataset": "ragtruth_test",
            "latency_mean_ms": timing.mean_ms,
            "latency_p95_ms": timing.p95_ms,
            "gpu_peak_mb": memory.gpu_peak_mb,
            "total_time_sec": elapsed,
            "prompt_template_note": (
                "MiniCheck seq2seq format 'Document: {context}\\nClaim: {response}', "
                "truncated to 512 tokens; score = P(No) over the Yes/No logits of the "
                "first generated token, max over context passages"
            ),
        }
        native = write_native(
            "minicheck_ragtruth_test",
            {
                "metadata": meta,
                "metrics": metrics.to_dict(),
                "timing": timing.to_dict(),
                "predictions": [
                    {
                        "sample_id": p.sample_id,
                        "ground_truth": p.ground_truth,
                        "predicted_score": p.predicted_score,
                        "predicted_label": p.predicted_label,
                        "latency_ms": round(p.latency_ms, 3),
                        "task_type": task_map.get(p.sample_id),
                    }
                    for p in predictions
                ],
            },
        )
        paper = write_compact(
            "minicheck_ragtruth_test",
            meta,
            compact_metrics(metrics),
            compact_predictions(predictions, task_map),
        )
        print(f"\nPer-sample scores: {native}\n                   {paper}")

        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 200, f"Too slow: {timing.mean_ms:.2f}ms"

        print(f"\n{'=' * 60}")
        print("NLI Detector Benchmark Results")
        print(f"{'=' * 60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc is not None else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 is not None else "F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
        if memory.gpu_peak_mb:
            print(f"GPU Peak: {memory.gpu_peak_mb:.1f}MB")
