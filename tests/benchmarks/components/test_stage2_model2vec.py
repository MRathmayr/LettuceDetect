"""Benchmark tests for Model2Vec (NCS) component."""

import pytest

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)


@pytest.mark.benchmark
class TestModel2VecBenchmark:
    """Benchmarks for Model2Vec NCS computation."""

    def test_model2vec_latency(self, model2vec_encoder, ragtruth_samples, benchmark_config):
        """Benchmark Model2Vec encoding latency."""
        timer = BenchmarkTimer(sync_cuda=False)  # Model2Vec is CPU

        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                model2vec_encoder.compute_ncs(sample.context, sample.response or "")

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    model2vec_encoder.compute_ncs(sample.context, sample.response or "")

        stats = timer.get_stats()
        assert stats.mean_ms < 20, f"Model2Vec too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 50, f"Model2Vec P95 too slow: {stats.p95_ms:.2f}ms"

    def test_model2vec_accuracy(self, model2vec_encoder, ragtruth_samples, benchmark_config):
        """Benchmark Model2Vec NCS accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=False)

        for sample in ragtruth_samples:
            if not sample.context or not sample.response:
                continue

            with timer.measure():
                ncs = model2vec_encoder.compute_ncs(sample.context, sample.response)

            # NCS is cosine similarity in [-1, 1], map to [0, 1] hallucination score
            # ncs=1 (identical) -> 0, ncs=0 (orthogonal) -> 0.5, ncs=-1 (opposite) -> 1
            hallucination_score = (1.0 - ncs["max"]) / 2.0

            predictions.append(
                PredictionResult(
                    sample_id=sample.id,
                    ground_truth=sample.ground_truth,
                    predicted_score=hallucination_score,
                    predicted_label=1 if hallucination_score >= 0.5 else 0,
                    latency_ms=timer.last_ms,
                    component_name="model2vec",
                )
            )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()

        results = BenchmarkResults(
            component="model2vec",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            config={"model": "minishlab/potion-base-32M"},
        )

        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 20, f"Too slow: {timing.mean_ms:.2f}ms"

        print(f"\n{'='*60}")
        print(f"Model2Vec (NCS) Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc is not None else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 is not None else "F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
