"""Benchmark tests for NERVerifier."""

import pytest

from tests.benchmarks.core import (
    BenchmarkResults,
    BenchmarkTimer,
    PredictionResult,
    compute_accuracy_metrics,
)


@pytest.fixture(scope="module")
def ner_verifier():
    """Create NERVerifier instance."""
    from lettucedetect.detectors.stage1.augmentations.ner_verifier import NERVerifier

    verifier = NERVerifier()
    verifier.preload()
    return verifier


@pytest.mark.benchmark
class TestNERVerifierBenchmark:
    """Benchmarks for NERVerifier."""

    def test_ner_latency(self, ner_verifier, ragtruth_samples, benchmark_config):
        """Benchmark NER verification latency."""
        timer = BenchmarkTimer(sync_cuda=False)  # spaCy is CPU

        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        # Warmup
        for sample in samples[:3]:
            if sample.context:
                ner_verifier.score(
                    context=sample.context,
                    answer=sample.response or "",
                    question=sample.question,
                    token_predictions=None,
                )

        # Timed runs
        for sample in samples:
            if sample.context:
                with timer.measure():
                    ner_verifier.score(
                        context=sample.context,
                        answer=sample.response or "",
                        question=sample.question,
                        token_predictions=None,
                    )

        stats = timer.get_stats()
        # NER is slow on long contexts - realistic target is 50-100ms
        assert stats.mean_ms < 150, f"NER verification too slow: {stats.mean_ms:.2f}ms"
        assert stats.p95_ms < 250, f"NER verification P95 too slow: {stats.p95_ms:.2f}ms"

    def test_ner_accuracy(self, ner_verifier, ragtruth_samples, benchmark_config):
        """Benchmark NER verifier accuracy on RAGTruth."""
        predictions = []
        timer = BenchmarkTimer(sync_cuda=False)

        for sample in ragtruth_samples:
            if not sample.context or not sample.response:
                continue

            with timer.measure():
                result = ner_verifier.score(
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
                    component_name="ner",
                )
            )

        metrics = compute_accuracy_metrics(
            predictions, compute_ci=benchmark_config.component.compute_ci
        )
        timing = timer.get_stats()

        results = BenchmarkResults(
            component="ner",
            dataset="ragtruth",
            predictions=predictions,
            metrics=metrics,
            timing=timing,
            config={"model": "en_core_web_sm"},
        )

        assert metrics.n_samples > 0, "No predictions made"
        assert timing.mean_ms < 150, f"Too slow: {timing.mean_ms:.2f}ms"

        print(f"\n{'='*60}")
        print(f"NER Verifier Benchmark Results")
        print(f"{'='*60}")
        print(f"Samples: {metrics.n_samples}")
        print(f"AUROC: {metrics.auroc:.3f}" if metrics.auroc else "AUROC: N/A")
        print(f"F1: {metrics.f1:.3f}" if metrics.f1 else "F1: N/A")
        print(f"Latency: {timing.mean_ms:.2f}ms (P95: {timing.p95_ms:.2f}ms)")
