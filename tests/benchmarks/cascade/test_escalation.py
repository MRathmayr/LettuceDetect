"""Benchmark tests for escalation timing and paths."""

import pytest

from tests.benchmarks.core import BenchmarkTimer


@pytest.fixture(scope="module")
def cascade_for_escalation():
    """Create cascade detector for escalation analysis."""
    from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage2Config
    from lettucedetect.detectors.cascade import CascadeDetector

    config = CascadeConfig(
        stages=[1, 2],
        stage1=Stage1Config(augmentations=["ner", "numeric", "lexical"]),
        stage2=Stage2Config(components=["ncs", "nli"]),
    )

    detector = CascadeDetector(config)
    detector._stage1.warmup()
    detector._stage2.warmup()
    return detector


@pytest.mark.benchmark
@pytest.mark.gpu
class TestEscalationBenchmark:
    """Benchmarks for escalation behavior."""

    def test_escalation_timing_paths(self, cascade_for_escalation, ragtruth_samples, benchmark_config):
        """Analyze timing for different escalation paths."""
        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        # Group by path
        path_stage1_only = []  # Resolved at Stage 1
        path_stage1_then_2 = []  # Escalated to Stage 2

        for sample in samples:
            if not sample.context or not sample.response:
                continue

            result = cascade_for_escalation.predict(
                context=sample.context,
                answer=sample.response,
                question=sample.question,
                output_format="detailed",
            )

            if isinstance(result, dict):
                routing = result.get("routing", {})
                total_latency = routing.get("total_latency_ms", 0)
                resolved_at = routing.get("resolved_at_stage", 1)

                if resolved_at == 1:
                    path_stage1_only.append(total_latency)
                else:
                    path_stage1_then_2.append(total_latency)

        import numpy as np

        print(f"\n{'='*60}")
        print(f"Escalation Path Timing Analysis")
        print(f"{'='*60}")

        if path_stage1_only:
            print(f"\nPath: Stage 1 only (n={len(path_stage1_only)})")
            print(f"  Mean: {np.mean(path_stage1_only):.2f}ms")
            print(f"  P50:  {np.percentile(path_stage1_only, 50):.2f}ms")
            print(f"  P95:  {np.percentile(path_stage1_only, 95):.2f}ms")

        if path_stage1_then_2:
            print(f"\nPath: Stage 1 -> Stage 2 (n={len(path_stage1_then_2)})")
            print(f"  Mean: {np.mean(path_stage1_then_2):.2f}ms")
            print(f"  P50:  {np.percentile(path_stage1_then_2, 50):.2f}ms")
            print(f"  P95:  {np.percentile(path_stage1_then_2, 95):.2f}ms")

        # Compute expected latency based on distribution
        if path_stage1_only and path_stage1_then_2:
            total = len(path_stage1_only) + len(path_stage1_then_2)
            expected = (
                len(path_stage1_only) * np.mean(path_stage1_only) +
                len(path_stage1_then_2) * np.mean(path_stage1_then_2)
            ) / total
            print(f"\nWeighted average latency: {expected:.2f}ms")

    def test_escalation_reasons(self, cascade_for_escalation, ragtruth_samples, benchmark_config):
        """Analyze reasons for escalation."""
        samples = ragtruth_samples[:50] if len(ragtruth_samples) > 50 else ragtruth_samples

        escalation_reasons = {}

        for sample in samples:
            if not sample.context or not sample.response:
                continue

            result = cascade_for_escalation.predict(
                context=sample.context,
                answer=sample.response,
                question=sample.question,
                output_format="detailed",
            )

            if isinstance(result, dict):
                routing = result.get("routing", {})
                resolved_at = routing.get("resolved_at_stage", 1)

                if resolved_at > 1:
                    # Get Stage 1 result for escalation reason
                    per_stage = result.get("scores", {}).get("per_stage", {})
                    stage1_info = per_stage.get("stage1", {})
                    reason = stage1_info.get("routing_reason", "unknown")

                    if reason not in escalation_reasons:
                        escalation_reasons[reason] = 0
                    escalation_reasons[reason] += 1

        if escalation_reasons:
            print(f"\n{'='*60}")
            print(f"Escalation Reasons")
            print(f"{'='*60}")
            for reason, count in sorted(escalation_reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")

    def test_agreement_vs_escalation(self, cascade_for_escalation, ragtruth_samples, benchmark_config):
        """Analyze relationship between agreement and escalation."""
        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        agreements_resolved = []
        agreements_escalated = []

        for sample in samples:
            if not sample.context or not sample.response:
                continue

            result = cascade_for_escalation.predict(
                context=sample.context,
                answer=sample.response,
                question=sample.question,
                output_format="detailed",
            )

            if isinstance(result, dict):
                routing = result.get("routing", {})
                resolved_at = routing.get("resolved_at_stage", 1)

                per_stage = result.get("scores", {}).get("per_stage", {})
                stage1_info = per_stage.get("stage1", {})
                agreement = stage1_info.get("agreement", 0.5)

                if resolved_at == 1:
                    agreements_resolved.append(agreement)
                else:
                    agreements_escalated.append(agreement)

        import numpy as np

        print(f"\n{'='*60}")
        print(f"Agreement vs Escalation Analysis")
        print(f"{'='*60}")

        if agreements_resolved:
            print(f"\nResolved at Stage 1 (n={len(agreements_resolved)}):")
            print(f"  Agreement mean: {np.mean(agreements_resolved):.3f}")
            print(f"  Agreement std:  {np.std(agreements_resolved):.3f}")

        if agreements_escalated:
            print(f"\nEscalated to Stage 2 (n={len(agreements_escalated)}):")
            print(f"  Agreement mean: {np.mean(agreements_escalated):.3f}")
            print(f"  Agreement std:  {np.std(agreements_escalated):.3f}")
