"""Benchmark tests for cascade routing behavior."""

import pytest

from tests.benchmarks.core import BenchmarkTimer


@pytest.fixture(scope="module")
def cascade_detector_detailed():
    """Create cascade detector for routing analysis."""
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
class TestRoutingBenchmark:
    """Benchmarks for cascade routing decisions."""

    def test_routing_distribution(self, cascade_detector_detailed, ragtruth_samples, benchmark_config):
        """Analyze routing distribution across samples."""
        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        resolved_stage1 = 0
        escalated_stage2 = 0
        stage1_times = []
        stage2_times = []

        for sample in samples:
            if not sample.context or not sample.response:
                continue

            result = cascade_detector_detailed.predict(
                context=sample.context,
                answer=sample.response,
                question=sample.question,
                output_format="detailed",
            )

            if isinstance(result, dict):
                routing = result.get("routing", {})
                resolved_at = routing.get("resolved_at_stage", 1)
                stage_latencies = routing.get("stage_latencies_ms", {})

                if resolved_at == 1:
                    resolved_stage1 += 1
                    if "stage1" in stage_latencies:
                        stage1_times.append(stage_latencies["stage1"])
                else:
                    escalated_stage2 += 1
                    if "stage1" in stage_latencies:
                        stage1_times.append(stage_latencies["stage1"])
                    if "stage2" in stage_latencies:
                        stage2_times.append(stage_latencies["stage2"])

        total = resolved_stage1 + escalated_stage2
        if total > 0:
            import numpy as np

            print(f"\n{'='*60}")
            print(f"Routing Distribution Analysis")
            print(f"{'='*60}")
            print(f"Total samples: {total}")
            print(f"Resolved at Stage 1: {resolved_stage1} ({100*resolved_stage1/total:.1f}%)")
            print(f"Escalated to Stage 2: {escalated_stage2} ({100*escalated_stage2/total:.1f}%)")

            if stage1_times:
                print(f"\nStage 1 latency: {np.mean(stage1_times):.2f}ms (P95: {np.percentile(stage1_times, 95):.2f}ms)")
            if stage2_times:
                print(f"Stage 2 latency: {np.mean(stage2_times):.2f}ms (P95: {np.percentile(stage2_times, 95):.2f}ms)")

            # Cascade should resolve most samples at Stage 1
            escalation_rate = escalated_stage2 / total
            assert escalation_rate < 0.8, f"Too many escalations: {escalation_rate:.1%}"

    def test_routing_accuracy(self, cascade_detector_detailed, ragtruth_samples, benchmark_config):
        """Test whether routing improves accuracy."""
        samples = ragtruth_samples[:100] if len(ragtruth_samples) > 100 else ragtruth_samples

        stage1_correct = 0
        cascade_correct = 0
        total = 0

        for sample in samples:
            if not sample.context or not sample.response:
                continue

            result = cascade_detector_detailed.predict(
                context=sample.context,
                answer=sample.response,
                question=sample.question,
                output_format="detailed",
            )

            if isinstance(result, dict):
                scores = result.get("scores", {})
                per_stage = scores.get("per_stage", {})

                # Get Stage 1 prediction
                stage1_score = per_stage.get("stage1", {}).get("score", 0.5)
                stage1_pred = 1 if stage1_score >= 0.5 else 0

                # Get final cascade prediction
                final_score = scores.get("final_score", 0.5)
                final_pred = 1 if final_score >= 0.5 else 0

                if stage1_pred == sample.ground_truth:
                    stage1_correct += 1
                if final_pred == sample.ground_truth:
                    cascade_correct += 1
                total += 1

        if total > 0:
            print(f"\n{'='*60}")
            print(f"Routing Accuracy Comparison")
            print(f"{'='*60}")
            print(f"Stage 1 only accuracy: {100*stage1_correct/total:.1f}%")
            print(f"Cascade (1+2) accuracy: {100*cascade_correct/total:.1f}%")
            print(f"Improvement: {100*(cascade_correct - stage1_correct)/total:.1f}%")
