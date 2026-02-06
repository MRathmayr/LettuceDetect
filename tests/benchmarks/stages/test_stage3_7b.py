"""Benchmark tests for Stage 3: Reading Probe detector (Qwen 7B).

Separate file from test_stage3.py so pytest tears down the 3B model
(freeing ~3 GB VRAM) before attempting to load the 7B model.

Requires ~5-7 GB VRAM (4-bit). Marginal on GTX 1080 (8 GB).

Run:
    pytest tests/benchmarks/stages/test_stage3_7b.py --quick -v -s
    pytest tests/benchmarks/stages/test_stage3_7b.py --full -v -s
"""

import pytest

from tests.benchmarks.stages._stage3_helpers import _run_stage3_accuracy

_7B_CONFIG = {"model": "Qwen/Qwen2.5-7B-Instruct", "layer": -9, "position": "mean"}


@pytest.mark.benchmark
@pytest.mark.gpu
class TestStage3BenchmarkQwen7B:
    """Benchmarks for Stage 3 with Qwen 2.5 7B reading probe.

    Requires ~5-7 GB VRAM (4-bit). Marginal on GTX 1080 (8 GB).
    """

    def test_stage3_accuracy_halueval_qa(
        self, stage3_detector_7b, halueval_qa_samples, benchmark_config
    ):
        """Benchmark Stage 3 (7B) accuracy on HaluEval QA."""
        _run_stage3_accuracy(
            stage3_detector_7b, halueval_qa_samples,
            "stage3_reading_probe_7b", "halueval_qa", _7B_CONFIG, benchmark_config,
        )

    def test_stage3_accuracy_ragtruth(
        self, stage3_detector_7b, ragtruth_samples, benchmark_config
    ):
        """Benchmark Stage 3 (7B) accuracy on RAGTruth."""
        _run_stage3_accuracy(
            stage3_detector_7b, ragtruth_samples,
            "stage3_reading_probe_7b", "ragtruth", _7B_CONFIG, benchmark_config,
        )
