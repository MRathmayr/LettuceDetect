"""Benchmark tests for Stage 3: Hallu Probe detector (Llama 8B).

Separate file from test_stage3_3b.py so pytest tears down the 3B model
(freeing ~3 GB VRAM) before attempting to load the 8B model.

Requires ~5-7 GB VRAM (4-bit).

Run:
    pytest tests/benchmarks/stages/test_stage3_7b.py --quick -v -s
    pytest tests/benchmarks/stages/test_stage3_7b.py --full -v -s
"""

import pytest

from tests.benchmarks.stages._stage3_helpers import _run_stage3_accuracy

_8B_CONFIG = {"model": "meta-llama/Llama-3.1-8B-Instruct", "layer": -16, "position": "mean"}


@pytest.mark.benchmark
@pytest.mark.gpu
class TestStage3BenchmarkLlama8B:
    """Benchmarks for Stage 3 with Llama 3.1 8B hallu probe."""

    def test_stage3_accuracy_halueval_qa(
        self, stage3_detector_8b, halueval_qa_samples, benchmark_config
    ):
        """Benchmark Stage 3 (8B) accuracy on HaluEval QA."""
        _run_stage3_accuracy(
            stage3_detector_8b, halueval_qa_samples,
            "stage3_hallu_probe_8b", "halueval_qa", _8B_CONFIG, benchmark_config,
        )

    def test_stage3_accuracy_ragtruth(
        self, stage3_detector_8b, ragtruth_samples, benchmark_config
    ):
        """Benchmark Stage 3 (8B) accuracy on RAGTruth."""
        _run_stage3_accuracy(
            stage3_detector_8b, ragtruth_samples,
            "stage3_hallu_probe_8b", "ragtruth", _8B_CONFIG, benchmark_config,
        )
