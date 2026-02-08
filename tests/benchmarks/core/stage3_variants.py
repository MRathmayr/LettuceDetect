"""Shared Stage 3 model variant configurations.

Used by run_full_benchmark.py, run_complete_benchmark.py, and conftest.py
to keep probe paths, layer indices, and model names in sync.
"""

import os

# Stage 3 model variant configs
STAGE3_VARIANTS = {
    "3b": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "probe_subdir": "training_430k_3b_qwen/reading_probe_3b_qwen.joblib",
        "layer_index": -12,  # From sweep: AUROC 0.767
    },
    "7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "probe_subdir": "training_430k_7b_qwen/reading_probe_7b_qwen.joblib",
        "layer_index": -9,  # From sweep: AUROC 0.792
    },
    "8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "probe_subdir": "training_100k_8b_llama/reading_probe_8b_llama.joblib",
        "layer_index": -16,  # From sweep: AUROC 0.815
    },
    "14b": {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "probe_subdir": "training_430k_14b_qwen/reading_probe_14b_qwen.joblib",
        "layer_index": -16,  # From sweep: AUROC 0.826
    },
}


def resolve_probe_path(probe_subdir: str) -> str | None:
    """Resolve probe path relative to project root.

    Probes are stored in Diploma/read-training/results/ (3 levels above
    tests/benchmarks/), not inside LettuceDetect/.
    """
    # From tests/benchmarks/core/ -> 4 levels up to Diploma/
    probe_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../..",
        f"read-training/results/{probe_subdir}",
    ))
    return probe_path if os.path.exists(probe_path) else None
