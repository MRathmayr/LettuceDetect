"""Shared Stage 3 model variant configurations.

Used by run_full_benchmark.py and conftest.py
to keep probe paths, layer indices, and model names in sync.
"""

import os

# Stage 3 model variant configs (hallu probes trained on RAGTruth)
STAGE3_VARIANTS = {
    "3b": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "probe_subdir": "training_3b_qwen/probe_3b_qwen_pca512.joblib",
        "layer_index": -15,  # PCA 512: AUROC 0.889
    },
    "7b": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "probe_subdir": "training_7b_qwen/probe_7b_qwen_pca512.joblib",
        "layer_index": -12,  # PCA 512: AUROC 0.897
    },
    "8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "probe_subdir": "training_8b_llama/probe_8b_llama_pca512.joblib",
        "layer_index": -16,  # PCA 512: AUROC 0.897
    },
    "14b": {
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "probe_subdir": "training_14b_qwen/probe_14b_qwen_pca512.joblib",
        "layer_index": -20,  # PCA 512: AUROC 0.905
    },
}


def resolve_probe_path(probe_subdir: str) -> str | None:
    """Resolve probe path relative to project root.

    Probes are stored in Diploma/hallu-training/results/ (3 levels above
    tests/benchmarks/), not inside LettuceDetect/.
    """
    # From tests/benchmarks/core/ -> 4 levels up to Diploma/
    probe_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../../..",
        f"hallu-training/results/{probe_subdir}",
    ))
    return probe_path if os.path.exists(probe_path) else None
