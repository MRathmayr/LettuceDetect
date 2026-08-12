"""Per-sample export for the paper's verdict-style baselines.

The MiniCheck entailment baseline and the prompted-judge baseline both need to
be compared against the Grounding probe with a paired bootstrap, which requires
per-sample scores rather than the aggregates the component benchmarks store.
Both write the same compact JSON here so the paper reads one schema.

Score direction for every writer: higher = more likely hallucinated.
"""

import json
from datetime import datetime
from pathlib import Path

# tests/benchmarks/core/verdict_export.py -> repo root of the Diploma project
_DIPLOMA_ROOT = Path(__file__).resolve().parents[4]

NATIVE_DIR = (
    _DIPLOMA_ROOT / "LettuceDetect" / "tests" / "benchmarks" / "results" / "verdict_baselines"
)
PAPER_DIR = _DIPLOMA_ROOT / "paper" / "data" / "verdict_baselines"

# RAGTruth ships task types capitalized; the benchmark adapter lowercases them.
# The paper-side files carry the dataset's own spelling.
TASK_TYPE_LABELS = {"qa": "QA", "summary": "Summary", "data2txt": "Data2txt"}


def compact_metrics(metrics) -> dict:
    """Reduce an AccuracyMetrics to the fields the paper quotes."""
    return {
        "auroc": metrics.auroc,
        "f1_at_0.5": metrics.f1,
        "optimal_f1": metrics.optimal_f1,
        "optimal_threshold": metrics.optimal_threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }


def compact_predictions(predictions, task_map: dict) -> list[dict]:
    """Reduce PredictionResult objects to the paired-bootstrap fields."""
    out = []
    for p in predictions:
        raw_task = task_map.get(p.sample_id)
        out.append(
            {
                "sample_id": p.sample_id,
                "ground_truth": p.ground_truth,
                "predicted_score": round(p.predicted_score, 6),
                "predicted_label": p.predicted_label,
                "task_type": TASK_TYPE_LABELS.get(raw_task, raw_task),
            }
        )
    return out


def write_compact(name: str, metadata: dict, metrics: dict, predictions: list[dict]) -> Path:
    """Write the compact per-sample JSON under paper/data/verdict_baselines/."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {"timestamp": datetime.now().isoformat(), **metadata},
        "metrics": metrics,
        "predictions": predictions,
    }
    path = PAPER_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_native(name: str, payload: dict) -> Path:
    """Write the full run output under tests/benchmarks/results/verdict_baselines/."""
    NATIVE_DIR.mkdir(parents=True, exist_ok=True)
    path = NATIVE_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
