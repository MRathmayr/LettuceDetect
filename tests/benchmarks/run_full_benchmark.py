#!/usr/bin/env python3
"""Run full benchmark suite and save results to JSON.

3-phase benchmark:
  Phase 1: Model-independent components (lexical, numeric, ner, model2vec, nli, stage2)
  Phase 2: Stage 3 standalone (once per LLM - 3B, 7B, 8B, 14B)
  Phase 3: Per-transformer benchmarks (base, large) with cascade combinations

Output: One JSON per (transformer x probe) combination:
  benchmark_base_3b_{timestamp}.json, ..., benchmark_large_14b_{timestamp}.json

Usage:
    python tests/benchmarks/run_full_benchmark.py --quick                    # 100 samples
    python tests/benchmarks/run_full_benchmark.py --limit 500               # 500 samples
    python tests/benchmarks/run_full_benchmark.py --transformer base        # base only
    python tests/benchmarks/run_full_benchmark.py --stage3 3b               # 3B probe only
    python tests/benchmarks/run_full_benchmark.py                           # Full run (all)
"""

import argparse
import copy
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.core.stage3_variants import STAGE3_VARIANTS, resolve_probe_path

TRANSFORMER_MODELS = {
    "base": "KRLabsOrg/lettucedect-base-modernbert-en-v1",
    "large": "KRLabsOrg/lettucedect-large-modernbert-en-v1",
}


def _gpu_mem_log(label: str):
    """Log GPU memory allocated/reserved."""
    try:
        import torch

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"   [GPU] {label}: {alloc:.0f}MB allocated, {reserved:.0f}MB reserved")
    except Exception:
        pass


def _cuda_cleanup():
    """Synchronize CUDA, garbage collect, and empty cache."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _compute_per_task_metrics(predictions, valid_samples, component_name):
    """Compute metrics split by task_type for samples that have it."""
    from tests.benchmarks.core import compute_accuracy_metrics

    task_map = {s.id: s.task_type for s in valid_samples if s.task_type}
    if not task_map:
        return {}

    by_task = {}
    for p in predictions:
        task = task_map.get(p.sample_id)
        if task:
            by_task.setdefault(task, []).append(p)

    results = {}
    for task_type, task_preds in sorted(by_task.items()):
        metrics = compute_accuracy_metrics(task_preds, compute_ci=False)
        results[task_type] = {
            "auroc": metrics.auroc,
            "f1": metrics.f1,
            "optimal_f1": metrics.optimal_f1,
            "optimal_threshold": metrics.optimal_threshold,
            "n_samples": metrics.n_samples,
        }
        auroc_str = f"{metrics.auroc:.3f}" if metrics.auroc is not None else "N/A"
        f1_str = f"{metrics.f1:.3f}" if metrics.f1 is not None else "N/A"
        opt_f1_str = f"{metrics.optimal_f1:.3f}" if metrics.optimal_f1 is not None else "N/A"
        print(f"     {component_name} [{task_type}]: AUROC={auroc_str}, F1={f1_str}, OptF1={opt_f1_str}, n={metrics.n_samples}")

    return results


def _make_result_dict(metrics, stats, mem_stats=None):
    """Build standard result dict from metrics/stats."""
    d = {
        "auroc": metrics.auroc,
        "f1": metrics.f1,
        "optimal_f1": metrics.optimal_f1,
        "optimal_threshold": metrics.optimal_threshold,
        "latency_mean_ms": stats.mean_ms,
        "latency_p95_ms": stats.p95_ms,
        "n_samples": metrics.n_samples,
    }
    if mem_stats is not None:
        d["gpu_peak_mb"] = mem_stats.gpu_peak_mb
    return d


def _print_component_result(metrics, stats, mem_stats=None):
    """Print component benchmark result."""
    auroc = f"{metrics.auroc:.3f}" if metrics.auroc is not None else "N/A"
    opt_f1 = f"{metrics.optimal_f1:.3f}" if metrics.optimal_f1 is not None else "N/A"
    parts = [f"   AUROC: {auroc}, OptF1: {opt_f1}, Latency: {stats.mean_ms:.2f}ms"]
    if mem_stats is not None:
        parts.append(f"GPU: {mem_stats.gpu_peak_mb:.0f}MB")
    print(", ".join(parts))


# ================================================================
# PHASE 1: Model-Independent Components
# ================================================================


def run_model_independent_benchmarks(valid_samples: list) -> dict:
    """Run model-independent component + stage2 benchmarks.

    Includes: lexical, numeric, ner, model2vec, nli, stage2.
    Does NOT include transformer, stage1, or cascade[1,2].
    """
    from tests.benchmarks.core import BenchmarkTimer, PredictionResult, compute_accuracy_metrics
    from tests.benchmarks.core.memory import MemoryTracker

    results = {"components": {}, "stages": {}, "cascade": {}}

    print("\n" + "=" * 60)
    print("PHASE 1: Model-Independent Components")
    print("=" * 60)

    # --- Lexical ---
    print("\n[1/5] Lexical Overlap...")
    from lettucedetect.utils.lexical import LexicalOverlapCalculator

    lexical = LexicalOverlapCalculator()
    lexical.preload()
    timer = BenchmarkTimer(sync_cuda=False)
    predictions = []
    for s in valid_samples:
        with timer.measure():
            r = lexical.score(s.context, s.response, s.question, None)
        predictions.append(PredictionResult(s.id, s.ground_truth, r.score, int(r.score >= 0.5), timer.last_ms, "lexical"))
    stats = timer.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["components"]["lexical"] = {
        **_make_result_dict(metrics, stats),
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "lexical"),
    }
    _print_component_result(metrics, stats)

    # --- Numeric ---
    print("\n[2/5] Numeric Validator...")
    from lettucedetect.detectors.stage1.augmentations.numeric_validator import NumericValidator

    numeric = NumericValidator()
    numeric.preload()
    timer = BenchmarkTimer(sync_cuda=False)
    predictions = []
    for s in valid_samples:
        with timer.measure():
            r = numeric.score(s.context, s.response, s.question, None)
        predictions.append(PredictionResult(s.id, s.ground_truth, r.score, int(r.score >= 0.5), timer.last_ms, "numeric"))
    stats = timer.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["components"]["numeric"] = {
        **_make_result_dict(metrics, stats),
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "numeric"),
    }
    _print_component_result(metrics, stats)

    # --- NER ---
    print("\n[3/5] NER Verifier...")
    from lettucedetect.detectors.stage1.augmentations.ner_verifier import NERVerifier

    ner = NERVerifier()
    ner.preload()
    timer = BenchmarkTimer(sync_cuda=False)
    predictions = []
    for s in valid_samples:
        with timer.measure():
            r = ner.score(s.context, s.response, s.question, None)
        predictions.append(PredictionResult(s.id, s.ground_truth, r.score, int(r.score >= 0.5), timer.last_ms, "ner"))
    stats = timer.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["components"]["ner"] = {
        **_make_result_dict(metrics, stats),
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "ner"),
    }
    _print_component_result(metrics, stats)

    # --- Model2Vec ---
    print("\n[4/5] Model2Vec (NCS)...")
    from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder

    m2v = Model2VecEncoder()
    m2v.preload()
    timer = BenchmarkTimer(sync_cuda=False)
    predictions = []
    for s in valid_samples:
        with timer.measure():
            ncs = m2v.compute_ncs(s.context, s.response)
        score = (1.0 - ncs["max"]) / 2.0
        predictions.append(PredictionResult(s.id, s.ground_truth, score, int(score >= 0.5), timer.last_ms, "model2vec"))
    stats = timer.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["components"]["model2vec"] = {
        **_make_result_dict(metrics, stats),
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "model2vec"),
    }
    _print_component_result(metrics, stats)

    # --- NLI ---
    print("\n[5/5] NLI Detector...")
    from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector

    nli = NLIContradictionDetector()
    nli.preload()
    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                r = nli.compute_context_nli(s.context, s.response)
            score = r["hallucination_score"]
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(score >= 0.5), timer.last_ms, "nli"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["components"]["nli"] = {
        **_make_result_dict(metrics, stats, mem_stats),
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "nli"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del nli
    _cuda_cleanup()

    # --- Stage 2 ---
    print("\n[Stage 2] Full pipeline (NLI-only)...")
    from lettucedetect.detectors.stage2.detector import Stage2Detector

    stage2 = Stage2Detector()
    stage2.warmup()
    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                spans = stage2.predict(s.context, s.response, s.question, output_format="spans")
            score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(bool(spans)), timer.last_ms, "stage2"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["stages"]["stage2"] = {
        **_make_result_dict(metrics, stats, mem_stats),
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "stage2"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del stage2
    _cuda_cleanup()

    return results


# ================================================================
# PHASE 2: Stage 3 Standalone
# ================================================================


def run_stage3_standalone(valid_samples: list, model_size: str, variant: dict) -> dict:
    """Run Stage 3 standalone (LLM + probe only, no transformer).

    Returns result dict, or empty dict on failure.
    """
    import torch

    from tests.benchmarks.core import BenchmarkTimer, PredictionResult, compute_accuracy_metrics
    from tests.benchmarks.core.memory import MemoryTracker

    probe_path = resolve_probe_path(variant["probe_subdir"])
    if not probe_path:
        print(f"   SKIPPED: Probe file not found for {model_size}")
        return {}
    if not torch.cuda.is_available():
        print("   SKIPPED: CUDA GPU required for Stage 3")
        return {}

    _cuda_cleanup()
    _gpu_mem_log(f"before loading {model_size}")

    from lettucedetect.detectors.stage3.reading_probe_detector import ReadingProbeDetector

    detector = ReadingProbeDetector(
        model_name_or_path=variant["model"],
        probe_path=probe_path,
        layer_index=variant["layer_index"],
        token_position="mean",
    )
    detector.warmup()
    _gpu_mem_log(f"after loading {model_size}")

    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                spans = detector.predict(s.context, s.response, s.question, output_format="spans")
            score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(bool(spans)), timer.last_ms, f"stage3_{model_size}"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)

    result = {
        **_make_result_dict(metrics, stats, mem_stats),
        "per_task": _compute_per_task_metrics(predictions, valid_samples, f"stage3_{model_size}"),
    }
    _print_component_result(metrics, stats, mem_stats)

    del detector
    _cuda_cleanup()
    _gpu_mem_log(f"after cleanup {model_size}")

    return result


# ================================================================
# PHASE 3: Transformer-Dependent Benchmarks
# ================================================================


def run_transformer_benchmarks(valid_samples: list, model_path: str, model_tag: str) -> dict:
    """Run transformer standalone + stage1 + cascade[1,2] for a specific transformer.

    Returns dict with keys: transformer_{tag}, stage1_{tag}, cascade_12_{tag}.
    """
    from tests.benchmarks.core import BenchmarkTimer, PredictionResult, compute_accuracy_metrics
    from tests.benchmarks.core.memory import MemoryTracker

    results = {}

    # --- Transformer standalone ---
    print(f"\n[Transformer] {model_tag} standalone...")
    from lettucedetect.detectors.transformer import TransformerDetector

    transformer = TransformerDetector(model_path=model_path)
    transformer.warmup()
    _gpu_mem_log(f"transformer {model_tag} loaded")
    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                spans = transformer.predict(s.context, s.response, s.question, output_format="spans")
            score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(bool(spans)), timer.last_ms, f"transformer_{model_tag}"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results[f"transformer_{model_tag}"] = {
        **_make_result_dict(metrics, stats, mem_stats),
        "model_path": model_path,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, f"transformer_{model_tag}"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del transformer
    _cuda_cleanup()

    # --- Stage 1 ---
    print(f"\n[Stage 1] {model_tag} + lexical + model2vec...")
    from lettucedetect.configs.models import Stage1Config
    from lettucedetect.detectors.stage1.detector import Stage1Detector

    stage1 = Stage1Detector(config=Stage1Config(model_path=model_path, augmentations=["lexical", "model2vec"]))
    stage1.warmup()
    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                spans = stage1.predict(s.context, s.response, s.question, output_format="spans")
            score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(bool(spans)), timer.last_ms, f"stage1_{model_tag}"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results[f"stage1_{model_tag}"] = {
        **_make_result_dict(metrics, stats, mem_stats),
        "model_path": model_path,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, f"stage1_{model_tag}"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del stage1
    _cuda_cleanup()

    # --- Cascade [1,2] ---
    print(f"\n[Cascade 1+2] {model_tag}...")
    from lettucedetect.configs.models import CascadeConfig, Stage2Config
    from lettucedetect.detectors.cascade import CascadeDetector

    config = CascadeConfig(
        stages=[1, 2],
        stage1=Stage1Config(model_path=model_path, augmentations=["lexical", "model2vec"]),
        stage2=Stage2Config(components=["ncs", "nli"]),
    )
    cascade = CascadeDetector(config)
    cascade.warmup()

    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    stage1_resolved = 0
    stage2_resolved = 0

    with memory.track():
        for s in valid_samples:
            with timer.measure():
                result = cascade.predict(s.context, s.response, s.question, output_format="detailed")

            if isinstance(result, dict):
                score = result.get("scores", {}).get("final_score", 0.0)
                resolved_at = result.get("routing", {}).get("resolved_at_stage", 1)
                if resolved_at == 1:
                    stage1_resolved += 1
                else:
                    stage2_resolved += 1
            else:
                score = max((sp.get("confidence", 0.5) for sp in result), default=0.0)

            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(score >= 0.5), timer.last_ms, f"cascade_12_{model_tag}"))

    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)

    total = stage1_resolved + stage2_resolved
    results[f"cascade_12_{model_tag}"] = {
        **_make_result_dict(metrics, stats, mem_stats),
        "model_path": model_path,
        "stage1_resolved_pct": 100 * stage1_resolved / total if total > 0 else 0,
        "stage2_resolved_pct": 100 * stage2_resolved / total if total > 0 else 0,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, f"cascade_12_{model_tag}"),
    }
    _print_component_result(metrics, stats, mem_stats)
    if total > 0:
        print(f"   Routing: {stage1_resolved}/{total} ({100*stage1_resolved/total:.1f}%) resolved at Stage 1")

    del cascade
    _cuda_cleanup()

    return results


def run_cascade_13(valid_samples: list, model_size: str, variant: dict, model_path: str, model_tag: str) -> dict:
    """Run Cascade[1,3] for a specific transformer + stage3 variant.

    Returns result dict, or empty dict on failure.
    """
    import torch

    from tests.benchmarks.core import BenchmarkTimer, PredictionResult, compute_accuracy_metrics
    from tests.benchmarks.core.memory import MemoryTracker

    probe_path = resolve_probe_path(variant["probe_subdir"])
    if not probe_path:
        print(f"   SKIPPED: Probe file not found for {model_size}")
        return {}
    if not torch.cuda.is_available():
        print("   SKIPPED: CUDA GPU required for Stage 3")
        return {}

    _cuda_cleanup()
    _gpu_mem_log(f"before cascade [1,3] {model_tag}+{model_size}")

    from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage3Config
    from lettucedetect.detectors.cascade import CascadeDetector

    config = CascadeConfig(
        stages=[1, 3],
        stage1=Stage1Config(model_path=model_path, augmentations=["lexical", "model2vec"]),
        stage3=Stage3Config(
            llm_model=variant["model"],
            probe_path=probe_path,
            layer_index=variant["layer_index"],
            token_position="mean",
        ),
    )
    cascade = CascadeDetector(config)
    cascade.warmup()
    _gpu_mem_log(f"cascade [1,3] {model_tag}+{model_size} loaded")

    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    stage1_resolved = 0
    stage3_resolved = 0

    with memory.track():
        for s in valid_samples:
            with timer.measure():
                result = cascade.predict(s.context, s.response, s.question, output_format="detailed")

            if isinstance(result, dict):
                score = result.get("scores", {}).get("final_score", 0.0)
                resolved_at = result.get("routing", {}).get("resolved_at_stage", 1)
                if resolved_at == 1:
                    stage1_resolved += 1
                else:
                    stage3_resolved += 1
            else:
                score = max((sp.get("confidence", 0.5) for sp in result), default=0.0)

            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(score >= 0.5), timer.last_ms, f"cascade_13_{model_tag}_{model_size}"))

    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)

    total = stage1_resolved + stage3_resolved
    key = f"cascade_13_{model_tag}_{model_size}"
    result = {
        **_make_result_dict(metrics, stats, mem_stats),
        "transformer_model": model_path,
        "llm_model": variant["model"],
        "stage1_resolved_pct": 100 * stage1_resolved / total if total > 0 else 0,
        "stage3_resolved_pct": 100 * stage3_resolved / total if total > 0 else 0,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, key),
    }
    _print_component_result(metrics, stats, mem_stats)
    if total > 0:
        print(f"   Routing: {stage1_resolved}/{total} ({100*stage1_resolved/total:.1f}%) resolved at Stage 1")

    del cascade
    _cuda_cleanup()
    _gpu_mem_log(f"after cleanup cascade [1,3] {model_tag}+{model_size}")

    return result


# ================================================================
# Summary + Save
# ================================================================


def print_summary(results: dict, title: str = "BENCHMARK SUMMARY"):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

    print("\n{:<24} {:>8} {:>8} {:>8} {:>10} {:>10} {:>8}".format(
        "Component", "AUROC", "F1@0.5", "OptF1", "Latency", "P95", "GPU MB"
    ))
    print("-" * 90)

    def _print_row(name, data):
        auroc = f"{data['auroc']:.3f}" if data.get("auroc") is not None else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get("f1") is not None else "N/A"
        opt_f1 = f"{data['optimal_f1']:.3f}" if data.get("optimal_f1") is not None else "N/A"
        latency = f"{data['latency_mean_ms']:.1f}ms"
        p95 = f"{data['latency_p95_ms']:.1f}ms"
        gpu = f"{data['gpu_peak_mb']:.0f}" if data.get("gpu_peak_mb") else "-"
        print(f"{name:<24} {auroc:>8} {f1:>8} {opt_f1:>8} {latency:>10} {p95:>10} {gpu:>8}")

    for name, data in results.get("components", {}).items():
        _print_row(name, data)
    if results.get("components"):
        print("-" * 90)

    for name, data in results.get("stages", {}).items():
        _print_row(name, data)
    if results.get("stages"):
        print("-" * 90)

    for name, data in results.get("cascade", {}).items():
        _print_row(name, data)
    print("=" * 90)

    # Per-task breakdown
    has_per_task = any(
        data.get("per_task")
        for section in ("components", "stages", "cascade")
        for data in results.get(section, {}).values()
    )
    if has_per_task:
        print("\n" + "=" * 90)
        print("PER-TASK BREAKDOWN")
        print("=" * 90)
        print("\n{:<24} {:<12} {:>8} {:>8} {:>8} {:>6}".format(
            "Component", "Task Type", "AUROC", "F1@0.5", "OptF1", "N"
        ))
        print("-" * 90)
        for section in ("components", "stages", "cascade"):
            for name, data in results.get(section, {}).items():
                per_task = data.get("per_task", {})
                if not per_task:
                    continue
                for task_type, tm in sorted(per_task.items()):
                    auroc = f"{tm['auroc']:.3f}" if tm.get("auroc") is not None else "N/A"
                    f1 = f"{tm['f1']:.3f}" if tm.get("f1") is not None else "N/A"
                    opt_f1 = f"{tm['optimal_f1']:.3f}" if tm.get("optimal_f1") is not None else "N/A"
                    n = tm.get("n_samples", 0)
                    print(f"{name:<24} {task_type:<12} {auroc:>8} {f1:>8} {opt_f1:>8} {n:>6}")
                print()
        print("=" * 90)


def _save_results(results: dict, output_dir: Path, filename: str):
    """Save results dict to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode (100 samples)")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Load all datasets (RAGTruth + HaluEval QA/Dialogue/Summarization)")
    parser.add_argument("--stage3", type=str, default=None,
                        choices=list(STAGE3_VARIANTS.keys()),
                        help="Run only this Stage 3 variant (default: all)")
    parser.add_argument("--transformer", type=str, default=None,
                        choices=list(TRANSFORMER_MODELS.keys()),
                        help="Run only this transformer model (default: both)")
    parser.add_argument("--output", type=str, default="tests/benchmarks/results", help="Output directory")
    args = parser.parse_args()

    limit = 100 if args.quick else args.limit

    # Load datasets
    if args.all_datasets:
        from tests.benchmarks.data_adapters.base import load_all_datasets

        print(f"Loading all datasets (limit={limit} per dataset)...")
        samples = load_all_datasets(limit=limit)
    else:
        from tests.benchmarks.data_adapters import RAGTruthAdapter

        print(f"Loading RAGTruth (limit={limit})...")
        adapter = RAGTruthAdapter()
        samples = adapter.load(limit=limit)

    valid_samples = [s for s in samples if s.context and s.response]
    print(f"Loaded {len(samples)} samples ({len(valid_samples)} valid)")

    task_counts = {}
    for s in valid_samples:
        key = s.task_type or "unknown"
        task_counts[key] = task_counts.get(key, 0) + 1
    print(f"Task types: {task_counts}")

    output_dir = Path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_start = time.time()

    # ================================================================
    # PHASE 1: Model-Independent Components
    # ================================================================
    phase1_start = time.time()
    phase1_results = run_model_independent_benchmarks(valid_samples)
    phase1_elapsed = time.time() - phase1_start

    # ================================================================
    # PHASE 2: Stage 3 Standalone (once per LLM)
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Stage 3 Standalone (LLM-only, run once per model)")
    print("=" * 60)

    variants = {args.stage3: STAGE3_VARIANTS[args.stage3]} if args.stage3 else STAGE3_VARIANTS
    stage3_cache = {}  # model_size -> result dict

    for model_size, variant in variants.items():
        print(f"\n[Stage 3] {model_size.upper()} ({variant['model']}) — PCA 512...")
        try:
            result = run_stage3_standalone(valid_samples, model_size, variant)
            if result:
                stage3_cache[model_size] = result
        except Exception as e:
            print(f"   FAILED: {model_size} standalone crashed: {e}")
            _cuda_cleanup()

    # ================================================================
    # PHASE 3: Transformer-Dependent Benchmarks
    # ================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Transformer-Dependent Benchmarks")
    print("=" * 60)

    transformers = (
        {args.transformer: TRANSFORMER_MODELS[args.transformer]}
        if args.transformer
        else TRANSFORMER_MODELS
    )

    for model_tag, model_path in transformers.items():
        print(f"\n--- TRANSFORMER: {model_tag} ({model_path}) ---")

        # Run transformer standalone + stage1 + cascade[1,2]
        try:
            transformer_results = run_transformer_benchmarks(valid_samples, model_path, model_tag)
        except Exception as e:
            print(f"   FAILED: {model_tag} transformer benchmarks crashed: {e}")
            _cuda_cleanup()
            transformer_results = {}

        # Run cascade[1,3] for each stage3 variant
        for model_size, variant in variants.items():
            print(f"\n  [Cascade 1+3] {model_tag} + {model_size.upper()}...")

            try:
                cascade_result = run_cascade_13(valid_samples, model_size, variant, model_path, model_tag)
            except Exception as e:
                print(f"   FAILED: cascade[1,3] {model_tag}+{model_size} crashed: {e}")
                _cuda_cleanup()
                cascade_result = {}

            # Merge Phase 1 + Phase 2 (cached) + Phase 3 results
            merged = copy.deepcopy(phase1_results)

            # Phase 2: stage3 standalone from cache
            if model_size in stage3_cache:
                merged["stages"][f"stage3_{model_size}"] = stage3_cache[model_size]

            # Phase 3: transformer-dependent results
            for key, val in transformer_results.items():
                if key.startswith("cascade_"):
                    merged["cascade"][key] = val
                elif key.startswith("stage1_"):
                    merged["stages"][key] = val
                else:
                    merged["components"][key] = val

            # Phase 3: cascade[1,3]
            if cascade_result:
                merged["cascade"][f"cascade_13_{model_tag}_{model_size}"] = cascade_result

            merged["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "limit": limit,
                    "all_datasets": args.all_datasets,
                    "n_samples": len(samples),
                    "n_valid": len(valid_samples),
                    "task_type_distribution": task_counts,
                },
                "transformer_model": model_tag,
                "transformer_path": model_path,
                "probe_variant": "pca512",
                "stage3_model": model_size,
                "stage3_llm": variant["model"],
                "phase1_time_sec": phase1_elapsed,
                "total_time_sec": time.time() - overall_start,
            }

            print_summary(merged, f"SUMMARY: {model_tag} + {model_size.upper()}")
            _save_results(merged, output_dir, f"benchmark_{model_tag}_{model_size}_{timestamp}.json")

        # Cleanup between transformer models
        _cuda_cleanup()
        _gpu_mem_log(f"after {model_tag} complete cleanup")

    total_elapsed = time.time() - overall_start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
