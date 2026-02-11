#!/usr/bin/env python3
"""Run full benchmark suite and save results to JSON.

Results are saved separately per Stage 3 model variant:
- benchmark_3b_{timestamp}.json: Components + Stage1/2 + Stage3(3B) + Cascade[1,3](3B)
- benchmark_8b_{timestamp}.json: Components + Stage1/2 + Stage3(8B) + Cascade[1,3](8B)

If a variant is unavailable (no probe file, no GPU, OOM), earlier variants are still saved.

Usage:
    python tests/benchmarks/run_full_benchmark.py --quick      # 100 samples
    python tests/benchmarks/run_full_benchmark.py --limit 500  # 500 samples
    python tests/benchmarks/run_full_benchmark.py              # Full dataset
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.core.stage3_variants import STAGE3_VARIANTS, resolve_probe_path


def _compute_per_task_metrics(predictions, valid_samples, component_name):
    """Compute metrics split by task_type for samples that have it."""
    from tests.benchmarks.core import compute_accuracy_metrics

    # Build sample_id -> task_type lookup
    task_map = {s.id: s.task_type for s in valid_samples if s.task_type}
    if not task_map:
        return {}

    # Group predictions by task type
    by_task = {}
    for p in predictions:
        task = task_map.get(p.sample_id)
        if task:
            by_task.setdefault(task, []).append(p)

    results = {}
    for task_type, task_preds in sorted(by_task.items()):
        metrics = compute_accuracy_metrics(task_preds, compute_ci=False)
        results[task_type] = {
            "auroc": metrics.auroc, "f1": metrics.f1,
            "optimal_f1": metrics.optimal_f1,
            "optimal_threshold": metrics.optimal_threshold,
            "n_samples": metrics.n_samples,
        }
        auroc_str = f"{metrics.auroc:.3f}" if metrics.auroc is not None else "N/A"
        f1_str = f"{metrics.f1:.3f}" if metrics.f1 is not None else "N/A"
        opt_f1_str = f"{metrics.optimal_f1:.3f}" if metrics.optimal_f1 is not None else "N/A"
        print(f"     {component_name} [{task_type}]: AUROC={auroc_str}, F1={f1_str}, OptF1={opt_f1_str}, n={metrics.n_samples}")

    return results




def _print_component_result(metrics, stats, mem_stats=None):
    """Print component benchmark result with None-safe AUROC formatting."""
    auroc = f"{metrics.auroc:.3f}" if metrics.auroc is not None else "N/A"
    parts = [f"   AUROC: {auroc}, Latency: {stats.mean_ms:.2f}ms"]
    if mem_stats is not None:
        parts.append(f"GPU: {mem_stats.gpu_peak_mb:.0f}MB")
    print(", ".join(parts))


def run_shared_benchmarks(valid_samples: list) -> dict:
    """Run component + stage1 + stage2 + cascade[1,2] benchmarks.

    These are shared across all Stage 3 variants.
    """
    from tests.benchmarks.core import BenchmarkTimer, PredictionResult, compute_accuracy_metrics
    from tests.benchmarks.core.memory import MemoryTracker

    results = {
        "components": {},
        "stages": {},
        "cascade": {},
    }

    # =========================================================
    # COMPONENT BENCHMARKS
    # =========================================================
    print("\n" + "=" * 60)
    print("COMPONENT BENCHMARKS")
    print("=" * 60)

    # --- Lexical ---
    print("\n[1/6] Lexical Overlap...")
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
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "lexical"),
    }
    _print_component_result(metrics, stats)

    # --- Numeric ---
    print("\n[2/6] Numeric Validator...")
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
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "numeric"),
    }
    _print_component_result(metrics, stats)

    # --- NER ---
    print("\n[3/6] NER Verifier...")
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
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "ner"),
    }
    _print_component_result(metrics, stats)

    # --- Transformer ---
    print("\n[4/6] Transformer Detector...")
    from lettucedetect.detectors.transformer import TransformerDetector
    transformer = TransformerDetector(model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    transformer.warmup()
    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                spans = transformer.predict(s.context, s.response, s.question, output_format="spans")
            score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(bool(spans)), timer.last_ms, "transformer"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["components"]["transformer"] = {
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "gpu_peak_mb": mem_stats.gpu_peak_mb,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "transformer"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del transformer

    # --- Model2Vec ---
    print("\n[5/6] Model2Vec (NCS)...")
    from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder
    m2v = Model2VecEncoder()
    m2v.preload()
    timer = BenchmarkTimer(sync_cuda=False)
    predictions = []
    for s in valid_samples:
        with timer.measure():
            ncs = m2v.compute_ncs(s.context, s.response)
        # Cosine similarity ranges [-1, 1], map to [0, 1] hallucination score
        score = (1.0 - ncs["max"]) / 2.0
        predictions.append(PredictionResult(s.id, s.ground_truth, score, int(score >= 0.5), timer.last_ms, "model2vec"))
    stats = timer.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["components"]["model2vec"] = {
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "model2vec"),
    }
    _print_component_result(metrics, stats)

    # --- NLI ---
    print("\n[6/6] NLI Detector...")
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
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "gpu_peak_mb": mem_stats.gpu_peak_mb,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "nli"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del nli

    # =========================================================
    # STAGE BENCHMARKS (1 + 2)
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE BENCHMARKS")
    print("=" * 60)

    # --- Stage 1 ---
    print("\n[Stage 1] Full pipeline with augmentations...")
    from lettucedetect.detectors.stage1.detector import Stage1Detector
    stage1 = Stage1Detector(augmentations=["lexical", "model2vec"])
    stage1.warmup()
    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                spans = stage1.predict(s.context, s.response, s.question, output_format="spans")
            score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(bool(spans)), timer.last_ms, "stage1"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["stages"]["stage1"] = {
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "gpu_peak_mb": mem_stats.gpu_peak_mb,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "stage1"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del stage1

    # --- Stage 2 ---
    print("\n[Stage 2] Full pipeline (NCS + NLI)...")
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
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "gpu_peak_mb": mem_stats.gpu_peak_mb,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "stage2"),
    }
    _print_component_result(metrics, stats, mem_stats)
    del stage2

    # =========================================================
    # CASCADE BENCHMARK (Stages 1+2)
    # =========================================================
    print("\n" + "=" * 60)
    print("CASCADE BENCHMARK (Stages 1+2)")
    print("=" * 60)

    from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage2Config
    from lettucedetect.detectors.cascade import CascadeDetector

    config = CascadeConfig(
        stages=[1, 2],
        stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
        stage2=Stage2Config(components=["ncs", "nli"]),
    )
    cascade = CascadeDetector(config)
    cascade._stages[1].warmup()
    cascade._stages[2].warmup()

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

            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(score >= 0.5), timer.last_ms, "cascade"))

    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)

    total = stage1_resolved + stage2_resolved
    results["cascade"]["stages_12"] = {
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "gpu_peak_mb": mem_stats.gpu_peak_mb,
        "n_samples": metrics.n_samples,
        "stage1_resolved_pct": 100 * stage1_resolved / total if total > 0 else 0,
        "stage2_resolved_pct": 100 * stage2_resolved / total if total > 0 else 0,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, "cascade_12"),
    }
    _print_component_result(metrics, stats, mem_stats)
    if total > 0:
        print(f"   Routing: {stage1_resolved}/{total} ({100*stage1_resolved/total:.1f}%) resolved at Stage 1")
    del cascade

    return results


def run_stage3_variant(valid_samples: list, model_size: str, variant: dict) -> dict:
    """Run Stage 3 standalone + Cascade[1,3] for a specific model variant.

    Returns dict with stage3 results and cascade[1,3] results, or empty dict on failure.
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

    results = {"stages": {}, "cascade": {}}

    gc.collect()
    torch.cuda.empty_cache()

    # Create cascade[1,3] which loads both stage1 and stage3
    # We reuse cascade._stages[3] for standalone stage3 benchmarking to avoid loading the LLM twice
    from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage3Config
    from lettucedetect.detectors.cascade import CascadeDetector

    print(f"\n   Loading cascade [1,3] with {variant['model']}...")
    config = CascadeConfig(
        stages=[1, 3],
        stage1=Stage1Config(augmentations=["lexical", "model2vec"]),
        stage3=Stage3Config(
            llm_model=variant["model"],
            probe_path=probe_path,
            layer_index=variant["layer_index"],
            token_position="mean",
        ),
    )
    cascade = CascadeDetector(config)
    cascade.warmup()

    # --- Stage 3 standalone ---
    print(f"\n   [Stage 3] Hallu Probe ({model_size.upper()}) standalone...")
    stage3 = cascade._stages[3]
    timer = BenchmarkTimer(sync_cuda=True)
    memory = MemoryTracker()
    predictions = []
    with memory.track():
        for s in valid_samples:
            with timer.measure():
                spans = stage3.predict(s.context, s.response, s.question, output_format="spans")
            score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(bool(spans)), timer.last_ms, f"stage3_{model_size}"))
    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)
    results["stages"][f"stage3_{model_size}"] = {
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "gpu_peak_mb": mem_stats.gpu_peak_mb,
        "n_samples": metrics.n_samples,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, f"stage3_{model_size}"),
    }
    _print_component_result(metrics, stats, mem_stats)

    # --- Cascade [1,3] ---
    print(f"\n   [Cascade 1+3] ({model_size.upper()})...")
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

            predictions.append(PredictionResult(s.id, s.ground_truth, score, int(score >= 0.5), timer.last_ms, f"cascade_13_{model_size}"))

    stats = timer.get_stats()
    mem_stats = memory.get_stats()
    metrics = compute_accuracy_metrics(predictions, compute_ci=False)

    total = stage1_resolved + stage3_resolved
    results["cascade"][f"stages_13_{model_size}"] = {
        "auroc": metrics.auroc, "f1": metrics.f1,
        "latency_mean_ms": stats.mean_ms, "latency_p95_ms": stats.p95_ms,
        "gpu_peak_mb": mem_stats.gpu_peak_mb,
        "n_samples": metrics.n_samples,
        "stage1_resolved_pct": 100 * stage1_resolved / total if total > 0 else 0,
        "stage3_resolved_pct": 100 * stage3_resolved / total if total > 0 else 0,
        "per_task": _compute_per_task_metrics(predictions, valid_samples, f"cascade_13_{model_size}"),
    }
    _print_component_result(metrics, stats, mem_stats)
    if total > 0:
        print(f"   Routing: {stage1_resolved}/{total} ({100*stage1_resolved/total:.1f}%) resolved at Stage 1")

    del cascade
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_summary(results: dict):
    """Print formatted summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print("\n{:<20} {:>10} {:>10} {:>12} {:>12} {:>10}".format(
        "Component", "AUROC", "F1", "Latency", "P95", "GPU MB"
    ))
    print("-" * 80)

    # Components
    for name, data in results["components"].items():
        auroc = f"{data['auroc']:.3f}" if data.get('auroc') is not None else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get('f1') is not None else "N/A"
        latency = f"{data['latency_mean_ms']:.2f}ms"
        p95 = f"{data['latency_p95_ms']:.2f}ms"
        gpu = f"{data['gpu_peak_mb']:.0f}" if data.get('gpu_peak_mb') else "-"
        print(f"{name:<20} {auroc:>10} {f1:>10} {latency:>12} {p95:>12} {gpu:>10}")

    print("-" * 80)

    # Stages
    for name, data in results["stages"].items():
        auroc = f"{data['auroc']:.3f}" if data.get('auroc') is not None else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get('f1') is not None else "N/A"
        latency = f"{data['latency_mean_ms']:.2f}ms"
        p95 = f"{data['latency_p95_ms']:.2f}ms"
        gpu = f"{data['gpu_peak_mb']:.0f}" if data.get('gpu_peak_mb') else "-"
        print(f"{name:<20} {auroc:>10} {f1:>10} {latency:>12} {p95:>12} {gpu:>10}")

    print("-" * 80)

    # Cascade
    for name, data in results["cascade"].items():
        auroc = f"{data['auroc']:.3f}" if data.get('auroc') is not None else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get('f1') is not None else "N/A"
        latency = f"{data['latency_mean_ms']:.2f}ms"
        p95 = f"{data['latency_p95_ms']:.2f}ms"
        gpu = f"{data['gpu_peak_mb']:.0f}" if data.get('gpu_peak_mb') else "-"
        print(f"{name:<20} {auroc:>10} {f1:>10} {latency:>12} {p95:>12} {gpu:>10}")

    print("=" * 80)

    # Per-task breakdown
    has_per_task = any(
        data.get("per_task")
        for section in ("components", "stages", "cascade")
        for data in results.get(section, {}).values()
    )
    if has_per_task:
        print("\n" + "=" * 80)
        print("PER-TASK BREAKDOWN")
        print("=" * 80)
        print("\n{:<20} {:<12} {:>10} {:>10} {:>10} {:>8}".format(
            "Component", "Task Type", "AUROC", "F1@0.5", "OptF1", "N"
        ))
        print("-" * 80)
        for section in ("components", "stages", "cascade"):
            for name, data in results.get(section, {}).items():
                per_task = data.get("per_task", {})
                if not per_task:
                    continue
                for task_type, tm in sorted(per_task.items()):
                    auroc = f"{tm['auroc']:.3f}" if tm.get('auroc') is not None else "N/A"
                    f1 = f"{tm['f1']:.3f}" if tm.get('f1') is not None else "N/A"
                    opt_f1 = f"{tm['optimal_f1']:.3f}" if tm.get('optimal_f1') is not None else "N/A"
                    n = tm.get('n_samples', 0)
                    print(f"{name:<20} {task_type:<12} {auroc:>10} {f1:>10} {opt_f1:>10} {n:>8}")
                print()
        print("=" * 80)



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

    # Show task type distribution
    task_counts = {}
    for s in valid_samples:
        key = s.task_type or "unknown"
        task_counts[key] = task_counts.get(key, 0) + 1
    print(f"Task types: {task_counts}")

    output_dir = Path(args.output)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run shared benchmarks (components + stage1/2 + cascade[1,2])
    start = time.time()
    shared_results = run_shared_benchmarks(valid_samples)
    shared_elapsed = time.time() - start

    shared_meta = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "limit": limit,
            "all_datasets": args.all_datasets,
            "n_samples": len(samples),
            "n_valid": len(valid_samples),
            "task_type_distribution": task_counts,
        },
        "shared_time_sec": shared_elapsed,
    }

    # Run Stage 3 variant(s)
    variants = {args.stage3: STAGE3_VARIANTS[args.stage3]} if args.stage3 else STAGE3_VARIANTS
    for model_size, variant in variants.items():
        print(f"\n{'='*60}")
        print(f"STAGE 3 VARIANT: {model_size.upper()} ({variant['model']})")
        print(f"{'='*60}")

        variant_start = time.time()
        try:
            variant_results = run_stage3_variant(valid_samples, model_size, variant)
        except Exception as e:
            print(f"\n   FAILED: {model_size} variant crashed: {e}")
            print(f"   Skipping to next variant...")
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        variant_elapsed = time.time() - variant_start

        # Merge shared + variant results
        merged = copy.deepcopy(shared_results)
        merged["stages"].update(variant_results.get("stages", {}))
        merged["cascade"].update(variant_results.get("cascade", {}))

        merged["metadata"] = {
            **shared_meta,
            "model_variant": model_size,
            "model_name": variant["model"],
            "variant_time_sec": variant_elapsed,
            "total_time_sec": shared_elapsed + variant_elapsed,
        }

        # Print and save
        print_summary(merged)
        _save_results(merged, output_dir, f"benchmark_{model_size}_{timestamp}.json")

    total_elapsed = time.time() - start
    print(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
