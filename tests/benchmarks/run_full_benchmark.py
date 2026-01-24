#!/usr/bin/env python3
"""Run full benchmark suite and save results to JSON.

Usage:
    python tests/benchmarks/run_full_benchmark.py --quick      # 100 samples
    python tests/benchmarks/run_full_benchmark.py --limit 500  # 500 samples
    python tests/benchmarks/run_full_benchmark.py              # Full dataset
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def run_benchmarks(limit: int | None = None) -> dict:
    """Run all benchmarks and collect results."""
    from tests.benchmarks.core import BenchmarkTimer, PredictionResult, compute_accuracy_metrics
    from tests.benchmarks.core.memory import MemoryTracker, get_gpu_memory_mb
    from tests.benchmarks.data_adapters import RAGTruthAdapter

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {"limit": limit},
        "components": {},
        "stages": {},
        "cascade": {},
    }

    # Load dataset
    print(f"Loading RAGTruth (limit={limit})...")
    adapter = RAGTruthAdapter()
    samples = adapter.load(limit=limit)
    print(f"Loaded {len(samples)} samples")
    results["config"]["n_samples"] = len(samples)

    # Filter valid samples
    valid_samples = [s for s in samples if s.context and s.response]
    print(f"Valid samples: {len(valid_samples)}")

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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms")

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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms")

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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms")

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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms, GPU: {mem_stats.gpu_peak_mb:.0f}MB")
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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms")

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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms, GPU: {mem_stats.gpu_peak_mb:.0f}MB")
    del nli

    # =========================================================
    # STAGE BENCHMARKS
    # =========================================================
    print("\n" + "=" * 60)
    print("STAGE BENCHMARKS")
    print("=" * 60)

    # --- Stage 1 ---
    print("\n[Stage 1] Full pipeline with augmentations...")
    from lettucedetect.detectors.stage1.detector import Stage1Detector
    stage1 = Stage1Detector(augmentations=["ner", "numeric", "lexical"])
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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms, GPU: {mem_stats.gpu_peak_mb:.0f}MB")
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
        "n_samples": metrics.n_samples
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms, GPU: {mem_stats.gpu_peak_mb:.0f}MB")
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
        stage1=Stage1Config(augmentations=["ner", "numeric", "lexical"]),
        stage2=Stage2Config(components=["ncs", "nli"]),
    )
    cascade = CascadeDetector(config)
    cascade._stage1.warmup()
    cascade._stage2.warmup()

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
    }
    print(f"   AUROC: {metrics.auroc:.3f}, Latency: {stats.mean_ms:.2f}ms, GPU: {mem_stats.gpu_peak_mb:.0f}MB")
    print(f"   Routing: {stage1_resolved}/{total} ({100*stage1_resolved/total:.1f}%) resolved at Stage 1")

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
        auroc = f"{data['auroc']:.3f}" if data.get('auroc') else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get('f1') else "N/A"
        latency = f"{data['latency_mean_ms']:.2f}ms"
        p95 = f"{data['latency_p95_ms']:.2f}ms"
        gpu = f"{data['gpu_peak_mb']:.0f}" if data.get('gpu_peak_mb') else "-"
        print(f"{name:<20} {auroc:>10} {f1:>10} {latency:>12} {p95:>12} {gpu:>10}")

    print("-" * 80)

    # Stages
    for name, data in results["stages"].items():
        auroc = f"{data['auroc']:.3f}" if data.get('auroc') else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get('f1') else "N/A"
        latency = f"{data['latency_mean_ms']:.2f}ms"
        p95 = f"{data['latency_p95_ms']:.2f}ms"
        gpu = f"{data['gpu_peak_mb']:.0f}" if data.get('gpu_peak_mb') else "-"
        print(f"{name:<20} {auroc:>10} {f1:>10} {latency:>12} {p95:>12} {gpu:>10}")

    print("-" * 80)

    # Cascade
    for name, data in results["cascade"].items():
        auroc = f"{data['auroc']:.3f}" if data.get('auroc') else "N/A"
        f1 = f"{data['f1']:.3f}" if data.get('f1') else "N/A"
        latency = f"{data['latency_mean_ms']:.2f}ms"
        p95 = f"{data['latency_p95_ms']:.2f}ms"
        gpu = f"{data['gpu_peak_mb']:.0f}" if data.get('gpu_peak_mb') else "-"
        print(f"{name:<20} {auroc:>10} {f1:>10} {latency:>12} {p95:>12} {gpu:>10}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Quick mode (100 samples)")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit")
    parser.add_argument("--output", type=str, default="tests/benchmarks/results", help="Output directory")
    args = parser.parse_args()

    limit = 100 if args.quick else args.limit

    start = time.time()
    results = run_benchmarks(limit=limit)
    elapsed = time.time() - start

    results["total_time_sec"] = elapsed

    # Print summary
    print_summary(results)
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save to JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
