#!/usr/bin/env python3
"""Complete benchmark suite for LettuceDetect.

Runs all components, stages, and cascade on benchmark datasets (RAGTruth + HaluEval QA).
Saves comprehensive results to JSON files.

Expected runtime: ~1-2 hours on GTX 1080
Datasets: RAGTruth (2.7k) + HaluEval QA (10k) = ~12.7k samples

Usage:
    python tests/benchmarks/run_complete_benchmark.py
    python tests/benchmarks/run_complete_benchmark.py --output results/my_benchmark
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch

# Disable torch.compile for older GPUs (GTX 1080 = CUDA 6.1, Triton requires >= 7.0)
torch._dynamo.config.suppress_errors = True

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks.core.metrics import compute_accuracy_metrics
from tests.benchmarks.core.models import PredictionResult
from tests.benchmarks.core.timer import BenchmarkTimer
from tests.benchmarks.data_adapters import BenchmarkSample, HaluEvalAdapter, RAGTruthAdapter

# Default classification threshold
DEFAULT_THRESHOLD = 0.5


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ComponentSpec:
    """Specification for a benchmark component."""

    name: str
    factory: Callable[[], Any]
    predict: Callable[[Any, BenchmarkSample], float]
    use_gpu: bool = False
    warmup_method: str = "preload"  # "preload" or "warmup"
    result_key: str = "components"  # Where to store results: "components" or "stages"


@dataclass
class ComponentResult:
    """Results for a single component on a single dataset."""

    component: str
    dataset: str
    n_samples: int
    # Accuracy
    auroc: float | None
    auroc_ci_lower: float | None
    auroc_ci_upper: float | None
    f1: float | None
    precision: float | None
    recall: float | None
    optimal_threshold: float | None
    optimal_f1: float | None
    # Timing
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    # Memory
    gpu_peak_mb: float | None
    # Metadata
    elapsed_sec: float
    # Optional cascade-specific fields
    extra: dict = field(default_factory=dict)


class ProgressTracker:
    """Minimal progress tracking with time estimates."""

    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()

    def start_step(self, step_name: str):
        """Mark start of a step."""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        eta = self._estimate_eta(elapsed)
        print(
            f"\n[{self.current_step}/{self.total_steps}] {step_name} "
            f"(elapsed: {self._format_time(elapsed)}, ETA: {eta})"
        )

    def _estimate_eta(self, elapsed: float) -> str:
        if self.current_step <= 1:
            return "calculating..."
        rate = elapsed / (self.current_step - 1)
        remaining = rate * (self.total_steps - self.current_step + 1)
        return self._format_time(remaining)

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"


# =============================================================================
# Utility Functions
# =============================================================================


def print_progress(current: int, total: int, prefix: str = "", width: int = 30):
    """Print progress bar on single line (overwrites previous)."""
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    print(f"\r  {prefix} [{bar}] {current}/{total} ({100*pct:.0f}%)", end="", flush=True)


def clear_gpu():
    """Clear GPU memory between components."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


AVAILABLE_DATASETS = ["ragtruth", "halueval_qa"]


def load_all_datasets(
    limit: int | None = None, dataset_filter: list[str] | None = None
) -> dict[str, list[BenchmarkSample]]:
    """Load benchmark datasets.

    Args:
        limit: Maximum samples per dataset (None = all)
        dataset_filter: List of dataset names to load (None = all)
    """
    datasets = {}

    print("Loading datasets...")

    configs = [
        ("ragtruth", RAGTruthAdapter, {}),
        ("halueval_qa", HaluEvalAdapter, {"subset": "qa_samples"}),
    ]

    for name, adapter_cls, kwargs in configs:
        if dataset_filter and name not in dataset_filter:
            continue
        print(f"  - {name}...", end=" ", flush=True)
        adapter = adapter_cls(**kwargs)
        datasets[name] = adapter.load(limit=limit)
        print(f"{len(datasets[name])} samples")

    total = sum(len(d) for d in datasets.values())
    print(f"\nTotal: {total} samples across {len(datasets)} datasets")

    return datasets


# =============================================================================
# Core Benchmarking Functions
# =============================================================================


def benchmark_component(
    name: str,
    dataset_name: str,
    samples: list[BenchmarkSample],
    predict_fn: Callable[[BenchmarkSample], float],
    use_gpu: bool = False,
    compute_ci: bool = True,
    show_progress: bool = True,
    per_sample_callback: Callable[[BenchmarkSample, float, Any], dict] | None = None,
) -> ComponentResult:
    """Benchmark a single component on a dataset.

    Args:
        name: Component name for logging
        dataset_name: Dataset name for logging
        samples: List of samples to benchmark
        predict_fn: Function that takes a sample and returns a score
        use_gpu: Whether to sync CUDA for accurate GPU timing
        compute_ci: Whether to compute bootstrap confidence intervals
        show_progress: Whether to show progress bar
        per_sample_callback: Optional callback(sample, score, raw_result) -> dict
            for collecting extra per-sample metadata (e.g., routing info)

    Returns:
        ComponentResult with metrics, timing, and optional extra data
    """
    start_time = time.time()

    valid_samples = [s for s in samples if s.context and s.response]
    total = len(valid_samples)

    timer = BenchmarkTimer(sync_cuda=use_gpu)
    predictions = []
    extra_data = {}

    if use_gpu and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    update_interval = max(total // 20, 100)

    for i, sample in enumerate(valid_samples):
        with timer.measure():
            score = predict_fn(sample)

        predictions.append(
            PredictionResult(
                sample_id=sample.id,
                ground_truth=sample.ground_truth,
                predicted_score=score,
                predicted_label=1 if score >= DEFAULT_THRESHOLD else 0,
                latency_ms=timer.last_ms,
                component_name=name,
            )
        )

        if per_sample_callback:
            callback_data = per_sample_callback(sample, score, None)
            for key, value in callback_data.items():
                if key not in extra_data:
                    extra_data[key] = 0
                extra_data[key] += value

        if show_progress and (i + 1) % update_interval == 0:
            print_progress(i + 1, total, dataset_name)

    if show_progress:
        print_progress(total, total, dataset_name)
        print()

    timing = timer.get_stats()

    gpu_peak = None
    if use_gpu and torch.cuda.is_available():
        gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

    metrics = compute_accuracy_metrics(predictions, compute_ci=compute_ci)

    elapsed = time.time() - start_time

    return ComponentResult(
        component=name,
        dataset=dataset_name,
        n_samples=len(valid_samples),
        auroc=metrics.auroc,
        auroc_ci_lower=metrics.auroc_ci_lower,
        auroc_ci_upper=metrics.auroc_ci_upper,
        f1=metrics.f1,
        precision=metrics.precision,
        recall=metrics.recall,
        optimal_threshold=metrics.optimal_threshold,
        optimal_f1=metrics.optimal_f1,
        latency_mean_ms=timing.mean_ms,
        latency_std_ms=timing.std_ms,
        latency_min_ms=timing.min_ms,
        latency_max_ms=timing.max_ms,
        latency_p50_ms=timing.p50_ms,
        latency_p95_ms=timing.p95_ms,
        gpu_peak_mb=gpu_peak,
        elapsed_sec=elapsed,
        extra=extra_data,
    )


def run_component_suite(
    specs: list[ComponentSpec],
    datasets: dict[str, list[BenchmarkSample]],
    progress: ProgressTracker,
    compute_ci: bool,
) -> dict[str, list[dict]]:
    """Run benchmarks for a list of component specifications.

    Args:
        specs: List of ComponentSpec defining what to benchmark
        datasets: Dict of dataset_name -> samples
        progress: Progress tracker for step updates
        compute_ci: Whether to compute bootstrap confidence intervals

    Returns:
        Dict with "components" and "stages" lists of result dicts
    """
    results = {"components": [], "stages": []}

    for spec in specs:
        # Create and warm up component
        component = spec.factory()
        warmup_fn = getattr(component, spec.warmup_method, None)
        if warmup_fn:
            warmup_fn()

        # Create predict function bound to this component
        def make_predict_fn(comp, spec_predict):
            return lambda sample: spec_predict(comp, sample)

        predict_fn = make_predict_fn(component, spec.predict)

        # Run on each dataset
        for ds_name, samples in datasets.items():
            progress.start_step(f"{spec.name} -> {ds_name}")
            result = benchmark_component(
                spec.name,
                ds_name,
                samples,
                predict_fn,
                use_gpu=spec.use_gpu,
                compute_ci=compute_ci,
            )
            results[spec.result_key].append(asdict(result))

            # Print result summary
            gpu_info = f", GPU={result.gpu_peak_mb:.0f}MB" if result.gpu_peak_mb else ""
            f1_info = f", F1={result.optimal_f1:.3f}@{result.optimal_threshold:.2f}" if result.optimal_f1 else ""
            print(f"  Result: AUROC={result.auroc:.3f}{f1_info}, latency={result.latency_mean_ms:.2f}ms{gpu_info}")

        # Cleanup
        del component
        clear_gpu()

    return results


def benchmark_cascade(
    datasets: dict[str, list[BenchmarkSample]],
    progress: ProgressTracker,
    compute_ci: bool,
) -> list[dict]:
    """Benchmark the full cascade with routing tracking.

    This is separate from run_component_suite because cascade needs to track
    which stage resolved each sample, requiring custom per-sample logic.
    """
    from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage2Config
    from lettucedetect.detectors.cascade import CascadeDetector

    config = CascadeConfig(
        stages=[1, 2],
        stage1=Stage1Config(augmentations=["ner", "numeric", "lexical"]),
        stage2=Stage2Config(components=["ncs", "nli"]),
    )
    cascade = CascadeDetector(config)
    cascade.warmup()

    results = []

    for ds_name, samples in datasets.items():
        progress.start_step(f"Cascade -> {ds_name}")

        # Track routing decisions
        routing_counts = {"stage1_resolved": 0, "stage2_resolved": 0}

        def predict_with_routing(sample: BenchmarkSample) -> float:
            result = cascade.predict(
                sample.context, sample.response, sample.question, output_format="detailed"
            )

            if isinstance(result, dict):
                score = result.get("scores", {}).get("final_score", 0.0)
                resolved_at = result.get("routing", {}).get("resolved_at_stage", 1)
                if resolved_at == 1:
                    routing_counts["stage1_resolved"] += 1
                else:
                    routing_counts["stage2_resolved"] += 1
            else:
                score = max((sp.get("confidence", 0.5) for sp in result), default=0.0)
                routing_counts["stage1_resolved"] += 1  # Default assumption

            return score

        result = benchmark_component(
            "cascade_stages12",
            ds_name,
            samples,
            predict_with_routing,
            use_gpu=True,
            compute_ci=compute_ci,
        )

        # Add routing stats to result
        result_dict = asdict(result)
        total_resolved = routing_counts["stage1_resolved"] + routing_counts["stage2_resolved"]
        result_dict["stage1_resolved"] = routing_counts["stage1_resolved"]
        result_dict["stage2_resolved"] = routing_counts["stage2_resolved"]
        result_dict["stage1_resolved_pct"] = (
            100 * routing_counts["stage1_resolved"] / total_resolved if total_resolved > 0 else 0
        )

        results.append(result_dict)

        stage1_pct = result_dict["stage1_resolved_pct"]
        f1_info = f", F1={result.optimal_f1:.3f}@{result.optimal_threshold:.2f}" if result.optimal_f1 else ""
        print(f"  Result: AUROC={result.auroc:.3f}{f1_info}, latency={result.latency_mean_ms:.2f}ms, Stage1={stage1_pct:.1f}%")

    del cascade
    clear_gpu()

    return results


# =============================================================================
# Component Definitions
# =============================================================================


def get_component_specs() -> list[ComponentSpec]:
    """Define all components to benchmark."""

    # Lazy imports to avoid loading everything at startup
    def make_lexical():
        from lettucedetect.utils.lexical import LexicalOverlapCalculator

        return LexicalOverlapCalculator()

    def make_numeric():
        from lettucedetect.detectors.stage1.augmentations.numeric_validator import NumericValidator

        return NumericValidator()

    def make_ner():
        from lettucedetect.detectors.stage1.augmentations.ner_verifier import NERVerifier

        return NERVerifier()

    def make_transformer():
        from lettucedetect.detectors.transformer import TransformerDetector

        return TransformerDetector(model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")

    def make_model2vec():
        from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder

        return Model2VecEncoder()

    def make_nli():
        from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector

        return NLIContradictionDetector()

    def make_stage1():
        from lettucedetect.detectors.stage1.detector import Stage1Detector

        return Stage1Detector(augmentations=["ner", "numeric", "lexical"])

    def make_stage2():
        from lettucedetect.detectors.stage2.detector import Stage2Detector

        return Stage2Detector()

    # Predict functions
    def predict_lexical(comp, s):
        return comp.score(s.context, s.response, s.question, None).score

    def predict_numeric(comp, s):
        return comp.score(s.context, s.response, s.question, None).score

    def predict_ner(comp, s):
        return comp.score(s.context, s.response, s.question, None).score

    def predict_transformer(comp, s):
        spans = comp.predict(s.context, s.response, s.question, output_format="spans")
        return max((sp.get("confidence", 0.5) for sp in spans), default=0.0)

    def predict_model2vec(comp, s):
        ncs = comp.compute_ncs(s.context, s.response)
        # Cosine similarity ranges [-1, 1], map to [0, 1] hallucination score
        # ncs=1 (identical) -> 0, ncs=0 (orthogonal) -> 0.5, ncs=-1 (opposite) -> 1
        return (1.0 - ncs["max"]) / 2.0

    def predict_nli(comp, s):
        return comp.compute_context_nli(s.context, s.response)["hallucination_score"]

    def predict_stage(comp, s):
        from lettucedetect.cascade.types import CascadeInput

        cascade_input = CascadeInput(context=s.context, answer=s.response, question=s.question)
        result = comp.predict_stage(input=cascade_input, has_next_stage=False)
        return result.hallucination_score

    return [
        # Stage 1 components
        ComponentSpec("lexical", make_lexical, predict_lexical, use_gpu=False),
        ComponentSpec("numeric", make_numeric, predict_numeric, use_gpu=False),
        ComponentSpec("ner", make_ner, predict_ner, use_gpu=False),
        ComponentSpec("transformer", make_transformer, predict_transformer, use_gpu=True, warmup_method="warmup"),
        # Stage 2 components
        ComponentSpec("model2vec", make_model2vec, predict_model2vec, use_gpu=False),
        ComponentSpec("nli", make_nli, predict_nli, use_gpu=True),
        # Full stages
        ComponentSpec("stage1", make_stage1, predict_stage, use_gpu=True, warmup_method="warmup", result_key="stages"),
        ComponentSpec("stage2", make_stage2, predict_stage, use_gpu=True, warmup_method="warmup", result_key="stages"),
    ]


# =============================================================================
# Summary and Output
# =============================================================================


def compute_summaries(results: dict) -> dict:
    """Compute weighted average summaries across datasets."""
    summary = {}

    def weighted_avg(items: list[dict], key: str, total: int) -> float | None:
        """Compute weighted average, skipping None values."""
        valid = [(r[key], r["n_samples"]) for r in items if r.get(key) is not None]
        if not valid:
            return None
        return sum(v * n for v, n in valid) / total

    # Component summaries
    component_names = set(r["component"] for r in results["components"])
    for comp_name in component_names:
        comp_results = [r for r in results["components"] if r["component"] == comp_name]
        total_samples = sum(r["n_samples"] for r in comp_results)

        if total_samples > 0:
            summary[comp_name] = {
                "avg_auroc": weighted_avg(comp_results, "auroc", total_samples),
                "avg_f1": weighted_avg(comp_results, "f1", total_samples),
                "avg_optimal_f1": weighted_avg(comp_results, "optimal_f1", total_samples),
                "avg_optimal_threshold": weighted_avg(comp_results, "optimal_threshold", total_samples),
                "avg_latency_mean_ms": weighted_avg(comp_results, "latency_mean_ms", total_samples),
                "avg_latency_p95_ms": weighted_avg(comp_results, "latency_p95_ms", total_samples),
                "total_samples": total_samples,
            }

    # Stage summaries
    for stage_name in ["stage1", "stage2"]:
        stage_results = [r for r in results["stages"] if r["component"] == stage_name]
        if stage_results:
            total_samples = sum(r["n_samples"] for r in stage_results)
            if total_samples > 0:
                summary[stage_name] = {
                    "avg_auroc": weighted_avg(stage_results, "auroc", total_samples),
                    "avg_f1": weighted_avg(stage_results, "f1", total_samples),
                    "avg_optimal_f1": weighted_avg(stage_results, "optimal_f1", total_samples),
                    "avg_optimal_threshold": weighted_avg(stage_results, "optimal_threshold", total_samples),
                    "avg_latency_mean_ms": weighted_avg(stage_results, "latency_mean_ms", total_samples),
                    "total_samples": total_samples,
                }

    # Cascade summary
    cascade_results = results["cascade"]
    if cascade_results:
        total_samples = sum(r["n_samples"] for r in cascade_results)
        if total_samples > 0:
            summary["cascade_stages12"] = {
                "avg_auroc": weighted_avg(cascade_results, "auroc", total_samples),
                "avg_f1": weighted_avg(cascade_results, "f1", total_samples),
                "avg_optimal_f1": weighted_avg(cascade_results, "optimal_f1", total_samples),
                "avg_optimal_threshold": weighted_avg(cascade_results, "optimal_threshold", total_samples),
                "avg_latency_mean_ms": weighted_avg(cascade_results, "latency_mean_ms", total_samples),
                "avg_stage1_resolved_pct": weighted_avg(cascade_results, "stage1_resolved_pct", total_samples),
                "total_samples": total_samples,
            }

    return summary


def print_summary_table(results: dict):
    """Print formatted summary table."""
    print("\n" + "=" * 110)
    print("BENCHMARK SUMMARY (Weighted Average Across All Datasets)")
    print("=" * 110)

    summary = results["summary"]

    # Header
    print(f"\n{'Component':<20} {'AUROC':>8} {'F1':>8} {'Threshold':>10} {'Latency':>10} {'P95':>10} {'Samples':>8}")
    print("-" * 110)

    # Components
    for comp in ["lexical", "numeric", "ner", "transformer", "model2vec", "nli"]:
        if comp in summary:
            s = summary[comp]
            auroc = f"{s['avg_auroc']:.3f}" if s.get("avg_auroc") else "N/A"
            f1 = f"{s['avg_optimal_f1']:.3f}" if s.get("avg_optimal_f1") else "N/A"
            threshold = f"{s.get('avg_optimal_threshold', 0.5):.3f}" if s.get("avg_optimal_threshold") else "N/A"
            latency = f"{s['avg_latency_mean_ms']:.1f}ms"
            p95 = f"{s.get('avg_latency_p95_ms', 0):.1f}ms"
            samples = str(s.get("total_samples", 0))
            print(f"{comp:<20} {auroc:>8} {f1:>8} {threshold:>10} {latency:>10} {p95:>10} {samples:>8}")

    print("-" * 110)

    # Stages
    for stage in ["stage1", "stage2"]:
        if stage in summary:
            s = summary[stage]
            auroc = f"{s['avg_auroc']:.3f}" if s.get("avg_auroc") else "N/A"
            f1 = f"{s['avg_optimal_f1']:.3f}" if s.get("avg_optimal_f1") else "N/A"
            threshold = f"{s.get('avg_optimal_threshold', 0.5):.3f}" if s.get("avg_optimal_threshold") else "N/A"
            latency = f"{s['avg_latency_mean_ms']:.1f}ms"
            samples = str(s.get("total_samples", 0))
            print(f"{stage:<20} {auroc:>8} {f1:>8} {threshold:>10} {latency:>10} {'-':>10} {samples:>8}")

    print("-" * 110)

    # Cascade
    if "cascade_stages12" in summary:
        s = summary["cascade_stages12"]
        auroc = f"{s['avg_auroc']:.3f}" if s.get("avg_auroc") else "N/A"
        f1 = f"{s['avg_optimal_f1']:.3f}" if s.get("avg_optimal_f1") else "N/A"
        threshold = f"{s.get('avg_optimal_threshold', 0.5):.3f}" if s.get("avg_optimal_threshold") else "N/A"
        latency = f"{s['avg_latency_mean_ms']:.1f}ms"
        stage1_pct = f"{s.get('avg_stage1_resolved_pct', 0):.0f}%S1"
        samples = str(s.get("total_samples", 0))
        print(f"{'cascade (1+2)':<20} {auroc:>8} {f1:>8} {threshold:>10} {latency:>10} {stage1_pct:>10} {samples:>8}")

    print("=" * 110)


def save_results(results: dict, output_dir: Path):
    """Save results to multiple files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full results
    full_path = output_dir / f"benchmark_full_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results: {full_path}")

    # Summary only
    summary_path = output_dir / f"benchmark_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump({"metadata": results["metadata"], "summary": results["summary"]}, f, indent=2)
    print(f"Summary: {summary_path}")

    # CSV for easy import
    csv_path = output_dir / f"benchmark_components_{timestamp}.csv"
    with open(csv_path, "w") as f:
        f.write("component,dataset,n_samples,auroc,f1,optimal_f1,latency_mean_ms,latency_p95_ms,gpu_peak_mb\n")
        for r in results["components"] + results["stages"] + results["cascade"]:
            f.write(
                f"{r['component']},{r['dataset']},{r['n_samples']},"
                f"{r.get('auroc', '')},{r.get('f1', '')},{r.get('optimal_f1', '')},"
                f"{r['latency_mean_ms']:.2f},{r['latency_p95_ms']:.2f},{r.get('gpu_peak_mb', '')}\n"
            )
    print(f"CSV: {csv_path}")

    # Timing summary
    timing_path = output_dir / f"benchmark_timing_{timestamp}.txt"
    with open(timing_path, "w") as f:
        f.write("TIMING SUMMARY (ms)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Component':<25} {'Mean':>10} {'Std':>10} {'P50':>10} {'P95':>10}\n")
        f.write("-" * 60 + "\n")
        for r in results["components"] + results["stages"] + results["cascade"]:
            f.write(
                f"{r['component']:<25} {r['latency_mean_ms']:>10.2f} "
                f"{r['latency_std_ms']:>10.2f} {r['latency_p50_ms']:>10.2f} "
                f"{r['latency_p95_ms']:>10.2f}\n"
            )
    print(f"Timing: {timing_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_benchmarks(
    datasets: dict[str, list[BenchmarkSample]],
    compute_ci: bool = True,
    component_filter: list[str] | None = None,
) -> dict:
    """Run all benchmarks.

    Args:
        datasets: Dict mapping dataset name to list of samples
        compute_ci: Whether to compute bootstrap confidence intervals
        component_filter: Optional list of component names to benchmark (default: all)
    """
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "datasets": {name: len(samples) for name, samples in datasets.items()},
            "total_samples": sum(len(s) for s in datasets.values()),
            "compute_ci": compute_ci,
            "component_filter": component_filter,
        },
        "components": [],
        "stages": [],
        "cascade": [],
        "summary": {},
    }

    all_specs = get_component_specs()

    # Filter specs if component_filter is provided
    if component_filter:
        specs = [s for s in all_specs if s.name in component_filter]
        run_cascade = "cascade" in component_filter
    else:
        specs = all_specs
        run_cascade = True

    n_datasets = len(datasets)
    # Calculate total steps based on filtered specs
    total_steps = len(specs) * n_datasets + (n_datasets if run_cascade else 0)
    progress = ProgressTracker(total_steps)

    # Run components and stages
    if specs:
        print("\n" + "=" * 70)
        print("RUNNING COMPONENT AND STAGE BENCHMARKS")
        print("=" * 70)

        component_results = run_component_suite(specs, datasets, progress, compute_ci)
        results["components"] = component_results["components"]
        results["stages"] = component_results["stages"]

    # Run cascade
    if run_cascade:
        print("\n" + "=" * 70)
        print("RUNNING CASCADE BENCHMARK")
        print("=" * 70)

        results["cascade"] = benchmark_cascade(datasets, progress, compute_ci)

    # Compute summaries
    print("\nComputing summaries...")
    results["summary"] = compute_summaries(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark suite")
    parser.add_argument("--output", type=str, default="tests/benchmarks/results", help="Output directory")
    parser.add_argument("--no-ci", action="store_true", help="Skip bootstrap CI (faster)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per dataset (for quick testing)")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        choices=AVAILABLE_DATASETS,
        default=None,
        help=f"Datasets to benchmark (default: all). Choices: {', '.join(AVAILABLE_DATASETS)}",
    )
    parser.add_argument(
        "--component",
        type=str,
        nargs="+",
        default=None,
        help="Components to benchmark (default: all). Choices: lexical, numeric, ner, transformer, model2vec, nli, stage1, stage2, cascade",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LETTUCEDETECT BENCHMARK SUITE")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(f"Output: {args.output}")
    print(f"Bootstrap CI: {'No (--no-ci)' if args.no_ci else 'Yes'}")
    if args.dataset:
        print(f"Datasets: {', '.join(args.dataset)}")
    if args.component:
        print(f"Components: {', '.join(args.component)}")
    if args.limit:
        print(f"Sample limit: {args.limit} per dataset")

    start_time = time.time()

    datasets = load_all_datasets(limit=args.limit, dataset_filter=args.dataset)
    results = run_benchmarks(datasets, compute_ci=not args.no_ci, component_filter=args.component)

    total_time = time.time() - start_time
    results["metadata"]["total_time_sec"] = total_time
    results["metadata"]["total_time_hours"] = total_time / 3600

    print_summary_table(results)
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

    save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
