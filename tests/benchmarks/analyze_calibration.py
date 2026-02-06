#!/usr/bin/env python3
"""Analyze component calibration to inform confidence-gating thresholds.

This script runs benchmarks with per-sample output to analyze:
1. Stage 2's AUROC specifically on escalated samples
2. Per-component calibration curves (predicted probability vs actual frequency)
3. Which components are overconfident/underconfident in which direction

Run this BEFORE implementing confidence-gating to understand:
- Is Stage 2 actually hurting cascade performance?
- Which components need their predictions scaled/calibrated?
- Are components confident when wrong (dangerous) or uncertain when wrong (recoverable)?

Usage:
    python tests/benchmarks/analyze_calibration.py --output results/calibration/
    python tests/benchmarks/analyze_calibration.py --limit 500  # Quick test
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Disable torch.compile for older GPUs (GTX 1080 = CUDA 6.1, triton needs 7.0+)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from tests.benchmarks.data_adapters import BenchmarkSample, RAGTruthAdapter


@dataclass
class SamplePrediction:
    """Per-sample prediction data for calibration analysis."""

    sample_id: str
    ground_truth: int  # 0=factual, 1=hallucination
    # Per-component scores
    transformer_score: float | None = None
    lexical_score: float | None = None
    numeric_score: float | None = None
    ner_score: float | None = None
    model2vec_score: float | None = None
    nli_score: float | None = None
    # Stage/cascade scores
    stage1_score: float | None = None
    stage1_confident: bool | None = None
    stage2_score: float | None = None
    cascade_score: float | None = None
    resolved_at_stage: int | None = None


def load_dataset(limit: int | None = None) -> list[BenchmarkSample]:
    """Load RAGTruth dataset for calibration analysis."""
    print("Loading RAGTruth dataset...")
    adapter = RAGTruthAdapter()
    samples = adapter.load(limit=limit)
    print(f"  Loaded {len(samples)} samples")
    return samples


def _clear_gpu_memory():
    """Clear GPU memory by garbage collecting and emptying CUDA cache."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_predictions(samples: list[BenchmarkSample]) -> list[SamplePrediction]:
    """Run all components and collect per-sample predictions.

    Loads models sequentially to fit within 8GB VRAM:
    1. CPU components (lexical, numeric, ner) - no GPU
    2. Transformer - ~500MB
    3. Model2Vec - ~100MB
    4. NLI - ~3GB
    5. Cascade (Stage1 + Stage2) - shares NLI model
    """
    from lettucedetect.configs.models import CascadeConfig, Stage1Config, Stage2Config
    from lettucedetect.detectors.cascade import CascadeDetector
    from lettucedetect.detectors.stage1.augmentations.ner_verifier import NERVerifier
    from lettucedetect.detectors.stage1.augmentations.numeric_validator import NumericValidator
    from lettucedetect.detectors.stage1.detector import Stage1Detector
    from lettucedetect.detectors.stage2.detector import Stage2Detector
    from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder
    from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector
    from lettucedetect.detectors.transformer import TransformerDetector
    from lettucedetect.utils.lexical import LexicalOverlapCalculator

    # Initialize prediction list
    predictions = []
    total = len(samples)

    # Pre-filter valid samples
    valid_samples = [(i, s) for i, s in enumerate(samples) if s.context and s.response]
    for _, sample in valid_samples:
        predictions.append(SamplePrediction(
            sample_id=sample.id,
            ground_truth=sample.ground_truth,
        ))

    print(f"\nCollecting predictions for {len(valid_samples)} valid samples...")

    # ========================================
    # PHASE 1: CPU components (no GPU needed)
    # ========================================
    print("\n[Phase 1/4] CPU components (lexical, numeric, ner)...")
    lexical = LexicalOverlapCalculator()
    numeric = NumericValidator()
    ner = NERVerifier()
    ner.preload()

    for idx, (_, sample) in enumerate(valid_samples):
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(valid_samples)}...", end="\r")
        predictions[idx].lexical_score = lexical.score(sample.context, sample.response, sample.question, None).score
        predictions[idx].numeric_score = numeric.score(sample.context, sample.response, sample.question, None).score
        predictions[idx].ner_score = ner.score(sample.context, sample.response, sample.question, None).score
    print(f"  {len(valid_samples)}/{len(valid_samples)} done")

    # ========================================
    # PHASE 2: Transformer (~500MB GPU)
    # ========================================
    print("\n[Phase 2/4] Transformer...")
    _clear_gpu_memory()
    transformer = TransformerDetector(model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    transformer.warmup()

    for idx, (_, sample) in enumerate(valid_samples):
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(valid_samples)}...", end="\r")
        spans = transformer.predict(sample.context, sample.response, sample.question, output_format="spans")
        predictions[idx].transformer_score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
    print(f"  {len(valid_samples)}/{len(valid_samples)} done")

    # Unload transformer
    del transformer
    _clear_gpu_memory()

    # ========================================
    # PHASE 3: Model2Vec (~100MB GPU)
    # ========================================
    print("\n[Phase 3/4] Model2Vec...")
    model2vec = Model2VecEncoder()

    for idx, (_, sample) in enumerate(valid_samples):
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(valid_samples)}...", end="\r")
        ncs = model2vec.compute_ncs(sample.context, sample.response)
        predictions[idx].model2vec_score = (1.0 - ncs["max"]) / 2.0
    print(f"  {len(valid_samples)}/{len(valid_samples)} done")

    del model2vec
    _clear_gpu_memory()

    # ========================================
    # PHASE 4: NLI (~3GB GPU)
    # ========================================
    print("\n[Phase 4/4] NLI...")
    nli = NLIContradictionDetector()
    nli.preload()

    for idx, (_, sample) in enumerate(valid_samples):
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(valid_samples)}...", end="\r")
        predictions[idx].nli_score = nli.compute_context_nli(sample.context, sample.response)["hallucination_score"]
    print(f"  {len(valid_samples)}/{len(valid_samples)} done")

    del nli
    _clear_gpu_memory()

    # ========================================
    # PHASE 5: Compute Stage1/Stage2/Cascade from component scores
    # This avoids loading more models and gives us per-component analysis
    # ========================================
    print("\n[Phase 5/5] Computing stage scores from components...")

    # Stage 1 weights (from config)
    s1_weights = {"transformer": 0.5, "ner": 0.2, "numeric": 0.15, "lexical": 0.15}

    # Stage 2 weights
    s2_weights = {"model2vec": 0.4, "nli": 0.6}

    for idx, pred in enumerate(predictions):
        # Stage 1: weighted average of transformer + augmentations
        s1_sum = 0.0
        s1_total = 0.0
        for name, weight in s1_weights.items():
            score = getattr(pred, f"{name}_score", None)
            if score is not None:
                s1_sum += score * weight
                s1_total += weight
        pred.stage1_score = s1_sum / s1_total if s1_total > 0 else 0.5

        # Stage 1 confident if score < 0.3 or > 0.7
        pred.stage1_confident = pred.stage1_score < 0.3 or pred.stage1_score > 0.7

        # Stage 2: weighted average of model2vec + nli
        s2_sum = 0.0
        s2_total = 0.0
        for name, weight in s2_weights.items():
            score = getattr(pred, f"{name}_score", None)
            if score is not None:
                s2_sum += score * weight
                s2_total += weight
        pred.stage2_score = s2_sum / s2_total if s2_total > 0 else 0.5

        # Cascade: use stage1 if confident, otherwise blend with stage2
        if pred.stage1_confident:
            pred.cascade_score = pred.stage1_score
            pred.resolved_at_stage = 1
        else:
            # Blend stage1 and stage2 (50/50)
            pred.cascade_score = 0.5 * pred.stage1_score + 0.5 * pred.stage2_score
            pred.resolved_at_stage = 2

    print("  Done")

    print(f"\n  Collected {len(predictions)} predictions")
    return predictions


def analyze_stage2_on_escalated(predictions: list[SamplePrediction]) -> dict:
    """Measure Stage 2 AUROC only on samples it actually processes (escalated from Stage 1)."""
    print("\n" + "=" * 70)
    print("STAGE 2 ANALYSIS ON ESCALATED SAMPLES")
    print("=" * 70)

    # Samples where Stage 1 was NOT confident (escalated to Stage 2)
    escalated = [p for p in predictions if not p.stage1_confident]
    not_escalated = [p for p in predictions if p.stage1_confident]

    print(f"\nTotal samples: {len(predictions)}")
    print(f"Stage 1 confident (resolved): {len(not_escalated)} ({100*len(not_escalated)/len(predictions):.1f}%)")
    print(f"Stage 1 uncertain (escalated): {len(escalated)} ({100*len(escalated)/len(predictions):.1f}%)")

    # Stage 2 AUROC on escalated samples
    if len(escalated) >= 2:
        y_true = [p.ground_truth for p in escalated]
        y_stage2 = [p.stage2_score for p in escalated]
        y_cascade = [p.cascade_score for p in escalated]

        # Check for class balance
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        print(f"\nEscalated sample class balance: {n_pos} hallucinations, {n_neg} factual")

        if n_pos > 0 and n_neg > 0:
            stage2_auroc = roc_auc_score(y_true, y_stage2)
            cascade_auroc_escalated = roc_auc_score(y_true, y_cascade)
            print(f"\nStage 2 AUROC on escalated: {stage2_auroc:.3f}")
            print(f"Cascade AUROC on escalated: {cascade_auroc_escalated:.3f}")

            # Compare to Stage 1 would have done on these (counterfactual)
            y_stage1_escalated = [p.stage1_score for p in escalated]
            stage1_auroc_counterfactual = roc_auc_score(y_true, y_stage1_escalated)
            print(f"Stage 1 AUROC on escalated (counterfactual): {stage1_auroc_counterfactual:.3f}")
        else:
            stage2_auroc = None
            print("  Cannot compute AUROC - single class in escalated samples")
    else:
        stage2_auroc = None
        print("  Too few escalated samples for analysis")

    # Stage 1 AUROC on resolved (confident) samples
    if len(not_escalated) >= 2:
        y_true_resolved = [p.ground_truth for p in not_escalated]
        y_stage1_resolved = [p.stage1_score for p in not_escalated]
        n_pos = sum(y_true_resolved)
        n_neg = len(y_true_resolved) - n_pos

        if n_pos > 0 and n_neg > 0:
            stage1_auroc_resolved = roc_auc_score(y_true_resolved, y_stage1_resolved)
            print(f"\nStage 1 AUROC on resolved samples: {stage1_auroc_resolved:.3f}")
        else:
            stage1_auroc_resolved = None
    else:
        stage1_auroc_resolved = None

    return {
        "n_escalated": len(escalated),
        "n_resolved": len(not_escalated),
        "escalated_pct": 100 * len(escalated) / len(predictions),
        "stage2_auroc_on_escalated": stage2_auroc,
        "stage1_auroc_on_resolved": stage1_auroc_resolved,
    }


def plot_calibration_curves(predictions: list[SamplePrediction], output_dir: Path):
    """Plot calibration curves for all components."""
    print("\n" + "=" * 70)
    print("CALIBRATION CURVES")
    print("=" * 70)

    y_true = [p.ground_truth for p in predictions]

    components = [
        ("transformer", [p.transformer_score for p in predictions]),
        ("lexical", [p.lexical_score for p in predictions]),
        ("numeric", [p.numeric_score for p in predictions]),
        ("ner", [p.ner_score for p in predictions]),
        ("model2vec", [p.model2vec_score for p in predictions]),
        ("nli", [p.nli_score for p in predictions]),
        ("stage1", [p.stage1_score for p in predictions]),
        ("stage2", [p.stage2_score for p in predictions]),
        ("cascade", [p.cascade_score for p in predictions]),
    ]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    calibration_stats = {}

    for idx, (name, y_pred) in enumerate(components):
        if None in y_pred:
            print(f"  {name}: skipped (has None values)")
            continue

        ax = axes[idx]

        # Compute calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="uniform")

            # Plot
            ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=name)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(f"{name} (AUROC={roc_auc_score(y_true, y_pred):.3f})")
            ax.legend()
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)

            # Compute calibration error (ECE)
            bin_counts = []
            for i in range(len(prob_pred)):
                # Count samples in this bin
                lower = i / 10
                upper = (i + 1) / 10
                count = sum(1 for p in y_pred if lower <= p < upper)
                bin_counts.append(count)

            # Expected Calibration Error
            ece = np.mean(np.abs(np.array(prob_true) - np.array(prob_pred)))

            calibration_stats[name] = {
                "ece": float(ece),
                "auroc": float(roc_auc_score(y_true, y_pred)),
                "prob_true": [float(p) for p in prob_true],
                "prob_pred": [float(p) for p in prob_pred],
            }

            # Determine if over/underconfident
            mean_pred = np.mean(y_pred)
            mean_true = np.mean(y_true)
            if mean_pred > mean_true + 0.05:
                bias = "overconfident (predicts too many hallucinations)"
            elif mean_pred < mean_true - 0.05:
                bias = "underconfident (predicts too few hallucinations)"
            else:
                bias = "well-calibrated"

            print(f"  {name}: ECE={ece:.3f}, AUROC={calibration_stats[name]['auroc']:.3f}, {bias}")

        except Exception as e:
            print(f"  {name}: error - {e}")
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curves.png", dpi=150)
    plt.close()

    print(f"\n  Saved calibration_curves.png")
    return calibration_stats


def analyze_confidence_correlation(predictions: list[SamplePrediction]) -> dict:
    """Check if component confidence correlates with correctness."""
    print("\n" + "=" * 70)
    print("CONFIDENCE-CORRECTNESS CORRELATION")
    print("=" * 70)

    print("\nFor each component, we bin samples by confidence (|score - 0.5|)")
    print("and measure accuracy in each bin. Good components should have")
    print("higher accuracy when more confident.\n")

    components = [
        ("transformer", [p.transformer_score for p in predictions]),
        ("lexical", [p.lexical_score for p in predictions]),
        ("ner", [p.ner_score for p in predictions]),
        ("nli", [p.nli_score for p in predictions]),
        ("stage2", [p.stage2_score for p in predictions]),
    ]

    y_true = [p.ground_truth for p in predictions]
    threshold = 0.5

    correlation_stats = {}

    for name, y_pred in components:
        if None in y_pred:
            continue

        print(f"\n{name}:")

        # Compute confidence and correctness
        confidence = [abs(p - 0.5) for p in y_pred]
        predicted_label = [1 if p > threshold else 0 for p in y_pred]
        correct = [1 if pred == true else 0 for pred, true in zip(predicted_label, y_true)]

        # Bin by confidence
        bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
        bin_stats = []

        for low, high in bins:
            mask = [(low <= c < high) for c in confidence]
            n_samples = sum(mask)
            if n_samples > 0:
                acc = sum(c for c, m in zip(correct, mask) if m) / n_samples
                bin_stats.append({"range": f"[{low:.1f}, {high:.1f})", "n": n_samples, "accuracy": acc})
                print(f"  Confidence [{low:.1f}, {high:.1f}): {n_samples:4d} samples, accuracy {acc:.3f}")

        # Check monotonicity (higher confidence should mean higher accuracy)
        accuracies = [b["accuracy"] for b in bin_stats if b["n"] >= 10]
        if len(accuracies) >= 2:
            is_monotonic = all(accuracies[i] <= accuracies[i + 1] for i in range(len(accuracies) - 1))
            if is_monotonic:
                print("  [OK] Accuracy increases with confidence")
            else:
                print("  [WARN] Accuracy does NOT increase with confidence - component may be unreliable")

        correlation_stats[name] = bin_stats

    return correlation_stats


def _convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_results(
    predictions: list[SamplePrediction],
    escalation_stats: dict,
    calibration_stats: dict,
    correlation_stats: dict,
    output_dir: Path,
):
    """Save all results to JSON."""
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(predictions),
        },
        "escalation_analysis": _convert_to_serializable(escalation_stats),
        "calibration_stats": _convert_to_serializable(calibration_stats),
        "confidence_correlation": _convert_to_serializable(correlation_stats),
        "per_sample_predictions": [_convert_to_serializable(asdict(p)) for p in predictions],
    }

    output_path = output_dir / "calibration_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {output_path}")


def print_recommendations(escalation_stats: dict, calibration_stats: dict):
    """Print actionable recommendations based on analysis."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR CONFIDENCE-GATING")
    print("=" * 70)

    # Check if Stage 2 is helping
    if escalation_stats.get("stage2_auroc_on_escalated"):
        stage2_auroc = escalation_stats["stage2_auroc_on_escalated"]
        if stage2_auroc < 0.6:
            print("\n[CRITICAL] Stage 2 AUROC on escalated samples is LOW ({:.3f})".format(stage2_auroc))
            print("  -> Stage 2 may be hurting cascade performance")
            print("  -> Consider: skip Stage 2 when Stage 1 is uncertain")
            print("  -> Or: only use Stage 2 components with AUROC > 0.65")
        elif stage2_auroc < 0.7:
            print("\n[WARN] Stage 2 AUROC on escalated samples is MODERATE ({:.3f})".format(stage2_auroc))
            print("  -> Stage 2 provides some value but not strong")
            print("  -> Consider: higher confidence thresholds for Stage 2")
        else:
            print("\n[OK] Stage 2 AUROC on escalated samples is GOOD ({:.3f})".format(stage2_auroc))

    # Component-specific recommendations
    if calibration_stats:
        print("\nPer-component confidence threshold recommendations:")
        for name, stats in calibration_stats.items():
            auroc = stats.get("auroc", 0)
            ece = stats.get("ece", 0)

            if auroc < 0.65:
                print(f"  {name}: AUROC={auroc:.3f} - consider EXCLUDING or high threshold (0.35+)")
            elif auroc < 0.75:
                print(f"  {name}: AUROC={auroc:.3f} - use moderate threshold (0.25)")
            else:
                print(f"  {name}: AUROC={auroc:.3f} - use low threshold (0.15)")


def main():
    parser = argparse.ArgumentParser(description="Analyze component calibration for confidence-gating")
    parser.add_argument("--output", type=str, default="tests/benchmarks/results/calibration", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for quick testing)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CALIBRATION ANALYSIS FOR CONFIDENCE-GATING")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(f"Output: {output_dir}")
    if args.limit:
        print(f"Sample limit: {args.limit}")

    start_time = time.time()

    # Load data
    samples = load_dataset(limit=args.limit)

    # Collect predictions
    predictions = collect_predictions(samples)

    # Analysis
    escalation_stats = analyze_stage2_on_escalated(predictions)
    calibration_stats = plot_calibration_curves(predictions, output_dir)
    correlation_stats = analyze_confidence_correlation(predictions)

    # Save results
    save_results(predictions, escalation_stats, calibration_stats, correlation_stats, output_dir)

    # Recommendations
    print_recommendations(escalation_stats, calibration_stats)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
