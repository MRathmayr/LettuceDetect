#!/usr/bin/env python3
"""Analyze score blending strategies for Stage 1 + Stage 3 cascade.

Loads benchmark JSONs with per-sample predictions (--save-predictions),
tests blending strategies to recover F1 while preserving AUROC gains.

Strategies:
  1. Weighted average (all samples): alpha * s1 + (1-alpha) * s3
  2. Weighted average (escalated only): keep s1 for confident, blend for escalated
  3. Max rule: max(s1, s3)
  4. Product rule: 1 - (1-s1)*(1-s3)
  5. Routing threshold sweep: vary confident zone boundaries

Usage:
    python tests/benchmarks/analyze_score_blending.py \
        --results-dir tests/benchmarks/results/benchmark_all_halluprobe_baseline \
        --output blending_analysis.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def load_predictions(json_path: Path) -> dict:
    """Load benchmark JSON and extract per-sample predictions.

    Returns dict with keys: stage1, stage3, cascade, metadata.
    Each value is a list of {sample_id, ground_truth, predicted_score, task_type}.
    """
    with open(json_path) as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    transformer_tag = meta.get("transformer_model", "unknown")
    stage3_size = meta.get("stage3_model", "unknown")

    result = {"metadata": {"transformer": transformer_tag, "stage3": stage3_size, "file": str(json_path)}}

    # Stage 1 predictions
    s1_key = f"stage1_{transformer_tag}"
    s1_data = data.get("stages", {}).get(s1_key, {})
    result["stage1"] = s1_data.get("predictions")

    # Stage 3 predictions
    s3_key = f"stage3_{stage3_size}"
    s3_data = data.get("stages", {}).get(s3_key, {})
    result["stage3"] = s3_data.get("predictions")

    # Cascade predictions
    c13_key = f"cascade_13_{transformer_tag}_{stage3_size}"
    c13_data = data.get("cascade", {}).get(c13_key, {})
    result["cascade"] = c13_data.get("predictions")

    # Stage 1 baseline metrics (for comparison)
    result["stage1_metrics"] = {
        "auroc": s1_data.get("auroc"),
        "f1": s1_data.get("f1"),
        "optimal_f1": s1_data.get("optimal_f1"),
    }
    result["cascade_metrics"] = {
        "auroc": c13_data.get("auroc"),
        "f1": c13_data.get("f1"),
        "optimal_f1": c13_data.get("optimal_f1"),
    }

    return result


def match_predictions(s1_preds: list, s3_preds: list) -> list[dict]:
    """Match stage1 and stage3 predictions by sample_id.

    Returns list of {sample_id, s1_score, s3_score, ground_truth, task_type}.
    """
    s3_map = {p["sample_id"]: p for p in s3_preds}
    matched = []
    for p1 in s1_preds:
        sid = p1["sample_id"]
        p3 = s3_map.get(sid)
        if p3 is None:
            continue
        matched.append({
            "sample_id": sid,
            "s1_score": p1["predicted_score"],
            "s3_score": p3["predicted_score"],
            "ground_truth": p1["ground_truth"],
            "task_type": p1.get("task_type", "unknown"),
        })
    return matched


def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute AUROC, F1@threshold, and optimal F1."""
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "f1": None, "optimal_f1": None, "optimal_threshold": None}

    auroc = roc_auc_score(labels, scores)
    preds = (scores >= threshold).astype(int)
    f1 = f1_score(labels, preds, zero_division=0)

    # Optimal F1
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        p = (scores >= t).astype(int)
        f = f1_score(labels, p, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t

    return {
        "auroc": round(auroc, 4),
        "f1": round(f1, 4),
        "optimal_f1": round(best_f1, 4),
        "optimal_threshold": round(best_thresh, 3),
    }


def compute_per_task_metrics(scores: np.ndarray, labels: np.ndarray,
                             task_types: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute metrics per task type."""
    results = {}
    for task in np.unique(task_types):
        mask = task_types == task
        if mask.sum() < 10:
            continue
        results[task] = compute_metrics(scores[mask], labels[mask], threshold)
    return results


def strategy_weighted_avg_all(matched: list, alpha: float) -> np.ndarray:
    """Weighted average of s1 and s3 for all samples."""
    s1 = np.array([m["s1_score"] for m in matched])
    s3 = np.array([m["s3_score"] for m in matched])
    return alpha * s1 + (1 - alpha) * s3


def strategy_weighted_avg_escalated(matched: list, alpha: float,
                                    conf_low: float = 0.15, conf_high: float = 0.85) -> np.ndarray:
    """Keep s1 for confident samples, blend for escalated."""
    s1 = np.array([m["s1_score"] for m in matched])
    s3 = np.array([m["s3_score"] for m in matched])
    confident = (s1 <= conf_low) | (s1 >= conf_high)
    blended = alpha * s1 + (1 - alpha) * s3
    return np.where(confident, s1, blended)


def strategy_max_rule(matched: list) -> np.ndarray:
    """Max of s1 and s3."""
    s1 = np.array([m["s1_score"] for m in matched])
    s3 = np.array([m["s3_score"] for m in matched])
    return np.maximum(s1, s3)


def strategy_product_rule(matched: list) -> np.ndarray:
    """Product rule: P(at least one flags) = 1 - (1-s1)*(1-s3)."""
    s1 = np.array([m["s1_score"] for m in matched])
    s3 = np.array([m["s3_score"] for m in matched])
    return 1.0 - (1.0 - s1) * (1.0 - s3)


def strategy_routing_threshold(matched: list, conf_low: float, conf_high: float) -> np.ndarray:
    """Routing with variable thresholds: confident zone uses s1, uncertain uses s3."""
    s1 = np.array([m["s1_score"] for m in matched])
    s3 = np.array([m["s3_score"] for m in matched])
    confident = (s1 <= conf_low) | (s1 >= conf_high)
    return np.where(confident, s1, s3)


def run_blending_analysis(matched: list) -> dict:
    """Run all blending strategies and return results."""
    labels = np.array([m["ground_truth"] for m in matched])
    task_types = np.array([m["task_type"] for m in matched])
    s1 = np.array([m["s1_score"] for m in matched])
    s3 = np.array([m["s3_score"] for m in matched])

    results = {}

    # Baselines
    results["baseline_s1_only"] = {
        **compute_metrics(s1, labels),
        "per_task": compute_per_task_metrics(s1, labels, task_types),
    }
    results["baseline_s3_only"] = {
        **compute_metrics(s3, labels),
        "per_task": compute_per_task_metrics(s3, labels, task_types),
    }

    # Strategy 1: Weighted average (all samples)
    alpha_results = {}
    best_auroc_alpha, best_f1_alpha = 0.0, 0.0
    best_auroc, best_f1 = 0.0, 0.0
    best_auroc_metrics, best_f1_metrics = None, None
    for alpha in np.linspace(0.0, 1.0, 21):
        alpha = round(alpha, 2)
        scores = strategy_weighted_avg_all(matched, alpha)
        m = compute_metrics(scores, labels)
        alpha_results[str(alpha)] = m
        if m["auroc"] is not None and m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            best_auroc_alpha = alpha
            best_auroc_metrics = m
        if m["f1"] is not None and m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_f1_alpha = alpha
            best_f1_metrics = m

    results["weighted_avg_all"] = {
        "alpha_sweep": alpha_results,
        "best_auroc_alpha": best_auroc_alpha,
        "best_f1_alpha": best_f1_alpha,
        "best": best_auroc_metrics or {},
        "best_f1_config": best_f1_metrics or {},
        "per_task": compute_per_task_metrics(
            strategy_weighted_avg_all(matched, best_auroc_alpha), labels, task_types
        ),
    }

    # Strategy 2: Weighted average (escalated only)
    alpha_results_esc = {}
    best_auroc_esc, best_f1_esc = 0.0, 0.0
    best_auroc_alpha_esc, best_f1_alpha_esc = 0.0, 0.0
    best_auroc_metrics_esc, best_f1_metrics_esc = None, None
    for alpha in np.linspace(0.0, 1.0, 21):
        alpha = round(alpha, 2)
        scores = strategy_weighted_avg_escalated(matched, alpha)
        m = compute_metrics(scores, labels)
        alpha_results_esc[str(alpha)] = m
        if m["auroc"] is not None and m["auroc"] > best_auroc_esc:
            best_auroc_esc = m["auroc"]
            best_auroc_alpha_esc = alpha
            best_auroc_metrics_esc = m
        if m["f1"] is not None and m["f1"] > best_f1_esc:
            best_f1_esc = m["f1"]
            best_f1_alpha_esc = alpha
            best_f1_metrics_esc = m

    results["weighted_avg_escalated"] = {
        "alpha_sweep": alpha_results_esc,
        "best_auroc_alpha": best_auroc_alpha_esc,
        "best_f1_alpha": best_f1_alpha_esc,
        "best": best_auroc_metrics_esc or {},
        "best_f1_config": best_f1_metrics_esc or {},
        "per_task": compute_per_task_metrics(
            strategy_weighted_avg_escalated(matched, best_auroc_alpha_esc), labels, task_types
        ),
    }

    # Strategy 3: Max rule
    scores = strategy_max_rule(matched)
    results["max_rule"] = {
        **compute_metrics(scores, labels),
        "per_task": compute_per_task_metrics(scores, labels, task_types),
    }

    # Strategy 4: Product rule
    scores = strategy_product_rule(matched)
    results["product_rule"] = {
        **compute_metrics(scores, labels),
        "per_task": compute_per_task_metrics(scores, labels, task_types),
    }

    # Strategy 5: Routing threshold sweep
    threshold_results = {}
    best_auroc_thresh, best_f1_thresh = 0.0, 0.0
    best_auroc_pair, best_f1_pair = (0.15, 0.85), (0.15, 0.85)
    for low in np.arange(0.05, 0.45, 0.05):
        for high in np.arange(0.55, 0.96, 0.05):
            low_r, high_r = round(low, 2), round(high, 2)
            scores = strategy_routing_threshold(matched, low_r, high_r)
            m = compute_metrics(scores, labels)
            # Compute escalation % (s1 already computed above)
            esc_pct = round(100 * np.mean((s1 > low_r) & (s1 < high_r)), 1)
            key = f"{low_r}_{high_r}"
            threshold_results[key] = {**m, "escalation_pct": esc_pct}
            if m["auroc"] is not None and m["auroc"] > best_auroc_thresh:
                best_auroc_thresh = m["auroc"]
                best_auroc_pair = (low_r, high_r)
            if m["f1"] is not None and m["f1"] > best_f1_thresh:
                best_f1_thresh = m["f1"]
                best_f1_pair = (low_r, high_r)

    best_thresh_scores = strategy_routing_threshold(matched, *best_auroc_pair)
    results["routing_threshold_sweep"] = {
        "sweep": threshold_results,
        "best_auroc_thresholds": list(best_auroc_pair),
        "best_f1_thresholds": list(best_f1_pair),
        "best": threshold_results[f"{best_auroc_pair[0]}_{best_auroc_pair[1]}"],
        "per_task": compute_per_task_metrics(best_thresh_scores, labels, task_types),
    }

    return results


def print_summary(all_results: dict):
    """Print formatted summary table."""
    print("\n" + "=" * 100)
    print("SCORE BLENDING ANALYSIS SUMMARY")
    print("=" * 100)

    for config_key, data in all_results.items():
        if config_key == "metadata":
            continue
        meta = data.get("metadata", {})
        print(f"\n--- {meta.get('transformer', '?')} + {meta.get('stage3', '?')} ---")
        print(f"  Original cascade:  AUROC={meta.get('cascade_auroc', '?')}, F1={meta.get('cascade_f1', '?')}")
        print(f"  Stage 1 baseline:  AUROC={meta.get('s1_auroc', '?')}, F1={meta.get('s1_f1', '?')}")

        strategies = data.get("strategies", {})
        print(f"\n  {'Strategy':<28} {'AUROC':>8} {'F1@0.5':>8} {'OptF1':>8} {'Notes'}")
        print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*8} {'-'*30}")

        for name, s in strategies.items():
            if name.startswith("baseline"):
                auroc = s.get("auroc", "N/A")
                f1 = s.get("f1", "N/A")
                opt_f1 = s.get("optimal_f1", "N/A")
                print(f"  {name:<28} {auroc:>8} {f1:>8} {opt_f1:>8}")
            elif name in ("weighted_avg_all", "weighted_avg_escalated"):
                best = s.get("best", {})
                auroc = best.get("auroc", "N/A")
                f1 = best.get("f1", "N/A")
                opt_f1 = best.get("optimal_f1", "N/A")
                ba = s.get("best_auroc_alpha", "?")
                bf = s.get("best_f1_alpha", "?")
                print(f"  {name:<28} {auroc:>8} {f1:>8} {opt_f1:>8} alpha_auroc={ba}, alpha_f1={bf}")
            elif name == "routing_threshold_sweep":
                best = s.get("best", {})
                auroc = best.get("auroc", "N/A")
                f1 = best.get("f1", "N/A")
                opt_f1 = best.get("optimal_f1", "N/A")
                pair = s.get("best_auroc_thresholds", ["?", "?"])
                esc = best.get("escalation_pct", "?")
                print(f"  {name:<28} {auroc:>8} {f1:>8} {opt_f1:>8} low={pair[0]}, high={pair[1]}, esc={esc}%")
            else:
                auroc = s.get("auroc", "N/A")
                f1 = s.get("f1", "N/A")
                opt_f1 = s.get("optimal_f1", "N/A")
                print(f"  {name:<28} {auroc:>8} {f1:>8} {opt_f1:>8}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze score blending strategies")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing benchmark JSONs with predictions")
    parser.add_argument("--output", type=str, default="blending_analysis.json",
                        help="Output JSON path")
    parser.add_argument("--pattern", type=str, default="benchmark_*_20*.json",
                        help="Glob pattern for benchmark files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files = sorted(results_dir.glob(args.pattern))
    if not json_files:
        print(f"No files matching {args.pattern} in {results_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} benchmark files")

    all_results = {}
    skipped = 0

    for json_path in json_files:
        data = load_predictions(json_path)
        meta = data["metadata"]
        config_key = f"{meta['transformer']}_{meta['stage3']}"

        if data["stage1"] is None or data["stage3"] is None:
            print(f"  SKIP {json_path.name}: missing predictions (run with --save-predictions)")
            skipped += 1
            continue

        matched = match_predictions(data["stage1"], data["stage3"])
        if len(matched) < 100:
            print(f"  SKIP {json_path.name}: only {len(matched)} matched samples")
            skipped += 1
            continue

        print(f"\n  Analyzing {config_key} ({len(matched)} samples)...")
        strategies = run_blending_analysis(matched)

        all_results[config_key] = {
            "metadata": {
                "transformer": meta["transformer"],
                "stage3": meta["stage3"],
                "n_matched": len(matched),
                "s1_auroc": data["stage1_metrics"].get("auroc"),
                "s1_f1": data["stage1_metrics"].get("f1"),
                "cascade_auroc": data["cascade_metrics"].get("auroc"),
                "cascade_f1": data["cascade_metrics"].get("f1"),
            },
            "strategies": strategies,
        }

    if skipped:
        print(f"\nSkipped {skipped} files (missing predictions)")

    if not all_results:
        print("No valid results to analyze. Re-run benchmark with --save-predictions.")
        sys.exit(1)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
