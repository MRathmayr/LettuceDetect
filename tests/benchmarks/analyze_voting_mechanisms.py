#!/usr/bin/env python3
"""Analyze voting mechanisms to diagnose cascade performance regression.

Calibrated binary voting decreased cascade AUROC from 0.789 to 0.758.
This script diagnoses WHY and finds better combination mechanisms.

Key analyses:
1. Ablation study - What if we remove each augmentation?
2. Flip analysis - When components flip decisions, are they right or wrong?
3. Voting mechanisms - Compare different combination strategies
4. Disagreement quadrants - When transformer and augmentations disagree, who wins?

Prerequisites:
    Run analyze_calibration.py first to generate per-sample predictions.

Usage:
    python tests/benchmarks/analyze_voting_mechanisms.py
    python tests/benchmarks/analyze_voting_mechanisms.py --input results/calibration/calibration_analysis.json
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.metrics import f1_score, roc_auc_score


@dataclass
class SampleData:
    """Per-sample prediction data loaded from calibration JSON."""

    sample_id: str
    ground_truth: int
    transformer_score: float
    lexical_score: float
    numeric_score: float
    ner_score: float
    model2vec_score: float
    nli_score: float


def load_predictions(input_path: Path) -> list[SampleData]:
    """Load per-sample predictions from calibration_analysis.json."""
    if not input_path.exists():
        print(f"ERROR: Calibration data not found at {input_path}")
        print("Run analyze_calibration.py first:")
        print("  python tests/benchmarks/analyze_calibration.py --output results/calibration/")
        sys.exit(1)

    try:
        with open(input_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON at {input_path}")
        print(f"  {e}")
        print("\nThe file may be truncated. Re-run analyze_calibration.py:")
        print("  python tests/benchmarks/analyze_calibration.py --output results/calibration/")
        sys.exit(1)

    if "per_sample_predictions" not in data:
        print("ERROR: calibration_analysis.json missing 'per_sample_predictions'")
        sys.exit(1)

    samples = []
    for p in data["per_sample_predictions"]:
        # Skip samples with missing scores
        required = ["transformer_score", "lexical_score", "numeric_score", "ner_score"]
        if any(p.get(k) is None for k in required):
            continue

        samples.append(
            SampleData(
                sample_id=p["sample_id"],
                ground_truth=p["ground_truth"],
                transformer_score=p["transformer_score"],
                lexical_score=p["lexical_score"],
                numeric_score=p["numeric_score"],
                ner_score=p["ner_score"],
                model2vec_score=p.get("model2vec_score", 0.5),
                nli_score=p.get("nli_score", 0.5),
            )
        )

    print(f"Loaded {len(samples)} samples from {input_path}")
    return samples


def get_optimal_threshold(y_true: list[int], y_pred: list[float]) -> tuple[float, float]:
    """Find threshold that maximizes F1 score."""
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.05, 0.95, 0.05):
        preds = (np.array(y_pred) >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def compute_weighted_avg(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted average of scores."""
    total_weight = sum(weights.values())
    weighted_sum = sum(scores.get(k, 0.5) * w for k, w in weights.items())
    return weighted_sum / total_weight


def bootstrap_ci(y_true: list[int], y_pred: list[float], n_bootstrap: int = 1000) -> tuple[float, float]:
    """Compute 95% confidence interval for AUROC using bootstrap."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    aurocs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_t = [y_true[i] for i in indices]
        y_p = [y_pred[i] for i in indices]
        # Need both classes
        if len(set(y_t)) < 2:
            continue
        aurocs.append(roc_auc_score(y_t, y_p))

    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)


def run_ablation_study(samples: list[SampleData]) -> dict:
    """Ablation study: What if we remove each augmentation?"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY")
    print("=" * 80)

    y_true = [s.ground_truth for s in samples]

    # Stage 1 weights (from cascade config)
    full_weights = {"transformer": 0.5, "ner": 0.2, "numeric": 0.15, "lexical": 0.15}

    results = {}

    # Full cascade (baseline)
    full_scores = []
    for s in samples:
        scores = {
            "transformer": s.transformer_score,
            "ner": s.ner_score,
            "numeric": s.numeric_score,
            "lexical": s.lexical_score,
        }
        full_scores.append(compute_weighted_avg(scores, full_weights))

    full_auroc = roc_auc_score(y_true, full_scores)
    _, full_f1 = get_optimal_threshold(y_true, full_scores)
    results["full_cascade"] = {"auroc": full_auroc, "f1": full_f1}

    # Transformer only
    trans_scores = [s.transformer_score for s in samples]
    trans_auroc = roc_auc_score(y_true, trans_scores)
    _, trans_f1 = get_optimal_threshold(y_true, trans_scores)
    results["transformer_only"] = {"auroc": trans_auroc, "f1": trans_f1}

    # Remove each augmentation one at a time
    augmentations = ["ner", "numeric", "lexical"]
    for remove_aug in augmentations:
        ablated_weights = {k: v for k, v in full_weights.items() if k != remove_aug}
        # Renormalize weights
        total = sum(ablated_weights.values())
        ablated_weights = {k: v / total for k, v in ablated_weights.items()}

        ablated_scores = []
        for s in samples:
            scores = {
                "transformer": s.transformer_score,
                "ner": s.ner_score,
                "numeric": s.numeric_score,
                "lexical": s.lexical_score,
            }
            ablated_scores.append(compute_weighted_avg(scores, ablated_weights))

        auroc = roc_auc_score(y_true, ablated_scores)
        _, f1 = get_optimal_threshold(y_true, ablated_scores)
        results[f"without_{remove_aug}"] = {"auroc": auroc, "f1": f1}

    # Print results
    print(f"\n{'Configuration':<25} {'AUROC':>8} {'F1':>8} {'Delta':>10}")
    print("-" * 55)

    baseline_auroc = results["full_cascade"]["auroc"]
    for name, stats in results.items():
        delta = stats["auroc"] - baseline_auroc
        delta_str = f"{delta:+.3f}" if name != "full_cascade" else "-"
        label = name.replace("_", " ").title()
        print(f"{label:<25} {stats['auroc']:>8.3f} {stats['f1']:>8.3f} {delta_str:>10}")

    # Highlight key finding
    if results["transformer_only"]["auroc"] > baseline_auroc:
        improvement = results["transformer_only"]["auroc"] - baseline_auroc
        print(f"\n>>> Transformer alone ({results['transformer_only']['auroc']:.3f}) BEATS full cascade ({baseline_auroc:.3f}) by +{improvement:.3f}")

    return results


def run_flip_analysis(samples: list[SampleData]) -> dict:
    """Analyze when components flip the cascade decision and whether flips help."""
    print("\n" + "=" * 80)
    print("FLIP ANALYSIS (Root Cause)")
    print("=" * 80)

    y_true = np.array([s.ground_truth for s in samples])

    # Get component scores and optimal thresholds
    components = {
        "transformer": [s.transformer_score for s in samples],
        "ner": [s.ner_score for s in samples],
        "numeric": [s.numeric_score for s in samples],
        "lexical": [s.lexical_score for s in samples],
    }

    thresholds = {}
    for name, scores in components.items():
        thresh, _ = get_optimal_threshold(y_true.tolist(), scores)
        thresholds[name] = thresh

    # Binary predictions at optimal threshold
    trans_preds = (np.array(components["transformer"]) >= thresholds["transformer"]).astype(int)

    results = {}

    print(f"\n{'Component':<12} {'Flips':<15} {'Correct When Flip':<20} {'Verdict'}")
    print("-" * 65)

    for aug_name in ["ner", "numeric", "lexical"]:
        aug_scores = np.array(components[aug_name])
        aug_preds = (aug_scores >= thresholds[aug_name]).astype(int)

        # Find where augmentation flips transformer's decision
        flips = trans_preds != aug_preds
        n_flips = flips.sum()
        flip_rate = n_flips / len(samples)

        if n_flips > 0:
            # When it flips, is the augmentation correct?
            aug_correct_on_flips = (aug_preds[flips] == y_true[flips]).sum()
            correct_when_flip = aug_correct_on_flips / n_flips
        else:
            correct_when_flip = 0.5

        verdict = "HURTS" if correct_when_flip < 0.5 else "HELPS" if correct_when_flip > 0.55 else "neutral"

        results[aug_name] = {
            "flip_rate": float(flip_rate),
            "n_flips": int(n_flips),
            "correct_when_flip": float(correct_when_flip),
            "verdict": verdict,
        }

        pct_str = f"{100*flip_rate:.0f}%"
        correct_str = f"{100*correct_when_flip:.0f}%"
        verdict_suffix = f" (< 50%)" if correct_when_flip < 0.5 else ""
        print(f"{aug_name:<12} {pct_str:<15} {correct_str:<20} {verdict}{verdict_suffix}")

    return results


def run_voting_mechanisms(samples: list[SampleData]) -> dict:
    """Compare different voting/combination mechanisms."""
    print("\n" + "=" * 80)
    print("VOTING MECHANISMS COMPARISON")
    print("=" * 80)

    y_true = [s.ground_truth for s in samples]

    # Get optimal thresholds for each component
    components = {
        "transformer": [s.transformer_score for s in samples],
        "ner": [s.ner_score for s in samples],
        "numeric": [s.numeric_score for s in samples],
        "lexical": [s.lexical_score for s in samples],
    }

    thresholds = {}
    for name, scores in components.items():
        thresh, _ = get_optimal_threshold(y_true, scores)
        thresholds[name] = thresh

    results = {}

    # Stage 1 weights
    s1_weights = {"transformer": 0.5, "ner": 0.2, "numeric": 0.15, "lexical": 0.15}

    # 1. Transformer only (no augmentations)
    trans_scores = components["transformer"]
    results["transformer_only"] = _evaluate_mechanism(y_true, trans_scores, "Transformer only")

    # 2. Raw weighted average (current baseline)
    raw_scores = []
    for s in samples:
        scores = {
            "transformer": s.transformer_score,
            "ner": s.ner_score,
            "numeric": s.numeric_score,
            "lexical": s.lexical_score,
        }
        raw_scores.append(compute_weighted_avg(scores, s1_weights))
    results["raw_weighted_avg"] = _evaluate_mechanism(y_true, raw_scores, "Raw weighted avg")

    # 3. Calibrated binary voting (what broke things)
    binary_scores = []
    for s in samples:
        votes = 0
        total_weight = 0
        for name, weight in s1_weights.items():
            score = getattr(s, f"{name}_score")
            if score >= thresholds[name]:
                votes += weight
            total_weight += weight
        binary_scores.append(votes / total_weight)
    results["calibrated_binary"] = _evaluate_mechanism(y_true, binary_scores, "Calibrated binary")

    # 4. Max voting (any component triggers hallucination)
    max_scores = []
    for s in samples:
        scores = [s.transformer_score, s.ner_score, s.numeric_score, s.lexical_score]
        max_scores.append(max(scores))
    results["max_voting"] = _evaluate_mechanism(y_true, max_scores, "Max voting")

    # 5. Veto voting (augmentations can only increase hallucination score)
    veto_scores = []
    for s in samples:
        base = s.transformer_score
        # Augmentations can only push score UP (toward hallucination)
        for aug_name in ["ner", "numeric", "lexical"]:
            aug_score = getattr(s, f"{aug_name}_score")
            if aug_score >= thresholds[aug_name]:
                base = max(base, 0.6)  # Boost if augmentation flags
        veto_scores.append(min(base, 1.0))
    results["veto_voting"] = _evaluate_mechanism(y_true, veto_scores, "Veto voting")

    # 6. Transformer + lexical only (in case lexical is the only helpful augmentation)
    trans_lex_weights = {"transformer": 0.7, "lexical": 0.3}
    trans_lex_scores = []
    for s in samples:
        scores = {"transformer": s.transformer_score, "lexical": s.lexical_score}
        trans_lex_scores.append(compute_weighted_avg(scores, trans_lex_weights))
    results["transformer_lexical"] = _evaluate_mechanism(y_true, trans_lex_scores, "Transformer+lexical")

    # Print sorted by AUROC
    print(f"\n{'Mechanism':<22} {'AUROC':>8} {'F1':>8} {'95% CI':<18}")
    print("-" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["auroc"], reverse=True)
    for name, stats in sorted_results:
        ci_str = f"[{stats['ci_low']:.2f}-{stats['ci_high']:.2f}]"
        print(f"{name:<22} {stats['auroc']:>8.3f} {stats['f1']:>8.3f} {ci_str:<18}")

    return results


def _evaluate_mechanism(y_true: list[int], y_pred: list[float], name: str) -> dict:
    """Evaluate a voting mechanism and return stats."""
    auroc = roc_auc_score(y_true, y_pred)
    thresh, f1 = get_optimal_threshold(y_true, y_pred)
    ci_low, ci_high = bootstrap_ci(y_true, y_pred)
    return {
        "auroc": float(auroc),
        "f1": float(f1),
        "threshold": float(thresh),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def run_disagreement_analysis(samples: list[SampleData]) -> dict:
    """Analyze disagreement quadrants between transformer and augmentations."""
    print("\n" + "=" * 80)
    print("DISAGREEMENT QUADRANT ANALYSIS")
    print("=" * 80)

    y_true = np.array([s.ground_truth for s in samples])

    # Get scores and thresholds
    trans_scores = np.array([s.transformer_score for s in samples])
    aug_scores = np.array(
        [
            (s.ner_score + s.numeric_score + s.lexical_score) / 3
            for s in samples
        ]
    )

    trans_thresh, _ = get_optimal_threshold(y_true.tolist(), trans_scores.tolist())
    aug_thresh, _ = get_optimal_threshold(y_true.tolist(), aug_scores.tolist())

    trans_hal = trans_scores >= trans_thresh  # Transformer predicts hallucination
    aug_hal = aug_scores >= aug_thresh  # Augmentations predict hallucination
    gt_hal = y_true == 1  # Ground truth is hallucination

    results = {}

    # Quadrant 1: Both agree hallucination
    q1_mask = trans_hal & aug_hal
    q1_n = q1_mask.sum()
    q1_correct = (gt_hal[q1_mask]).sum() if q1_n > 0 else 0
    results["both_hal"] = {
        "n": int(q1_n),
        "gt_hal": int(q1_correct),
        "gt_safe": int(q1_n - q1_correct),
    }

    # Quadrant 2: Both agree safe
    q2_mask = ~trans_hal & ~aug_hal
    q2_n = q2_mask.sum()
    q2_correct = (~gt_hal[q2_mask]).sum() if q2_n > 0 else 0
    results["both_safe"] = {
        "n": int(q2_n),
        "gt_safe": int(q2_correct),
        "gt_hal": int(q2_n - q2_correct),
    }

    # Quadrant 3: Transformer=hal, Aug=safe (disagreement)
    q3_mask = trans_hal & ~aug_hal
    q3_n = q3_mask.sum()
    if q3_n > 0:
        q3_gt_hal = (gt_hal[q3_mask]).sum()
        q3_gt_safe = q3_n - q3_gt_hal
    else:
        q3_gt_hal = q3_gt_safe = 0
    results["trans_hal_aug_safe"] = {
        "n": int(q3_n),
        "gt_hal": int(q3_gt_hal),
        "gt_safe": int(q3_gt_safe),
        "trans_wins": "transformer" if q3_gt_hal > q3_gt_safe else "augmentations",
    }

    # Quadrant 4: Transformer=safe, Aug=hal (disagreement)
    q4_mask = ~trans_hal & aug_hal
    q4_n = q4_mask.sum()
    if q4_n > 0:
        q4_gt_hal = (gt_hal[q4_mask]).sum()
        q4_gt_safe = q4_n - q4_gt_hal
    else:
        q4_gt_hal = q4_gt_safe = 0
    results["trans_safe_aug_hal"] = {
        "n": int(q4_n),
        "gt_hal": int(q4_gt_hal),
        "gt_safe": int(q4_gt_safe),
        "trans_wins": "transformer" if q4_gt_safe > q4_gt_hal else "augmentations",
    }

    # Print analysis
    print("\nWhen transformer and augmentations AGREE:")
    print(f"  Both predict hallucination: {results['both_hal']['n']:>4} samples")
    print(f"    -> GT was hallucination:  {results['both_hal']['gt_hal']:>4} (correct)")
    print(f"    -> GT was safe:           {results['both_hal']['gt_safe']:>4} (incorrect)")
    print(f"  Both predict safe:          {results['both_safe']['n']:>4} samples")
    print(f"    -> GT was safe:           {results['both_safe']['gt_safe']:>4} (correct)")
    print(f"    -> GT was hallucination:  {results['both_safe']['gt_hal']:>4} (incorrect)")

    print("\nWhen transformer and augmentations DISAGREE:")
    q3 = results["trans_hal_aug_safe"]
    print(f"  Transformer=hal, Aug=safe:  {q3['n']:>4} samples")
    if q3["n"] > 0:
        print(f"    -> GT was hallucination:  {q3['gt_hal']:>4} (transformer RIGHT)")
        print(f"    -> GT was safe:           {q3['gt_safe']:>4} (augmentations RIGHT)")
        winner = "TRANSFORMER" if q3["gt_hal"] > q3["gt_safe"] else "AUGMENTATIONS"
        print(f"    -> Winner: {winner}")

    q4 = results["trans_safe_aug_hal"]
    print(f"  Transformer=safe, Aug=hal:  {q4['n']:>4} samples")
    if q4["n"] > 0:
        print(f"    -> GT was hallucination:  {q4['gt_hal']:>4} (augmentations RIGHT)")
        print(f"    -> GT was safe:           {q4['gt_safe']:>4} (transformer RIGHT)")
        winner = "AUGMENTATIONS" if q4["gt_hal"] > q4["gt_safe"] else "TRANSFORMER"
        print(f"    -> Winner: {winner}")

    # Summary
    total_disagree = q3["n"] + q4["n"]
    if total_disagree > 0:
        trans_wins = q3["gt_hal"] + q4["gt_safe"]
        aug_wins = q3["gt_safe"] + q4["gt_hal"]
        print(f"\nSUMMARY: On {total_disagree} disagreements:")
        print(f"  Transformer correct: {trans_wins} ({100*trans_wins/total_disagree:.0f}%)")
        print(f"  Augmentations correct: {aug_wins} ({100*aug_wins/total_disagree:.0f}%)")

        if trans_wins > aug_wins:
            print("\n>>> Transformer wins most disagreements - augmentations hurt performance")
        else:
            print("\n>>> Augmentations win most disagreements - they add value")

    return results


def generate_conclusion(ablation: dict, flip: dict, voting: dict, disagreement: dict) -> str:
    """Generate actionable conclusion from all analyses."""
    conclusions = []

    # Check if transformer alone beats cascade
    if ablation["transformer_only"]["auroc"] > ablation["full_cascade"]["auroc"]:
        diff = ablation["transformer_only"]["auroc"] - ablation["full_cascade"]["auroc"]
        conclusions.append(
            f"Transformer alone ({ablation['transformer_only']['auroc']:.3f}) beats "
            f"full cascade ({ablation['full_cascade']['auroc']:.3f}) by +{diff:.3f}."
        )

    # Check flip analysis
    hurting = [name for name, data in flip.items() if data["verdict"] == "HURTS"]
    if hurting:
        conclusions.append(f"Components {hurting} flip decisions and are wrong >50% of the time.")

    # Best voting mechanism
    best_mech = max(voting.items(), key=lambda x: x[1]["auroc"])
    conclusions.append(f"Best mechanism: {best_mech[0]} with AUROC {best_mech[1]['auroc']:.3f}.")

    # Recommendation
    if "transformer_only" in [best_mech[0]]:
        conclusions.append("RECOMMENDATION: Use transformer only, disable augmentations.")
    elif best_mech[0] == "veto_voting":
        conclusions.append("RECOMMENDATION: Use veto voting - augmentations can only flag hallucinations.")

    return " ".join(conclusions)


def save_results(output_path: Path, **results):
    """Save all results to JSON."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_samples": results.get("n_samples", 0),
            "source": str(results.get("input_path", "")),
        },
        "ablation": results.get("ablation", {}),
        "flip_analysis": results.get("flip", {}),
        "voting_results": results.get("voting", {}),
        "disagreement_quadrants": results.get("disagreement", {}),
        "conclusion": results.get("conclusion", ""),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze voting mechanisms for cascade")
    parser.add_argument(
        "--input",
        type=str,
        default="tests/benchmarks/results/calibration/calibration_analysis.json",
        help="Path to calibration_analysis.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: same dir as input)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "voting_analysis.json"

    print("=" * 80)
    print("VOTING MECHANISMS ANALYSIS")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Load data
    samples = load_predictions(input_path)
    if len(samples) < 50:
        print(f"WARNING: Only {len(samples)} samples - results may be unreliable")

    # Run analyses
    ablation = run_ablation_study(samples)
    flip = run_flip_analysis(samples)
    voting = run_voting_mechanisms(samples)
    disagreement = run_disagreement_analysis(samples)

    # Generate conclusion
    conclusion = generate_conclusion(ablation, flip, voting, disagreement)
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"\n{conclusion}")

    # Save results
    save_results(
        output_path,
        n_samples=len(samples),
        input_path=input_path,
        ablation=ablation,
        flip=flip,
        voting=voting,
        disagreement=disagreement,
        conclusion=conclusion,
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
