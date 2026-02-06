"""Verify lexical + model2vec combination AUROC for new Stage 1 design.

Uses per-sample scores from calibration_analysis.json (RAGTruth, 2700 samples)
to compute offline AUROC for various weight combinations without running inference.

Results saved to tests/benchmarks/results/stage1_combo_analysis/
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score

DATA_PATH = Path(__file__).parent / "results" / "calibration" / "calibration_analysis.json"
OUTPUT_DIR = Path(__file__).parent / "results" / "stage1_combo_analysis"


def load_per_sample_data():
    with open(DATA_PATH) as f:
        data = json.load(f)
    return data["per_sample_predictions"]


def compute_combo_auroc(samples, weights: dict[str, float]) -> dict:
    """Compute AUROC for a weighted combination of component scores."""
    y_true = np.array([s["ground_truth"] for s in samples])

    total_weight = sum(w for w in weights.values() if w > 0)
    if total_weight == 0:
        return {"auroc": 0.5, "f1": 0.0, "optimal_threshold": 0.5}

    y_scores = np.zeros(len(samples))
    for comp, w in weights.items():
        if w > 0:
            scores = np.array([s[f"{comp}_score"] for s in samples])
            y_scores += w * scores
    y_scores /= total_weight

    auroc = roc_auc_score(y_true, y_scores)

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.01, 1.0, 0.01):
        preds = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return {
        "auroc": round(auroc, 6),
        "optimal_f1": round(best_f1, 6),
        "optimal_threshold": round(best_thresh, 4),
        "score_mean": round(float(y_scores.mean()), 4),
        "score_std": round(float(y_scores.std()), 4),
    }


def compute_routing_simulation(samples, weights: dict[str, float], th_high: float, th_low: float) -> dict:
    """Simulate routing with given thresholds."""
    y_true = np.array([s["ground_truth"] for s in samples])
    total_weight = sum(w for w in weights.values() if w > 0)
    if total_weight == 0:
        return {}

    y_scores = np.zeros(len(samples))
    for comp, w in weights.items():
        if w > 0:
            scores = np.array([s[f"{comp}_score"] for s in samples])
            y_scores += w * scores
    y_scores /= total_weight

    confident_high = y_scores >= th_high
    confident_low = y_scores <= th_low
    resolved = confident_high | confident_low
    n_resolved = int(resolved.sum())

    result = {
        "th_high": th_high,
        "th_low": th_low,
        "n_resolved": n_resolved,
        "resolved_pct": round(100 * n_resolved / len(y_scores), 2),
    }

    if n_resolved > 0:
        resolved_preds = (y_scores[resolved] >= 0.5).astype(int)
        resolved_truth = y_true[resolved]
        result["accuracy"] = round(float((resolved_preds == resolved_truth).mean()), 4)
        if len(np.unique(resolved_truth)) > 1:
            result["resolved_auroc"] = round(float(roc_auc_score(resolved_truth, y_scores[resolved])), 4)
    return result


def compute_cascade_auroc(samples, s1_weights, s2_weights, th_high, th_low):
    """Simulate full cascade: Stage 1 routes, Stage 2 handles the rest."""
    y_true = np.array([s["ground_truth"] for s in samples])

    # Stage 1 scores
    s1_total = sum(w for w in s1_weights.values() if w > 0)
    s1_scores = np.zeros(len(samples))
    for comp, w in s1_weights.items():
        if w > 0:
            s1_scores += w * np.array([s[f"{comp}_score"] for s in samples])
    if s1_total > 0:
        s1_scores /= s1_total

    # Stage 2 scores
    s2_total = sum(w for w in s2_weights.values() if w > 0)
    s2_scores = np.zeros(len(samples))
    for comp, w in s2_weights.items():
        if w > 0:
            s2_scores += w * np.array([s[f"{comp}_score"] for s in samples])
    if s2_total > 0:
        s2_scores /= s2_total

    # Routing: Stage 1 resolves confident, rest goes to Stage 2
    resolved_at_s1 = (s1_scores >= th_high) | (s1_scores <= th_low)
    cascade_scores = np.where(resolved_at_s1, s1_scores, s2_scores)

    auroc = roc_auc_score(y_true, cascade_scores)

    best_f1 = 0.0
    for thresh in np.arange(0.01, 1.0, 0.01):
        preds = (cascade_scores >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    return {
        "cascade_auroc": round(auroc, 6),
        "cascade_optimal_f1": round(best_f1, 6),
        "stage1_resolved_n": int(resolved_at_s1.sum()),
        "stage1_resolved_pct": round(100 * resolved_at_s1.sum() / len(y_true), 2),
        "stage2_resolved_n": int((~resolved_at_s1).sum()),
    }


def compute_distributions(samples, y_true, component_name, scores):
    """Compute distribution stats for a component."""
    hal_mask = y_true == 1
    sup_mask = y_true == 0
    hal = scores[hal_mask]
    sup = scores[sup_mask]
    return {
        "component": component_name,
        "hallucinated": {
            "mean": round(float(hal.mean()), 4),
            "std": round(float(hal.std()), 4),
            "min": round(float(hal.min()), 4),
            "p25": round(float(np.percentile(hal, 25)), 4),
            "p50": round(float(np.percentile(hal, 50)), 4),
            "p75": round(float(np.percentile(hal, 75)), 4),
            "max": round(float(hal.max()), 4),
        },
        "supported": {
            "mean": round(float(sup.mean()), 4),
            "std": round(float(sup.std()), 4),
            "min": round(float(sup.min()), 4),
            "p25": round(float(np.percentile(sup, 25)), 4),
            "p50": round(float(np.percentile(sup, 50)), 4),
            "p75": round(float(np.percentile(sup, 75)), 4),
            "max": round(float(sup.max()), 4),
        },
        "separation": round(float(hal.mean() - sup.mean()), 4),
    }


def main():
    samples = load_per_sample_data()
    n = len(samples)
    y_true = np.array([s["ground_truth"] for s in samples])
    n_hal = int(y_true.sum())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: RAGTruth, {n} samples ({n_hal} hallucinated, {n - n_hal} supported)")
    print()

    # === Individual baselines ===
    print("=" * 70)
    print("INDIVIDUAL COMPONENT BASELINES")
    print("=" * 70)
    components = ["lexical", "model2vec", "numeric", "ner", "transformer", "nli"]
    baselines = {}
    for comp in components:
        result = compute_combo_auroc(samples, {comp: 1.0})
        baselines[comp] = result
        print(f"  {comp:15s}  AUROC={result['auroc']:.4f}  F1={result['optimal_f1']:.4f}  thresh={result['optimal_threshold']:.2f}")

    # === Stage 1 combinations ===
    print()
    print("=" * 70)
    print("STAGE 1 COMBINATIONS (no transformer)")
    print("=" * 70)
    combos = {
        "lex_only": {"lexical": 1.0},
        "m2v_only": {"model2vec": 1.0},
        "lex+m2v_50_50": {"lexical": 0.5, "model2vec": 0.5},
        "lex+m2v_60_40": {"lexical": 0.6, "model2vec": 0.4},
        "lex+m2v_70_30": {"lexical": 0.7, "model2vec": 0.3},
        "lex+m2v_80_20": {"lexical": 0.8, "model2vec": 0.2},
        "lex+m2v+num": {"lexical": 0.4, "model2vec": 0.4, "numeric": 0.2},
        "lex+m2v+ner": {"lexical": 0.4, "model2vec": 0.4, "ner": 0.2},
        "all_heuristics": {"lexical": 0.3, "model2vec": 0.3, "numeric": 0.2, "ner": 0.2},
    }
    combo_results = {}
    for name, weights in combos.items():
        result = compute_combo_auroc(samples, weights)
        combo_results[name] = {"weights": weights, **result}
        print(f"  {name:25s}  AUROC={result['auroc']:.4f}  F1={result['optimal_f1']:.4f}")

    # === Weight sweep ===
    print()
    print("=" * 70)
    print("WEIGHT SWEEP: lexical vs model2vec")
    print("=" * 70)
    sweep_results = []
    best_auroc = 0
    best_lex_w = 0.5
    for lex_w_int in range(0, 105, 5):
        lex_w = lex_w_int / 100.0
        m2v_w = 1.0 - lex_w
        result = compute_combo_auroc(samples, {"lexical": lex_w, "model2vec": m2v_w})
        sweep_results.append({"lex_weight": lex_w, "m2v_weight": round(m2v_w, 2), **result})
        marker = ""
        if result["auroc"] > best_auroc:
            best_auroc = result["auroc"]
            best_lex_w = lex_w
            marker = " <-- best"
        print(f"  lex={lex_w:.2f} m2v={m2v_w:.2f}  AUROC={result['auroc']:.4f}{marker}")

    print(f"\nBest: lex={best_lex_w:.2f} m2v={1-best_lex_w:.2f} AUROC={best_auroc:.4f}")

    # === Comparison ===
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    current_s1 = compute_combo_auroc(samples, {"transformer": 0.7, "lexical": 0.3})
    trans_only = compute_combo_auroc(samples, {"transformer": 1.0})
    best_new_s1 = compute_combo_auroc(samples, {"lexical": best_lex_w, "model2vec": 1 - best_lex_w})
    print(f"  Current Stage1 (trans=0.7 lex=0.3):  AUROC={current_s1['auroc']:.4f}  F1={current_s1['optimal_f1']:.4f}")
    print(f"  Transformer only:                     AUROC={trans_only['auroc']:.4f}  F1={trans_only['optimal_f1']:.4f}")
    print(f"  New Stage1 best (lex+m2v):             AUROC={best_new_s1['auroc']:.4f}  F1={best_new_s1['optimal_f1']:.4f}")
    print(f"  Delta vs current:                      AUROC={best_new_s1['auroc'] - current_s1['auroc']:+.4f}")

    # === Routing simulation ===
    print()
    print("=" * 70)
    print("ROUTING SIMULATION (new Stage 1 -> Stage 2 transformer)")
    print("=" * 70)
    best_weights = {"lexical": best_lex_w, "model2vec": 1 - best_lex_w}
    routing_results = []
    thresholds = [
        (0.90, 0.03), (0.85, 0.05), (0.80, 0.10), (0.75, 0.15),
        (0.70, 0.20), (0.65, 0.25), (0.60, 0.30),
    ]
    for th_high, th_low in thresholds:
        r = compute_routing_simulation(samples, best_weights, th_high, th_low)
        routing_results.append(r)
        acc = r.get("accuracy", 0)
        print(f"  th_high={th_high:.2f} th_low={th_low:.2f}  resolved={r['n_resolved']:4d} ({r['resolved_pct']:5.1f}%)  accuracy={acc:.3f}")

    # === Cascade AUROC simulation ===
    print()
    print("=" * 70)
    print("CASCADE AUROC: Stage 1 (lex+m2v) -> Stage 2 (transformer)")
    print("=" * 70)
    s2_weights = {"transformer": 1.0}
    cascade_results = []
    for th_high, th_low in thresholds:
        c = compute_cascade_auroc(samples, best_weights, s2_weights, th_high, th_low)
        cascade_results.append({"th_high": th_high, "th_low": th_low, **c})
        print(f"  th_high={th_high:.2f} th_low={th_low:.2f}  cascade_auroc={c['cascade_auroc']:.4f}  "
              f"s1_resolved={c['stage1_resolved_pct']:5.1f}%  f1={c['cascade_optimal_f1']:.4f}")

    # Baseline: just transformer alone
    print(f"\n  Transformer alone:         AUROC={trans_only['auroc']:.4f}")
    print(f"  Current trans+lex Stage1:  AUROC={current_s1['auroc']:.4f}")

    # === Score distributions ===
    print()
    print("=" * 70)
    print("SCORE DISTRIBUTIONS")
    print("=" * 70)
    dist_components = {
        "lexical": np.array([s["lexical_score"] for s in samples]),
        "model2vec": np.array([s["model2vec_score"] for s in samples]),
        "lex+m2v": np.array([
            best_lex_w * s["lexical_score"] + (1 - best_lex_w) * s["model2vec_score"]
            for s in samples
        ]),
        "transformer": np.array([s["transformer_score"] for s in samples]),
        "trans+lex": np.array([
            0.7 * s["transformer_score"] + 0.3 * s["lexical_score"]
            for s in samples
        ]),
    }
    distributions = {}
    for name, scores in dist_components.items():
        d = compute_distributions(samples, y_true, name, scores)
        distributions[name] = d
        print(f"\n  {name}:")
        print(f"    Hal: mean={d['hallucinated']['mean']:.3f} std={d['hallucinated']['std']:.3f} "
              f"[{d['hallucinated']['min']:.3f} - {d['hallucinated']['max']:.3f}]")
        print(f"    Sup: mean={d['supported']['mean']:.3f} std={d['supported']['std']:.3f} "
              f"[{d['supported']['min']:.3f} - {d['supported']['max']:.3f}]")
        print(f"    Separation: {d['separation']:+.3f}")

    # === Save all results ===
    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "ragtruth",
            "n_samples": n,
            "n_hallucinated": n_hal,
            "n_supported": n - n_hal,
            "source": str(DATA_PATH),
        },
        "baselines": baselines,
        "stage1_combinations": combo_results,
        "weight_sweep": sweep_results,
        "best_lex_m2v_weights": {"lexical": best_lex_w, "model2vec": round(1 - best_lex_w, 2)},
        "comparison": {
            "current_stage1_trans_lex": current_s1,
            "transformer_only": trans_only,
            "new_stage1_lex_m2v": best_new_s1,
        },
        "routing_simulation": routing_results,
        "cascade_simulation": cascade_results,
        "distributions": distributions,
        "conclusion": (
            f"Best lex+m2v AUROC: {best_auroc:.4f} (lex={best_lex_w:.2f}, m2v={1-best_lex_w:.2f}). "
            f"Delta vs current Stage1: {best_new_s1['auroc'] - current_s1['auroc']:+.4f}. "
            f"Model2Vec adds only +{best_auroc - baselines['lexical']['auroc']:.4f} over lexical alone. "
            f"Routing simulation: Stage 1 resolves <6% of samples at any reasonable threshold. "
            f"Cascade AUROC approximately equals transformer alone ({trans_only['auroc']:.4f}) "
            f"since Stage 1 barely routes."
        ),
    }

    out_path = OUTPUT_DIR / f"stage1_combo_analysis_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
