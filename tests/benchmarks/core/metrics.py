"""Metrics computation for benchmark evaluation.

Ported from read-training/benchmark/evaluate_benchmark.py.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from tests.benchmarks.core.models import AccuracyMetrics, PredictionResult


def compute_auroc_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for AUROC.

    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities
        n_bootstraps: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    aurocs = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue

        aurocs.append(roc_auc_score(y_true_boot, y_scores_boot))

    if len(aurocs) < n_bootstraps // 2:
        # Not enough valid bootstrap samples
        return (0.0, 1.0)

    alpha = 1 - confidence_level
    lower = float(np.percentile(aurocs, 100 * alpha / 2))
    upper = float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    return (lower, upper)


def compute_accuracy_metrics(
    predictions: list[PredictionResult],
    compute_ci: bool = True,
    threshold: float = 0.5,
) -> AccuracyMetrics:
    """Compute accuracy metrics from predictions.

    Args:
        predictions: List of prediction results
        compute_ci: Whether to compute bootstrap CI (slower)
        threshold: Classification threshold for binary predictions

    Returns:
        AccuracyMetrics with all computed metrics
    """
    # Filter valid predictions
    valid = [p for p in predictions if p.ground_truth in (0, 1)]

    if len(valid) < 10:
        return AccuracyMetrics(
            auroc=None,
            auroc_ci_lower=None,
            auroc_ci_upper=None,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            optimal_threshold=None,
            optimal_f1=None,
            mcc=None,
            balanced_accuracy=None,
            specificity=None,
            brier_score=None,
            n_samples=len(valid),
            n_hallucinations=sum(1 for p in valid if p.ground_truth == 1),
            n_factual=sum(1 for p in valid if p.ground_truth == 0),
            error="Insufficient valid samples",
        )

    y_true = np.array([p.ground_truth for p in valid])
    y_scores = np.array([p.predicted_score for p in valid])

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return AccuracyMetrics(
            auroc=None,
            auroc_ci_lower=None,
            auroc_ci_upper=None,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            optimal_threshold=None,
            optimal_f1=None,
            mcc=None,
            balanced_accuracy=None,
            specificity=None,
            brier_score=None,
            n_samples=len(valid),
            n_hallucinations=n_pos,
            n_factual=n_neg,
            error="Only one class present",
        )

    # AUROC with confidence interval
    auroc = float(roc_auc_score(y_true, y_scores))
    if compute_ci:
        auroc_ci = compute_auroc_ci(y_true, y_scores)
    else:
        auroc_ci = (None, None)

    # PR curve for optimal threshold
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_scores
    )

    # Best F1 threshold
    f1_scores = (
        2
        * precision_curve[:-1]
        * recall_curve[:-1]
        / (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
    )
    best_f1_idx = int(np.argmax(f1_scores))
    best_f1_threshold = float(pr_thresholds[best_f1_idx])
    best_f1_value = float(f1_scores[best_f1_idx])

    # Predictions at default threshold
    y_pred_default = (y_scores >= threshold).astype(int)

    # Predictions at optimal F1 threshold
    y_pred_optimal = (y_scores >= best_f1_threshold).astype(int)

    # Confusion matrix at optimal threshold for specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Calibration metrics (clamp scores to [0, 1] for Brier score)
    y_scores_clamped = np.clip(y_scores, 0.0, 1.0)
    brier = float(brier_score_loss(y_true, y_scores_clamped))

    return AccuracyMetrics(
        auroc=auroc,
        auroc_ci_lower=auroc_ci[0],
        auroc_ci_upper=auroc_ci[1],
        accuracy=float(accuracy_score(y_true, y_pred_default)),
        precision=float(precision_score(y_true, y_pred_default, zero_division=0)),
        recall=float(recall_score(y_true, y_pred_default, zero_division=0)),
        f1=float(f1_score(y_true, y_pred_default, zero_division=0)),
        optimal_threshold=best_f1_threshold,
        optimal_f1=best_f1_value,
        mcc=float(matthews_corrcoef(y_true, y_pred_optimal)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred_optimal)),
        specificity=specificity,
        brier_score=brier,
        n_samples=len(valid),
        n_hallucinations=n_pos,
        n_factual=n_neg,
        error=None,
    )


def compute_curves(
    predictions: list[PredictionResult],
) -> dict:
    """Compute ROC and PR curves for plotting.

    Args:
        predictions: List of prediction results

    Returns:
        Dictionary with roc_curve and pr_curve data
    """
    valid = [p for p in predictions if p.ground_truth in (0, 1)]

    if len(valid) < 10:
        return {"roc_curve": None, "pr_curve": None}

    y_true = np.array([p.ground_truth for p in valid])
    y_scores = np.array([p.predicted_score for p in valid])

    if len(np.unique(y_true)) < 2:
        return {"roc_curve": None, "pr_curve": None}

    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)

    # PR curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_true, y_scores
    )

    return {
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": precision_curve.tolist(),
            "recall": recall_curve.tolist(),
        },
    }
