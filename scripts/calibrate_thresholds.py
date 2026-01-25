#!/usr/bin/env python3
"""Run threshold calibration on RAGTruth dataset.

This script calibrates the routing thresholds for Stage 1 and Stage 2 by finding
optimal values on the RAGTruth validation set.

Usage:
    python scripts/calibrate_thresholds.py --output thresholds.json

Output example:
    {
        "stage1_confident_high": 0.72,
        "stage1_confident_low": 0.35,
        "stage2_threshold": 0.58,
        "calibration_date": "2024-01-20",
        "dataset": "ragtruth",
        "samples": 1000
    }
"""

from __future__ import annotations

import argparse

# Disable torch.compile for older GPUs (GTX 1080 = CUDA 6.1, Triton requires >= 7.0)
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ragtruth_data(split: str = "test", limit: int | None = None) -> list[dict]:
    """Load RAGTruth dataset.

    Args:
        split: Dataset split ("train", "test")
        limit: Maximum number of samples to load

    Returns:
        List of dicts with context, answer, question, label
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets not installed. Run: pip install datasets")

    logger.info(f"Loading RAGTruth {split} split...")
    ds = load_dataset("wandb/RAGTruth-processed", split=split)

    samples = []
    for i, item in enumerate(ds):
        if limit and i >= limit:
            break

        # RAGTruth-processed format
        context_str = item.get("context", "")
        labels = item.get("hallucination_labels_processed", {})
        evident_conflict = labels.get("evident_conflict", 0) if labels else 0
        baseless_info = labels.get("baseless_info", 0) if labels else 0

        samples.append({
            "context": [context_str] if context_str else [],
            "answer": item.get("output", ""),
            "question": item.get("query", ""),
            "label": 1 if (evident_conflict or baseless_info) else 0,
        })

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def compute_scores(samples: list[dict], stage: int = 1) -> tuple[list[float], list[int]]:
    """Compute hallucination scores for samples.

    Args:
        samples: List of sample dicts
        stage: Which stage to evaluate (1 or 2)

    Returns:
        Tuple of (scores, labels)
    """
    from lettucedetect.cascade.types import CascadeInput
    from lettucedetect.configs import Stage1Config, Stage2Config
    from lettucedetect.detectors.stage1.detector import Stage1Detector
    from lettucedetect.detectors.stage2.detector import Stage2Detector

    if stage == 1:
        logger.info("Initializing Stage 1 detector...")
        config = Stage1Config(augmentations=["ner", "numeric", "lexical"])
        detector = Stage1Detector(config)
    else:
        logger.info("Initializing Stage 2 detector...")
        config = Stage2Config()
        detector = Stage2Detector(config)

    detector.warmup()

    scores = []
    labels = []

    logger.info(f"Computing scores for {len(samples)} samples...")
    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{len(samples)}")

        try:
            context = sample["context"]
            if isinstance(context, str):
                context = [context]

            cascade_input = CascadeInput(
                context=context,
                answer=sample["answer"],
                question=sample["question"],
            )
            result = detector.predict_stage(cascade_input, has_next_stage=True)
            scores.append(result.hallucination_score)
            labels.append(sample["label"])

        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue

    return scores, labels


def find_optimal_thresholds(
    scores: list[float],
    labels: list[int],
) -> dict:
    """Find optimal thresholds for routing.

    Args:
        scores: Hallucination scores
        labels: Ground truth labels (1 = hallucinated, 0 = supported)

    Returns:
        Dict with optimal thresholds
    """
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # Find threshold that maximizes F1
    precision, recall, thresholds = precision_recall_curve(labels_arr, scores_arr)

    # F1 at each threshold
    f1_scores = []
    for i, thresh in enumerate(thresholds):
        preds = (scores_arr >= thresh).astype(int)
        f1 = f1_score(labels_arr, preds, zero_division=0)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    logger.info(f"Best threshold: {best_threshold:.3f} (F1={best_f1:.3f})")

    # Find high confidence threshold: scores above this are likely hallucinated
    # Use 75th percentile of hallucinated samples (most hallucinated score above this)
    hal_scores = scores_arr[labels_arr == 1]
    confident_high = np.percentile(hal_scores, 75) if len(hal_scores) > 0 else 0.7

    # Find low confidence threshold: scores below this are likely supported
    # Use 25th percentile of supported samples (most supported score below this)
    sup_scores = scores_arr[labels_arr == 0]
    confident_low = np.percentile(sup_scores, 25) if len(sup_scores) > 0 else 0.3

    return {
        "threshold_optimal": float(best_threshold),
        "threshold_high": float(confident_high),
        "threshold_low": float(confident_low),
        "f1_at_optimal": float(best_f1),
    }


def main():
    """Run calibration."""
    parser = argparse.ArgumentParser(description="Calibrate routing thresholds")
    parser.add_argument("--output", type=str, default="thresholds.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, default=500, help="Max samples to evaluate")
    parser.add_argument("--stage", type=int, default=1, help="Stage to calibrate (1 or 2)")
    args = parser.parse_args()

    # Load data
    samples = load_ragtruth_data(split="test", limit=args.limit)

    # Compute scores
    scores, labels = compute_scores(samples, stage=args.stage)

    # Find optimal thresholds
    thresholds = find_optimal_thresholds(scores, labels)

    # Add metadata
    result = {
        f"stage{args.stage}_confident_high": thresholds["threshold_high"],
        f"stage{args.stage}_confident_low": thresholds["threshold_low"],
        f"stage{args.stage}_optimal": thresholds["threshold_optimal"],
        f"stage{args.stage}_f1": thresholds["f1_at_optimal"],
        "calibration_date": datetime.now().isoformat(),
        "dataset": "ragtruth",
        "samples": len(scores),
    }

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Results: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
