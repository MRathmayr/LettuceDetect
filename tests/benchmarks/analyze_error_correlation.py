#!/usr/bin/env python3
"""Analyze error correlation between components.

Key question: Do components make errors on the SAME samples or DIFFERENT samples?
- High correlation: combining can't help (same errors)
- Low correlation: combining might help (complementary errors)

Also analyzes: When transformer is wrong, are other components right?

Usage:
    python tests/benchmarks/analyze_error_correlation.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.metrics import roc_auc_score
from tests.benchmarks.data_adapters import RAGTruthAdapter


def get_optimal_threshold(y_true, y_pred):
    """Find threshold that maximizes accuracy."""
    best_thresh, best_acc = 0.5, 0
    for thresh in np.arange(0.1, 0.9, 0.05):
        preds = (np.array(y_pred) >= thresh).astype(int)
        acc = (preds == np.array(y_true)).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    return best_thresh, best_acc


def clear_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    print("=" * 70)
    print("ERROR CORRELATION ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\nLoading RAGTruth dataset...")
    adapter = RAGTruthAdapter()
    samples = adapter.load()
    print(f"Loaded {len(samples)} samples")

    y_true = [s.ground_truth for s in samples]

    # Collect predictions from each component
    component_scores = {}

    # === Transformer ===
    print("\n[1/6] Running Transformer...")
    from lettucedetect.detectors.transformer import TransformerDetector
    transformer = TransformerDetector(model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1")
    transformer.warmup()

    scores = []
    for i, s in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(samples)}")
        spans = transformer.predict(s.context, s.response, s.question, output_format="spans")
        score = max((sp.get("confidence", 0.5) for sp in spans), default=0.0)
        scores.append(score)
    component_scores["transformer"] = scores
    del transformer
    clear_gpu()

    # === Lexical ===
    print("\n[2/6] Running Lexical...")
    from lettucedetect.utils.lexical import LexicalOverlapCalculator
    lexical = LexicalOverlapCalculator()
    scores = []
    for s in samples:
        result = lexical.score(s.context, s.response, s.question, None)
        scores.append(result.score if result.score is not None else 0.5)
    component_scores["lexical"] = scores

    # === NER ===
    print("\n[3/6] Running NER...")
    from lettucedetect.detectors.stage1.augmentations.ner_verifier import NERVerifier
    ner = NERVerifier()
    ner.preload()
    scores = []
    for s in samples:
        result = ner.score(s.context, s.response, s.question, None)
        scores.append(result.score if result.score is not None else 0.5)
    component_scores["ner"] = scores

    # === Numeric ===
    print("\n[4/6] Running Numeric...")
    from lettucedetect.detectors.stage1.augmentations.numeric_validator import NumericValidator
    numeric = NumericValidator()
    scores = []
    for s in samples:
        result = numeric.score(s.context, s.response, s.question, None)
        scores.append(result.score if result.score is not None else 0.5)
    component_scores["numeric"] = scores

    # === Model2Vec ===
    print("\n[5/6] Running Model2Vec...")
    from lettucedetect.detectors.stage2.model2vec_encoder import Model2VecEncoder
    model2vec = Model2VecEncoder()
    scores = []
    for s in samples:
        ncs = model2vec.compute_ncs(s.context, s.response)
        # Convert NCS to hallucination score
        scores.append(1.0 - ncs["max"])
    component_scores["model2vec"] = scores
    del model2vec
    clear_gpu()

    # === NLI ===
    print("\n[6/6] Running NLI...")
    from lettucedetect.detectors.stage2.nli_detector import NLIContradictionDetector
    nli = NLIContradictionDetector()
    nli.preload()
    scores = []
    for i, s in enumerate(samples):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(samples)}")
        result = nli.compute_context_nli(s.context, s.response)
        scores.append(result["hallucination_score"])
    component_scores["nli"] = scores
    del nli
    clear_gpu()

    # === Analysis ===
    print("\n" + "=" * 70)
    print("COMPONENT PERFORMANCE")
    print("=" * 70)

    component_errors = {}  # Binary: 1 = error, 0 = correct
    component_thresholds = {}

    for name, scores in component_scores.items():
        auroc = roc_auc_score(y_true, scores)
        thresh, acc = get_optimal_threshold(y_true, scores)
        component_thresholds[name] = thresh

        # Compute binary errors at optimal threshold
        preds = (np.array(scores) >= thresh).astype(int)
        errors = (preds != np.array(y_true)).astype(int)
        component_errors[name] = errors

        print(f"{name:12s}: AUROC={auroc:.3f}, Acc={acc:.3f} @ thresh={thresh:.2f}, Errors={errors.sum()}/{len(errors)}")

    # === Error Correlation Matrix ===
    print("\n" + "=" * 70)
    print("ERROR CORRELATION MATRIX")
    print("(High = same errors, Low = different errors)")
    print("=" * 70)

    names = list(component_errors.keys())
    n = len(names)

    # Header
    print(f"{'':12s}", end="")
    for name in names:
        print(f"{name:>10s}", end="")
    print()

    # Correlation matrix
    for i, name_i in enumerate(names):
        print(f"{name_i:12s}", end="")
        for j, name_j in enumerate(names):
            corr = np.corrcoef(component_errors[name_i], component_errors[name_j])[0, 1]
            print(f"{corr:10.3f}", end="")
        print()

    # === Key Analysis: When Transformer is Wrong ===
    print("\n" + "=" * 70)
    print("WHEN TRANSFORMER IS WRONG, ARE OTHERS RIGHT?")
    print("=" * 70)

    transformer_errors = component_errors["transformer"]
    transformer_wrong_indices = np.where(transformer_errors == 1)[0]
    n_transformer_wrong = len(transformer_wrong_indices)

    print(f"\nTransformer wrong on {n_transformer_wrong} samples ({100*n_transformer_wrong/len(samples):.1f}%)")
    print("\nOn those samples, other components:")

    for name, errors in component_errors.items():
        if name == "transformer":
            continue
        # How many times is this component RIGHT when transformer is WRONG?
        other_correct_when_trans_wrong = (errors[transformer_wrong_indices] == 0).sum()
        pct = 100 * other_correct_when_trans_wrong / n_transformer_wrong
        print(f"  {name:12s}: correct {other_correct_when_trans_wrong}/{n_transformer_wrong} ({pct:.1f}%)")

    # === Complementary Analysis ===
    print("\n" + "=" * 70)
    print("COMPLEMENTARY POTENTIAL (Oracle Analysis)")
    print("=" * 70)

    # If we could perfectly choose the right component per sample
    all_errors = np.stack([component_errors[n] for n in names], axis=1)
    oracle_errors = all_errors.min(axis=1)  # 0 if ANY component is right
    oracle_acc = 1 - oracle_errors.mean()

    trans_acc = 1 - component_errors["transformer"].mean()

    print(f"\nTransformer accuracy:     {trans_acc:.3f}")
    print(f"Oracle (best of all):     {oracle_acc:.3f}")
    print(f"Potential improvement:    +{oracle_acc - trans_acc:.3f}")

    if oracle_acc > trans_acc + 0.01:
        print("\n[GOOD] Components have complementary signals - smart combination could help!")
    else:
        print("\n[BAD] Components mostly make same errors - combining won't help much")

    # === Agreement Analysis ===
    print("\n" + "=" * 70)
    print("AGREEMENT ANALYSIS")
    print("=" * 70)

    # When do all components agree vs disagree?
    trans_preds = (np.array(component_scores["transformer"]) >= component_thresholds["transformer"]).astype(int)

    for name in names:
        if name == "transformer":
            continue
        other_preds = (np.array(component_scores[name]) >= component_thresholds[name]).astype(int)
        agree = (trans_preds == other_preds)

        # Accuracy when they agree vs disagree
        agree_mask = agree
        disagree_mask = ~agree

        if agree_mask.sum() > 0:
            agree_acc = (trans_preds[agree_mask] == np.array(y_true)[agree_mask]).mean()
        else:
            agree_acc = 0

        if disagree_mask.sum() > 0:
            # When they disagree, who is right?
            trans_right = (trans_preds[disagree_mask] == np.array(y_true)[disagree_mask]).mean()
            other_right = (other_preds[disagree_mask] == np.array(y_true)[disagree_mask]).mean()
        else:
            trans_right = other_right = 0

        print(f"\n{name} vs transformer:")
        print(f"  Agree:    {agree_mask.sum():4d} samples, accuracy={agree_acc:.3f}")
        print(f"  Disagree: {disagree_mask.sum():4d} samples, transformer right={trans_right:.3f}, {name} right={other_right:.3f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
