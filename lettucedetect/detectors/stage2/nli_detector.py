"""NLI contradiction detector using DeBERTa-MNLI."""

from __future__ import annotations

import logging

import torch

from lettucedetect.detectors.stage2.config import NLIConfig

logger = logging.getLogger(__name__)


class NLIContradictionDetector:
    """DeBERTa-based NLI for detecting contradictions between context and answer.

    Uses batched inference to avoid sequential bottleneck on multi-passage contexts.
    Label mapping is auto-detected from model config (id2label).
    """

    def __init__(self, config: NLIConfig | None = None):
        self.config = config or NLIConfig()
        self._model = None
        self._tokenizer = None
        self._label_map = None  # Auto-detected from model

    def _load_model(self) -> None:
        """Lazy load model and tokenizer."""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name
            )

            # Auto-detect label mapping from model config
            id2label = self._model.config.id2label
            self._label_map = {v.lower(): k for k, v in id2label.items()}

            device = self.config.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = self._model.to(device)
            self._model.eval()
            self._device = device

    def predict_batch(self, premises: list[str], hypotheses: list[str]) -> list[dict]:
        """Batched NLI prediction - critical for multi-passage efficiency.

        Args:
            premises: List of premise texts (context passages).
            hypotheses: List of hypothesis texts (usually the answer repeated).

        Returns:
            List of dicts with entailment/neutral/contradiction probabilities.
            On error, returns neutral defaults (0.33, 0.34, 0.33) for each pair.
        """
        if not premises or not hypotheses:
            return []

        if len(premises) != len(hypotheses):
            raise ValueError(
                f"premises and hypotheses must have same length, "
                f"got {len(premises)} and {len(hypotheses)}"
            )

        try:
            self._load_model()

            inputs = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            # Use label map for correct indices
            ent_idx = self._label_map.get("entailment", 0)
            neu_idx = self._label_map.get("neutral", 1)
            con_idx = self._label_map.get("contradiction", 2)

            results = []
            for p in probs:
                entailment = float(p[ent_idx])
                neutral = float(p[neu_idx])
                contradiction = float(p[con_idx])
                results.append(
                    {
                        "entailment": entailment,
                        "neutral": neutral,
                        "contradiction": contradiction,
                        "non_contradiction": entailment + neutral,
                    }
                )
            return results

        except Exception as e:
            logger.warning(f"NLI prediction failed: {e}, returning neutral defaults")
            return [
                {
                    "entailment": 0.33,
                    "neutral": 0.34,
                    "contradiction": 0.33,
                    "non_contradiction": 0.67,
                }
                for _ in premises
            ]

    def predict_single(self, premise: str, hypothesis: str) -> dict:
        """Single-pair prediction (wrapper around batch)."""
        return self.predict_batch([premise], [hypothesis])[0]

    def compute_context_nli(self, context_texts: list[str], answer: str) -> dict:
        """Compute NLI against all context passages using batched inference.

        Args:
            context_texts: List of context passages.
            answer: The answer to check for contradiction.

        Returns:
            Dict with hallucination_score and raw component scores.
            hallucination_score uses max_contradiction (best AUROC on RAGTruth).
        """
        if not context_texts:
            return {
                "hallucination_score": 0.5,
                "max_contradiction": 0.0,
                "min_non_contradiction": 1.0,
                "mean_entailment": 0.5,
                "mean_contradiction": 0.0,
            }

        premises = context_texts
        hypotheses = [answer] * len(context_texts)
        results = self.predict_batch(premises, hypotheses)

        # Compute scores across all context passages
        max_contradiction = max(r["contradiction"] for r in results)
        mean_entailment = sum(r["entailment"] for r in results) / len(results)
        mean_contradiction = sum(r["contradiction"] for r in results) / len(results)

        # Compute hallucination score based on config mode
        if self.config.score_mode == "weighted":
            # Weighted combination (experimental)
            ent_weight = self.config.entailment_weight
            con_weight = self.config.contradiction_weight
            hallucination_score = (
                ent_weight * (1.0 - mean_entailment) + con_weight * mean_contradiction
            )
        else:
            # Default: use max_contradiction (best AUROC 0.667 on RAGTruth)
            hallucination_score = max_contradiction

        return {
            "hallucination_score": hallucination_score,
            "max_contradiction": max_contradiction,
            "min_non_contradiction": min(r["non_contradiction"] for r in results),
            "mean_entailment": mean_entailment,
            "mean_contradiction": mean_contradiction,
            "all_results": results,
        }

    def warmup(self) -> None:
        """Preload model for consistent latency."""
        self._load_model()
        _ = self.predict_single("warmup premise", "warmup hypothesis")

    def preload(self) -> None:
        """Alias for warmup() to match augmentation interface."""
        self.warmup()
