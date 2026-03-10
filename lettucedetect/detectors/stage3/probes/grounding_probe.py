"""Hallucination Probe: sklearn LogisticRegression wrapper for hidden state probes.

Loads .joblib files produced by the hallu-training pipeline.
Probe format: {"model": LogisticRegression, "scaler": StandardScaler | None,
               "metadata": dict, "probe_type": "hallucination"}

Score direction: Returns P(hallucinated) directly.
  0.0 = supported, 1.0 = hallucinated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class GroundingProbe:
    """Sklearn LogisticRegression probe for predicting P(hallucinated) from hidden states."""

    def __init__(self, model, scaler=None, pca=None, metadata: dict | None = None):
        self._model = model
        self._scaler = scaler
        self._pca = pca
        self.metadata = metadata or {}

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return P(hallucinated) for each sample.

        Args:
            X: Hidden states array of shape (n_samples, hidden_dim).

        Returns:
            1D array of P(hallucinated) values, shape (n_samples,).
        """
        if self._scaler is not None:
            X = self._scaler.transform(X)
        if self._pca is not None:
            X = self._pca.transform(X)
        # LogisticRegression.predict_proba returns (n_samples, 2) with [P(0), P(1)]
        # Class 1 = hallucinated, so we return the P(hallucinated) column
        return self._model.predict_proba(X)[:, 1]

    @classmethod
    def load(cls, path: str | Path) -> GroundingProbe:
        """Load a probe from a .joblib file.

        Args:
            path: Path to .joblib file containing
                  {"model": LogisticRegression, "scaler": StandardScaler | None,
                   "metadata": dict, "pca": PCA | None}

        Returns:
            GroundingProbe instance.

        Raises:
            FileNotFoundError: If probe file doesn't exist.
            KeyError: If probe file missing required 'model' key.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Probe file not found: {path}")

        data = joblib.load(path)

        if "model" not in data:
            raise KeyError(f"Probe file missing 'model' key. Keys found: {list(data.keys())}")

        cls._validate_loaded_objects(data)

        return cls(
            model=data["model"],
            scaler=data.get("scaler"),
            pca=data.get("pca"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_hub(cls, repo_id: str, filename: str) -> GroundingProbe:
        """Download probe from HuggingFace Hub and load it.

        Args:
            repo_id: HuggingFace repo ID (e.g., "MRathmayr/lettucedetect-grounding-probes").
            filename: Probe filename (e.g., "probe_3b_qwen_pca512.joblib").

        Returns:
            GroundingProbe instance.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for from_hub(). "
                "Install with: pip install lettucedetect[stage3]"
            )

        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.debug(f"Downloaded probe to: {local_path}")
        return cls.load(path=local_path)

    @staticmethod
    def _validate_loaded_objects(data: dict) -> None:
        """Validate deserialized probe objects to mitigate supply-chain risk."""
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        model = data["model"]
        if not isinstance(model, LogisticRegression):
            raise TypeError(
                f"Expected LogisticRegression model, got {type(model).__name__}"
            )

        scaler = data.get("scaler")
        if scaler is not None and not isinstance(scaler, StandardScaler):
            raise TypeError(
                f"Expected StandardScaler or None, got {type(scaler).__name__}"
            )

        pca = data.get("pca")
        if pca is not None and not isinstance(pca, PCA):
            raise TypeError(
                f"Expected PCA or None, got {type(pca).__name__}"
            )
