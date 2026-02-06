"""Reading Probe: sklearn LogisticRegression wrapper for hidden state probes.

Loads .joblib files produced by the training pipeline (read-training/).
Probe format: {"model": LogisticRegression, "scaler": StandardScaler | None, "metadata": dict}
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ReadingProbe:
    """Sklearn LogisticRegression probe for predicting P(correct) from hidden states."""

    def __init__(self, model, scaler=None, metadata: dict | None = None):
        self._model = model
        self._scaler = scaler
        self.metadata = metadata or {}

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Return P(correct) for each sample.

        Args:
            X: Hidden states array of shape (n_samples, hidden_dim).

        Returns:
            1D array of P(correct) values, shape (n_samples,).
        """
        if self._scaler is not None:
            X = self._scaler.transform(X)
        # LogisticRegression.predict_proba returns (n_samples, 2) with [P(0), P(1)]
        # Class 1 = correct, so we return the P(correct) column
        return self._model.predict_proba(X)[:, 1]

    @classmethod
    def load(cls, path: str | Path) -> ReadingProbe:
        """Load a probe from a .joblib file.

        Args:
            path: Path to .joblib file containing
                  {"model": LogisticRegression, "scaler": StandardScaler | None, "metadata": dict}

        Returns:
            ReadingProbe instance.

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

        return cls(
            model=data["model"],
            scaler=data.get("scaler"),
            metadata=data.get("metadata", {}),
        )
