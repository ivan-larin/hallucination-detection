"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


TAILS = (3, 8, 16)
BLOCK_DIM = 5 * 896  # num_layers * hidden_dim


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features."""

    def __init__(self) -> None:
        super().__init__()
        # per-block (StandardScaler + PCA) for each tail
        self._scalers: dict[int, StandardScaler] = {k: StandardScaler() for k in TAILS}
        self._pcas: dict[int, PCA] = {
            k: PCA(n_components=24, whiten=True, random_state=0) for k in TAILS
        }

        self._clf: LogisticRegression | None = None
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        last = X[:, :BLOCK_DIM]

        blocks = []
        for i, k in enumerate(TAILS):
            offset = (1 + i) * BLOCK_DIM
            tail = X[:, offset : offset + BLOCK_DIM]
            view = np.concatenate([last, tail], axis=1)

            scaler, pca = self._scalers[k], self._pcas[k]
            X_scaled = scaler.fit_transform(view) if fit else scaler.transform(view)
            X_reduced = pca.fit_transform(X_scaled) if fit else pca.transform(X_scaled)

            blocks.append(X_reduced)

        return np.concatenate(blocks, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Returns:
            ``self`` (for method chaining).
        """
        X_inner_train, X_inner_val, y_inner_train, y_inner_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=0
        )

        X_processed = self._preprocess(X_inner_train, fit=True)
        self._clf = LogisticRegression(
            C=0.05, max_iter=4000, class_weight="balanced", solver="liblinear"
        )
        self._clf.fit(X_processed, y_inner_train)
        self.fit_hyperparameters(X_inner_val, y_inner_val)

        # refit on full training data so we don't waste the inner-val rows
        Xp_full = self._preprocess(X, fit=True)
        self._clf = LogisticRegression(
            C=0.05, max_iter=4000, class_weight="balanced", solver="liblinear"
        )
        self._clf.fit(Xp_full, y)

        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_processed = self._preprocess(X, fit=False)
        return self._clf.predict_proba(X_processed)
