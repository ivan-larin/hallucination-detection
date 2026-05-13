"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Stratified 5-fold CV split with a held-out validation set per fold.
    Returns:
        A list of ``n_splits`` ``(idx_train, idx_val, idx_test)`` tuples.
    """

    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold_idx, (idx_non_test, idx_test) in enumerate(skf.split(idx, y)):
        relative_val = val_size / (1.0 - len(idx_test) / len(idx))
        idx_train, idx_val = train_test_split(
            idx_non_test,
            test_size=relative_val,
            random_state=random_state + fold_idx,
            stratify=y[idx_non_test],
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits
