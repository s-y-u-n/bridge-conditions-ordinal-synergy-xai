from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score


def _to_index(values: pd.Series, label_order: List[str]) -> pd.Series:
    mapping = {label: idx for idx, label in enumerate(label_order)}
    return values.map(mapping)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    label_order: List[str],
) -> Dict[str, float]:
    y_true_index = _to_index(y_true, label_order)
    y_pred_index = _to_index(y_pred, label_order)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "mae": (y_true_index - y_pred_index).abs().mean(),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
    }
    return metrics
