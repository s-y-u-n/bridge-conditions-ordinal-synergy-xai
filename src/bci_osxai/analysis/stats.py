from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V for categorical-categorical association.

    Returns NaN if association is undefined (e.g., empty table).
    Uses the bias-corrected variant (Bergsma 2013 style correction).
    """
    x = x.astype("object")
    y = y.astype("object")
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if df.empty:
        return float("nan")

    table = pd.crosstab(df["x"], df["y"])
    if table.size == 0:
        return float("nan")

    observed = table.to_numpy(dtype=float)
    n = float(observed.sum())
    if n <= 0:
        return float("nan")

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    r, k = observed.shape
    if r <= 1 or k <= 1:
        return 0.0

    phi2 = chi2 / n
    # Bias correction.
    phi2_corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / max(n - 1.0, 1.0))
    r_corr = r - ((r - 1.0) ** 2) / max(n - 1.0, 1.0)
    k_corr = k - ((k - 1.0) ** 2) / max(n - 1.0, 1.0)
    denom = min(k_corr - 1.0, r_corr - 1.0)
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(phi2_corr / denom))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio (eta) for categorical-numeric association."""
    df = pd.DataFrame({"c": categories.astype("object"), "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if df.empty:
        return float("nan")

    y = df["v"].to_numpy(dtype=float)
    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    if sst <= 0:
        return 0.0

    ssb = 0.0
    for _, group in df.groupby("c", observed=True):
        yg = group["v"].to_numpy(dtype=float)
        if len(yg) == 0:
            continue
        ssb += float(len(yg)) * float((np.mean(yg) - y_mean) ** 2)
    eta2 = max(0.0, min(1.0, ssb / sst))
    return float(math.sqrt(eta2))


def safe_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value

