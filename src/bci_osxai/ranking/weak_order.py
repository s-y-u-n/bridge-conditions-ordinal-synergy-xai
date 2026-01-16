from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd


Criterion = Literal["bic", "mdl"]


def _infer_player_columns(df: pd.DataFrame, *, score_col: str) -> list[str]:
    exclude = {
        str(score_col),
        "value",
        "abs_value",
        "metric",
        "order",
        "n_train",
        "n_test",
        "seed",
        "coalition_key",
        "coalition_id",
        "class_id",
        "k_selected",
        "class_score_max",
        "class_score_min",
        "class_size",
    }
    cols: list[str] = []
    for c in df.columns:
        sc = str(c)
        if sc in exclude:
            continue
        # Prefer 0/1 indicator columns.
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
            if vals.issubset({0, 1}):
                cols.append(sc)
    return cols


def _row_to_coalition_id(row: pd.Series, player_cols: Sequence[str]) -> str:
    members = [c for c in player_cols if int(row[c]) == 1]
    return "|".join(members) if members else "EMPTY"


def _segment_sse(prefix_x: np.ndarray, prefix_x2: np.ndarray, l: int, r: int) -> float:
    """SSE for segment [l, r] inclusive, 1-indexed."""
    n = r - l + 1
    if n <= 0:
        return 0.0
    s1 = prefix_x[r] - prefix_x[l - 1]
    s2 = prefix_x2[r] - prefix_x2[l - 1]
    mean = s1 / n
    return float(s2 - 2.0 * mean * s1 + n * mean * mean)


def _optimal_sse_1d_kmeans(x_desc: np.ndarray, k_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal SSE for k=1..k_max with contiguous segmentation.

    Returns:
      sse_by_k: shape (k_max,) with SSE for each k (1-indexed conceptually)
      boundaries_by_k: shape (k_max, n) with backpointers (end indices), -1 for unused
    """
    x = np.asarray(x_desc, dtype=float)
    n = int(len(x))
    k_max = int(min(max(k_max, 1), n))

    # 1-indexed prefix sums for O(1) segment cost.
    prefix_x = np.zeros(n + 1, dtype=float)
    prefix_x2 = np.zeros(n + 1, dtype=float)
    prefix_x[1:] = np.cumsum(x)
    prefix_x2[1:] = np.cumsum(x * x)

    def cost(l: int, r: int) -> float:
        return _segment_sse(prefix_x, prefix_x2, l, r)

    # dp[k][i] = min SSE for partitioning first i points into k clusters.
    dp_prev = np.full(n + 1, np.inf, dtype=float)
    dp_curr = np.full(n + 1, np.inf, dtype=float)
    dp_prev[0] = 0.0

    # back[k][i] = argmin m where last segment is (m+1..i)
    back = np.full((k_max + 1, n + 1), -1, dtype=int)

    def compute_layer(k: int, i_left: int, i_right: int, opt_left: int, opt_right: int) -> None:
        mid = (i_left + i_right) // 2
        best_m = -1
        best_val = np.inf

        m_start = max(opt_left, k - 1)
        m_end = min(opt_right, mid - 1)
        for m in range(m_start, m_end + 1):
            val = dp_prev[m] + cost(m + 1, mid)
            if val < best_val:
                best_val = val
                best_m = m

        dp_curr[mid] = best_val
        back[k, mid] = best_m

        if i_left <= mid - 1:
            compute_layer(k, i_left, mid - 1, opt_left, best_m)
        if mid + 1 <= i_right:
            compute_layer(k, mid + 1, i_right, best_m, opt_right)

    sse_by_k = np.full(k_max + 1, np.nan, dtype=float)
    for k in range(1, k_max + 1):
        dp_curr.fill(np.inf)
        dp_curr[0] = np.inf
        compute_layer(k, k, n, k - 1, n - 1)
        sse_by_k[k] = dp_curr[n]
        dp_prev, dp_curr = dp_curr, dp_prev

    return sse_by_k[1:], back[1:]


def _criterion_value(*, sse: float, n: int, k: int, criterion: Criterion) -> float:
    # Gaussian approximation, ignoring additive constants:
    # BIC ~ n*log(SSE/n) + p*log(n). Use p=k (means) + 1 (variance).
    n = int(n)
    k = int(k)
    if n <= 0:
        return float("nan")
    eps = 1e-12
    sse = float(max(sse, eps))
    sigma2 = sse / float(n)
    p = k + 1
    bic = float(n * math.log(sigma2) + p * math.log(n))
    if criterion in {"bic", "mdl"}:
        return bic
    raise ValueError(f"Unknown criterion: {criterion}")


def _reconstruct_boundaries(back: np.ndarray, *, k: int, n: int) -> list[int]:
    """Return end indices (1-indexed) of each segment in order."""
    boundaries: list[int] = []
    i = int(n)
    kk = int(k)
    while kk >= 1:
        m = int(back[kk - 1, i])
        boundaries.append(i)
        i = m
        kk -= 1
    boundaries.reverse()
    return boundaries


def rank_to_weak_order(
    df: pd.DataFrame,
    *,
    id_col: str = "coalition_id",
    score_col: str = "value",
    k_max: int | None = None,
    k_fixed: int | None = None,
    criterion: Criterion = "bic",
    higher_is_better: bool | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Assign class_id via optimal 1D k-means segmentation on descending scores.

    If id_col is missing, it is derived from 0/1 feature indicator columns.
    Rows with NaN scores are dropped (reported in summary).
    """
    if df.empty:
        out = df.copy()
        if id_col not in out.columns:
            out[id_col] = pd.Series([], dtype="object")
        out["class_id"] = pd.Series([], dtype="Int64")
        out["k_selected"] = pd.Series([], dtype="Int64")
        out["class_score_max"] = pd.Series([], dtype=float)
        out["class_score_min"] = pd.Series([], dtype=float)
        out["class_size"] = pd.Series([], dtype="Int64")
        return out, [], {"n": 0, "k_selected": 0, "criterion": str(criterion), "criterion_values": {}, "sse_values": {}, "boundaries": []}

    work = df.copy()

    if id_col not in work.columns:
        player_cols = _infer_player_columns(work, score_col=str(score_col))
        if not player_cols:
            # fallback: stable row id
            work[id_col] = [f"row{i}" for i in range(len(work))]
        else:
            work[id_col] = work.apply(lambda r: _row_to_coalition_id(r, player_cols), axis=1)

    scores = pd.to_numeric(work[score_col], errors="coerce")
    mask = scores.notna()
    dropped = int((~mask).sum())
    work = work.loc[mask].copy()
    scores = scores.loc[mask].astype(float)

    n = int(len(work))
    if n == 0:
        out = df.copy()
        return out, [], {"n": 0, "k_selected": 0, "criterion": str(criterion), "criterion_values": {}, "sse_values": {}, "boundaries": []}

    if higher_is_better is None and "metric" in work.columns:
        m = work["metric"].astype(str).dropna().unique().tolist()
        if len(m) == 1 and m[0] == "mae":
            higher_is_better = False
    if higher_is_better is None:
        higher_is_better = True

    # Stable sort by score (best->worst) then by id.
    score_key = scores.to_numpy()
    if bool(higher_is_better):
        score_key = -score_key
    order = np.lexsort((work[id_col].astype(str).to_numpy(), score_key))
    work_sorted = work.iloc[order].copy()
    x_desc = pd.to_numeric(work_sorted[score_col], errors="coerce").to_numpy(dtype=float)

    if n == 1:
        out_sorted = work_sorted.copy()
        out_sorted["class_id"] = 1
        out_sorted["k_selected"] = 1
        out_sorted["class_score_max"] = float(x_desc[0])
        out_sorted["class_score_min"] = float(x_desc[0])
        out_sorted["class_size"] = 1
        classes = [{"class_id": 1, "members": [str(out_sorted[id_col].iloc[0])]}]
        summary = {
            "n": int(n),
            "n_dropped_nan": int(dropped),
            "k_selected": 1,
            "k_fixed": None,
            "criterion": str(criterion),
            "criterion_values": {"1": float("nan")},
            "sse_values": {"1": 0.0},
            "boundaries": [1],
        }
        out = out_sorted.sort_index()
        return out, classes, summary

    if k_fixed is not None:
        k_fixed = int(k_fixed)
        if k_fixed < 1:
            raise ValueError("k_fixed must be >= 1.")
        if k_fixed > n:
            k_fixed = n

    if k_max is None:
        k_max = min(20, n)
    k_max = int(max(1, min(int(k_max), n)))
    if k_fixed is not None:
        k_max = max(k_max, int(k_fixed))

    sse_arr, back = _optimal_sse_1d_kmeans(x_desc, k_max)
    sse_values = [float(v) for v in sse_arr.tolist()]

    crit_values: list[float] = []
    for k in range(1, k_max + 1):
        crit_values.append(_criterion_value(sse=sse_values[k - 1], n=n, k=k, criterion=criterion))

    if k_fixed is not None:
        k_selected = int(k_fixed)
    else:
        k_selected = int(np.nanargmin(np.asarray(crit_values, dtype=float)) + 1)
    boundaries = _reconstruct_boundaries(back, k=k_selected, n=n)

    # Assign classes in sorted order.
    class_ids = np.empty(n, dtype=int)
    start = 0
    classes: list[dict[str, Any]] = []
    for class_id, end_1idx in enumerate(boundaries, start=1):
        end = int(end_1idx)
        class_ids[start:end] = class_id
        members = work_sorted.iloc[start:end][id_col].astype(str).tolist()
        classes.append({"class_id": int(class_id), "members": members})
        start = end

    out_sorted = work_sorted.copy()
    out_sorted["class_id"] = class_ids
    out_sorted["k_selected"] = int(k_selected)

    # Per-class ranges/sizes
    out_sorted["class_score_max"] = out_sorted.groupby("class_id")[score_col].transform("max")
    out_sorted["class_score_min"] = out_sorted.groupby("class_id")[score_col].transform("min")
    out_sorted["class_size"] = out_sorted.groupby("class_id")[score_col].transform("size").astype(int)

    summary: dict[str, Any] = {
        "n": int(n),
        "n_dropped_nan": int(dropped),
        "k_selected": int(k_selected),
        "k_fixed": int(k_fixed) if k_fixed is not None else None,
        "criterion": str(criterion),
        "higher_is_better": bool(higher_is_better),
        "criterion_values": {str(i + 1): float(v) for i, v in enumerate(crit_values)},
        "sse_values": {str(i + 1): float(v) for i, v in enumerate(sse_values)},
        "boundaries": [int(b) for b in boundaries],
    }

    # Return in original row order for easy join-back semantics.
    out = out_sorted.sort_index()
    return out, classes, summary


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
