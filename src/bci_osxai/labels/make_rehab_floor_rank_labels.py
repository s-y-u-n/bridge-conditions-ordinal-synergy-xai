from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RehabFloorRankConfig:
    n_ranks: int = 5
    increase_delta: float = 0.5
    label_prefix: str = "R"
    distribution_name: str = "beta"
    alpha: float = 5.0
    beta: float = 2.0


def _extract_year(col: str) -> Optional[int]:
    # expected: bci_YYYY
    try:
        suffix = col.split("_", 1)[1]
    except IndexError:
        return None
    if len(suffix) == 4 and suffix.isdigit():
        return int(suffix)
    return None


def _infer_bci_year_columns(df: pd.DataFrame) -> List[str]:
    cols: List[Tuple[int, str]] = []
    for col in df.columns:
        if not str(col).startswith("bci_"):
            continue
        year = _extract_year(str(col))
        if year is None:
            continue
        cols.append((year, str(col)))
    cols.sort(key=lambda t: t[0])
    return [c for _, c in cols]


def _compute_floor_before_increase(series: pd.Series, increase_delta: float) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return None
    values = values[mask]
    if len(values) == 1:
        return float(values[0])

    diffs = np.diff(values)
    idx = np.where(diffs > float(increase_delta))[0]
    if len(idx) > 0:
        return float(values[int(idx[0])])
    return float(np.nanmin(values))


def _rank_thresholds_beta(n_ranks: int, alpha: float, beta: float) -> np.ndarray:
    from scipy.stats import beta as beta_dist  # noqa: PLC0415

    # boundaries excluding 0 and 1
    probs = np.linspace(0.0, 1.0, n_ranks + 1)[1:-1]
    return beta_dist.ppf(probs, a=alpha, b=beta).astype(float)


def _compute_rank(
    target_bci: float,
    floor_bci: float,
    thresholds: np.ndarray,
) -> int:
    if not np.isfinite(target_bci) or not np.isfinite(floor_bci):
        raise ValueError("target_bci and floor_bci must be finite")

    if floor_bci >= 100.0:
        return len(thresholds)

    denom = 100.0 - floor_bci
    if denom <= 0:
        return len(thresholds)

    t = (target_bci - floor_bci) / denom
    t = float(np.clip(t, 0.0, 1.0))
    # number of thresholds passed -> rank index (0..n_ranks-1)
    return int((t > thresholds).sum())


def make_rehab_floor_rank_labels(
    df: pd.DataFrame,
    *,
    id_col: str,
    target_bci_col: str,
    bci_year_cols: Sequence[str] | None = None,
    config: RehabFloorRankConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = RehabFloorRankConfig()
    if bci_year_cols is None or len(bci_year_cols) == 0:
        bci_year_cols = _infer_bci_year_columns(df)

    thresholds: np.ndarray
    if config.distribution_name == "beta":
        thresholds = _rank_thresholds_beta(config.n_ranks, config.alpha, config.beta)
    else:
        thresholds = np.linspace(0.0, 1.0, config.n_ranks + 1)[1:-1]

    id_series = df[id_col].reset_index(drop=True)
    target_bci = pd.to_numeric(df[target_bci_col], errors="coerce").reset_index(drop=True)

    floors: List[Optional[float]] = []
    for _, row in df[list(bci_year_cols)].iterrows():
        floors.append(_compute_floor_before_increase(row, config.increase_delta))
    floor_series = pd.Series(floors, index=id_series.index, dtype="float64")

    label_index: List[Optional[int]] = []
    label: List[Optional[str]] = []
    for t_bci, f_bci in zip(target_bci.to_numpy(), floor_series.to_numpy()):
        if np.isnan(t_bci) or np.isnan(f_bci):
            label_index.append(None)
            label.append(None)
            continue
        idx = _compute_rank(float(t_bci), float(f_bci), thresholds)
        label_index.append(idx)
        label.append(f"{config.label_prefix}{idx}")

    out = pd.DataFrame(
        {
            "structure_id": id_series,
            "target_bci": target_bci,
            "floor_bci": floor_series,
            "label": pd.Series(label, index=id_series.index, dtype="object"),
            "label_index": pd.Series(label_index, index=id_series.index, dtype="Int64"),
        }
    )
    return out

