from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NextRankConfig:
    n_ranks: int = 5
    increase_delta: float = 0.5
    label_prefix: str = "R"
    distribution_name: str = "beta"
    alpha: float = 5.0
    beta: float = 2.0


def _rank_thresholds_beta(n_ranks: int, alpha: float, beta: float) -> np.ndarray:
    from scipy.stats import beta as beta_dist  # noqa: PLC0415

    probs = np.linspace(0.0, 1.0, n_ranks + 1)[1:-1]
    return beta_dist.ppf(probs, a=alpha, b=beta).astype(float)


def _compute_floor_prefix(values: np.ndarray, increase_delta: float) -> np.ndarray:
    floors = np.empty_like(values, dtype=float)
    running_min = np.inf
    pre_increase_floor: Optional[float] = None
    for i, v in enumerate(values):
        running_min = min(running_min, v)
        if pre_increase_floor is None and i > 0:
            if (v - values[i - 1]) > float(increase_delta):
                pre_increase_floor = float(values[i - 1])
        floors[i] = pre_increase_floor if pre_increase_floor is not None else running_min
    return floors


def _rank_from_floor(target_bci: float, floor_bci: float, thresholds: np.ndarray) -> int:
    if floor_bci >= 100.0:
        return len(thresholds)
    denom = 100.0 - floor_bci
    if denom <= 0:
        return len(thresholds)
    t = (target_bci - floor_bci) / denom
    t = float(np.clip(t, 0.0, 1.0))
    return int((t > thresholds).sum())


def _label(idx: int, prefix: str) -> str:
    return f"{prefix}{idx}"


def build_next_rank_dataset(
    *,
    structures: pd.DataFrame,
    bci_long: pd.DataFrame,
    config: NextRankConfig,
    id_col: str = "structure_id",
    year_col: str = "year",
    bci_col: str = "bci",
) -> pd.DataFrame:
    if config.distribution_name == "beta":
        thresholds = _rank_thresholds_beta(config.n_ranks, config.alpha, config.beta)
    else:
        thresholds = np.linspace(0.0, 1.0, config.n_ranks + 1)[1:-1]

    bci_long = bci_long.copy()
    bci_long = bci_long[bci_long[year_col].notna()].copy()
    bci_long[year_col] = pd.to_numeric(bci_long[year_col], errors="coerce").astype("Int64")
    bci_long[bci_col] = pd.to_numeric(bci_long[bci_col], errors="coerce")
    bci_long = bci_long.dropna(subset=[year_col, bci_col])

    structure_cols = [
        "category",
        "subcategory_1",
        "type_1",
        "material_1",
        "region",
        "county",
        "owner",
        "operation_status",
        "year_built",
        "last_major_rehab",
        "last_minor_rehab",
        "deck_or_culvert_length_m",
        "width_total_m",
        "span_cells_count",
    ]
    available_cols = [c for c in structure_cols if c in structures.columns]
    structures_small = structures[[id_col] + available_cols].copy()

    rows: List[Dict[str, Any]] = []
    for structure_id, grp in bci_long.groupby(id_col, sort=False):
        grp_sorted = grp.sort_values(year_col)
        years = grp_sorted[year_col].to_numpy(dtype=int)
        values = grp_sorted[bci_col].to_numpy(dtype=float)
        if len(values) < 2:
            continue

        floors = _compute_floor_prefix(values, config.increase_delta)
        ranks = np.array([_rank_from_floor(v, f, thresholds) for v, f in zip(values, floors)], dtype=int)

        prev_delta = np.full(len(values), np.nan, dtype=float)
        prev_delta[1:] = values[1:] - values[:-1]

        for i in range(len(values) - 1):
            year_t = int(years[i])
            year_next = int(years[i + 1])
            delta_years = int(year_next - year_t)

            rows.append(
                {
                    id_col: structure_id,
                    "year": year_t,
                    "year_next": year_next,
                    "delta_years": delta_years,
                    "current_bci": float(values[i]),
                    "current_floor_bci": float(floors[i]),
                    "current_rank_index": int(ranks[i]),
                    "current_rank": _label(int(ranks[i]), config.label_prefix),
                    "prev_bci_delta": float(prev_delta[i]) if np.isfinite(prev_delta[i]) else np.nan,
                    "target_bci": float(values[i + 1]),
                    "target_rank_index": int(ranks[i + 1]),
                    "target_rank": _label(int(ranks[i + 1]), config.label_prefix),
                }
            )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        return dataset

    dataset = dataset.merge(structures_small, on=id_col, how="left")

    for col in ["year_built", "last_major_rehab", "last_minor_rehab"]:
        if col in dataset.columns:
            dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

    if "year_built" in dataset.columns:
        dataset["age_at_year"] = dataset["year"] - dataset["year_built"]
    if "last_major_rehab" in dataset.columns:
        dataset["years_since_major_rehab_at_year"] = dataset["year"] - dataset["last_major_rehab"]
    if "last_minor_rehab" in dataset.columns:
        dataset["years_since_minor_rehab_at_year"] = dataset["year"] - dataset["last_minor_rehab"]

    dataset["label"] = dataset["target_rank"]
    dataset["label_index"] = dataset["target_rank_index"].astype("Int64")
    return dataset


def build_latest_features_for_next_rank(
    *,
    structures: pd.DataFrame,
    bci_long: pd.DataFrame,
    structure_id: str,
    config: NextRankConfig,
    id_col: str = "structure_id",
    year_col: str = "year",
    bci_col: str = "bci",
) -> pd.DataFrame:
    if config.distribution_name == "beta":
        thresholds = _rank_thresholds_beta(config.n_ranks, config.alpha, config.beta)
    else:
        thresholds = np.linspace(0.0, 1.0, config.n_ranks + 1)[1:-1]

    grp = bci_long[(bci_long[id_col].astype(str) == str(structure_id)) & bci_long[year_col].notna()].copy()
    if grp.empty:
        raise ValueError(f"No yearly BCI found for structure_id={structure_id}")
    grp[year_col] = pd.to_numeric(grp[year_col], errors="coerce").astype("Int64")
    grp[bci_col] = pd.to_numeric(grp[bci_col], errors="coerce")
    grp = grp.dropna(subset=[year_col, bci_col]).sort_values(year_col)

    years = grp[year_col].to_numpy(dtype=int)
    values = grp[bci_col].to_numpy(dtype=float)
    floors = _compute_floor_prefix(values, config.increase_delta)
    ranks = np.array([_rank_from_floor(v, f, thresholds) for v, f in zip(values, floors)], dtype=int)

    prev_bci_delta = np.nan
    if len(values) >= 2:
        prev_bci_delta = float(values[-1] - values[-2])

    last_year = int(years[-1])
    row = {
        id_col: str(structure_id),
        "year": last_year,
        "delta_years": np.nan,
        "current_bci": float(values[-1]),
        "current_floor_bci": float(floors[-1]),
        "current_rank_index": int(ranks[-1]),
        "current_rank": _label(int(ranks[-1]), config.label_prefix),
        "prev_bci_delta": prev_bci_delta,
    }

    structure_cols = [
        "category",
        "subcategory_1",
        "type_1",
        "material_1",
        "region",
        "county",
        "owner",
        "operation_status",
        "year_built",
        "last_major_rehab",
        "last_minor_rehab",
        "deck_or_culvert_length_m",
        "width_total_m",
        "span_cells_count",
    ]
    srow = structures[structures[id_col].astype(str) == str(structure_id)]
    if not srow.empty:
        for c in structure_cols:
            if c in srow.columns:
                row[c] = srow.iloc[0][c]

    out = pd.DataFrame([row])
    for col in ["year_built", "last_major_rehab", "last_minor_rehab"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "year_built" in out.columns:
        out["age_at_year"] = out["year"] - out["year_built"]
    if "last_major_rehab" in out.columns:
        out["years_since_major_rehab_at_year"] = out["year"] - out["last_major_rehab"]
    if "last_minor_rehab" in out.columns:
        out["years_since_minor_rehab_at_year"] = out["year"] - out["last_minor_rehab"]

    # Ensure missing numeric values are NaN (not pd.NA), for sklearn compatibility.
    for col in out.columns:
        if col in {id_col, "category", "subcategory_1", "type_1", "material_1", "region", "county", "owner", "operation_status", "current_rank"}:
            continue
        if pd.api.types.is_numeric_dtype(out[col]) or col in {
            "year",
            "delta_years",
            "current_bci",
            "current_floor_bci",
            "current_rank_index",
            "prev_bci_delta",
            "deck_or_culvert_length_m",
            "width_total_m",
            "span_cells_count",
            "age_at_year",
            "years_since_major_rehab_at_year",
            "years_since_minor_rehab_at_year",
        }:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    return out
