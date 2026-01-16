from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CropPolicySpec:
    """4-policy spec for crop yield policy game table (16 patterns).

    Players (binary flags):
      - high_rain_region: Rainfall_mm clustered into k=2; higher-mean cluster -> 1
      - irrigation_used: Irrigation_Used == True -> 1
      - fertilizer_used: Fertilizer_Used == True -> 1
      - improved_soil: Soil_Type == argmax(mean(yield)) -> 1 else 0
    """

    crop_col: str = "Crop"
    soil_col: str = "Soil_Type"
    rainfall_col: str = "Rainfall_mm"
    irrigation_col: str = "Irrigation_Used"
    fertilizer_col: str = "Fertilizer_Used"
    target_col: str = "Yield_tons_per_hectare"

    rainfall_sample_size: int = 200_000
    rainfall_random_state: int = 42

    out_high_rain_flag: str = "high_rain_region"
    out_irrigation_flag: str = "irrigation_used"
    out_fertilizer_flag: str = "fertilizer_used"
    out_improved_soil_flag: str = "improved_soil"


def _mode_with_tiebreak(values: pd.Series) -> str:
    vc = values.astype("object").value_counts(dropna=True)
    if vc.empty:
        raise ValueError("Cannot compute mode for empty values.")
    max_count = int(vc.max())
    candidates = sorted([str(x) for x in vc[vc == max_count].index.tolist()])
    return candidates[0]


def infer_most_frequent_crop(df: pd.DataFrame, *, crop_col: str) -> str:
    if crop_col not in df.columns:
        raise ValueError(f"Missing crop column: {crop_col}")
    return _mode_with_tiebreak(df[crop_col])


def infer_best_soil(df: pd.DataFrame, *, soil_col: str, target_col: str) -> str:
    if soil_col not in df.columns:
        raise ValueError(f"Missing soil column: {soil_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    means = df.groupby(soil_col)[target_col].mean(numeric_only=True).dropna()
    if means.empty:
        raise ValueError("Cannot infer best soil: no valid target values.")
    # Deterministic tie-break: max mean then string sort.
    best_mean = float(means.max())
    candidates = sorted([str(x) for x in means[means == best_mean].index.tolist()])
    return candidates[0]


def _to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float) != 0
    # Accept common string forms.
    ss = s.astype("object").astype(str).str.strip().str.lower()
    return ss.isin({"true", "1", "yes", "y", "t"})


def compute_high_rain_flag(
    rainfall: pd.Series,
    *,
    sample_size: int = 200_000,
    random_state: int = 42,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Cluster rainfall into k=2 (1D) and return a 0/1 flag (higher-mean cluster=1)."""
    x = pd.to_numeric(rainfall, errors="coerce")
    mask = x.notna()
    values = x.loc[mask].astype(float).to_numpy().reshape(-1, 1)
    if values.size == 0:
        raise ValueError("Cannot compute rainfall clusters: no valid Rainfall_mm values.")

    # Fit on a subsample for speed; assign all points by nearest center.
    rng = np.random.default_rng(int(random_state))
    if int(sample_size) > 0 and len(values) > int(sample_size):
        idx = rng.choice(len(values), size=int(sample_size), replace=False)
        fit_values = values[idx]
    else:
        fit_values = values

    from sklearn.cluster import MiniBatchKMeans  # noqa: PLC0415

    km = MiniBatchKMeans(n_clusters=2, random_state=int(random_state), n_init="auto")
    km.fit(fit_values)

    centers = km.cluster_centers_.reshape(-1)
    high_cluster = int(np.argmax(centers))
    labels = km.predict(values).astype(int)
    flag_valid = (labels == high_cluster).astype(int)

    flag = pd.Series(0, index=rainfall.index, dtype=int)
    flag.loc[mask] = flag_valid
    meta = {
        "centers": [float(c) for c in centers.tolist()],
        "high_cluster": int(high_cluster),
        "sample_size_used": int(len(fit_values)),
        "random_state": int(random_state),
    }
    return flag, meta


def build_crop_policy_flags(
    df: pd.DataFrame,
    *,
    spec: CropPolicySpec = CropPolicySpec(),
    crop_value: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return per-row policy flags + target for the selected crop subset."""
    required = [
        spec.crop_col,
        spec.soil_col,
        spec.rainfall_col,
        spec.irrigation_col,
        spec.fertilizer_col,
        spec.target_col,
    ]
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    base = df[required].copy()
    base = base[base[spec.target_col].notna()].copy()

    if crop_value is None:
        crop_value = infer_most_frequent_crop(base, crop_col=spec.crop_col)
    base = base[base[spec.crop_col].astype("object") == crop_value].copy()
    if base.empty:
        raise ValueError(f"No rows after filtering crop={crop_value!r}")

    best_soil = infer_best_soil(base, soil_col=spec.soil_col, target_col=spec.target_col)
    high_rain_flag, rain_meta = compute_high_rain_flag(
        base[spec.rainfall_col],
        sample_size=int(spec.rainfall_sample_size),
        random_state=int(spec.rainfall_random_state),
    )

    out = pd.DataFrame(index=base.index)
    out[spec.out_high_rain_flag] = high_rain_flag.astype(int)
    out[spec.out_irrigation_flag] = _to_bool_series(base[spec.irrigation_col]).astype(int)
    out[spec.out_fertilizer_flag] = _to_bool_series(base[spec.fertilizer_col]).astype(int)
    out[spec.out_improved_soil_flag] = (base[spec.soil_col].astype("object").astype(str) == str(best_soil)).astype(int)
    out[spec.target_col] = pd.to_numeric(base[spec.target_col], errors="coerce").astype(float)

    meta: Dict[str, Any] = {
        "crop_selected": str(crop_value),
        "best_soil": str(best_soil),
        "rainfall": rain_meta,
        "n_rows": int(len(out)),
    }
    return out, meta


def build_crop_policy_game_table(
    factors: pd.DataFrame,
    *,
    players: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    """Build 16-pattern table: exact-match of 0/1 flags -> mean target."""
    players = [str(p) for p in players]
    for col in [*players, target_col]:
        if col not in factors.columns:
            raise ValueError(f"Missing column in factors: {col}")

    rows: list[dict[str, object]] = []
    for bits in product([0, 1], repeat=len(players)):
        mask = pd.Series(True, index=factors.index)
        for p, b in zip(players, bits, strict=True):
            mask &= factors[p].astype(int) == int(b)
        sub = factors.loc[mask]
        n = int(len(sub))
        mean_yield = float(sub[target_col].mean()) if n else float("nan")
        rows.append(
            {
                "bits": list(map(int, bits)),
                "order": int(sum(bits)),
                "n_rows": n,
                "value": mean_yield,
                "abs_value": float(abs(mean_yield)) if not np.isnan(mean_yield) else float("nan"),
                "metric": "mean_yield",
            }
        )

    base = pd.DataFrame(rows)
    out = pd.DataFrame(0, index=base.index, columns=players, dtype=int)
    for i, bits in enumerate(base["bits"].tolist()):
        for p, b in zip(players, bits, strict=True):
            out.at[i, p] = int(b)
    out["order"] = base["order"].astype(int)
    out["n_rows"] = base["n_rows"].astype(int)
    out["value"] = base["value"].astype(float)
    out["abs_value"] = base["abs_value"].astype(float)
    out["metric"] = base["metric"].astype(str)
    # Sort deterministically (same as crop/wine game tables: by order then lexicographic on player bits).
    sort_cols = ["order", *players]
    return out.sort_values(by=sort_cols, ignore_index=True)
