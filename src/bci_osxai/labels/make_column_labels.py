from __future__ import annotations

from typing import Optional

import pandas as pd


def make_column_labels(
    df: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
) -> pd.DataFrame:
    if id_col not in df.columns:
        raise ValueError(f"Missing id column: {id_col}")
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    id_series = df[id_col].reset_index(drop=True)
    target = df[target_col].reset_index(drop=True)

    label: pd.Series
    if pd.api.types.is_numeric_dtype(target):
        label = pd.to_numeric(target, errors="coerce")
        label_index = pd.Series([pd.NA] * len(label), dtype="Int64")
    else:
        label = target.astype("object")
        label_index = pd.Series([pd.NA] * len(label), dtype="Int64")

    out = pd.DataFrame(
        {
            "structure_id": id_series,
            "label": label,
            "label_index": label_index,
        }
    )
    return out

