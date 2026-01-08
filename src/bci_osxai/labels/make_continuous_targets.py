from __future__ import annotations

import pandas as pd


def make_continuous_targets(df: pd.DataFrame, *, id_col: str, bci_col: str) -> pd.DataFrame:
    id_series = df[id_col].reset_index(drop=True)
    target = pd.to_numeric(df[bci_col], errors="coerce").reset_index(drop=True)
    return pd.DataFrame(
        {
            "structure_id": id_series,
            "label": target,  # keep column name for downstream compatibility
            "label_index": pd.Series([pd.NA] * len(target), dtype="Int64"),
        }
    )

