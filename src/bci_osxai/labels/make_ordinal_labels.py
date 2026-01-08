from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def _match_label(value: float, thresholds: List[Dict[str, float | str]]) -> Optional[str]:
    for item in thresholds:
        name = str(item["name"])
        min_inclusive = item.get("min_inclusive")
        max_inclusive = item.get("max_inclusive")
        max_exclusive = item.get("max_exclusive")

        if min_inclusive is not None and value < float(min_inclusive):
            continue
        if max_inclusive is not None and value > float(max_inclusive):
            continue
        if max_exclusive is not None and value >= float(max_exclusive):
            continue
        return name
    return None


def make_ordinal_labels(
    df: pd.DataFrame,
    id_col: str,
    bci_col: str,
    thresholds: List[Dict[str, float | str]],
) -> pd.DataFrame:
    id_series = df[id_col].reset_index(drop=True)
    bci_values = pd.to_numeric(df[bci_col], errors="coerce")
    labels: List[Optional[str]] = []
    for value in bci_values:
        if pd.isna(value):
            labels.append(None)
            continue
        labels.append(_match_label(float(value), thresholds))

    label_order = [str(item["name"]) for item in thresholds]
    labels_series = pd.Series(labels, index=id_series.index, name="label")
    label_index = labels_series.map({name: idx for idx, name in enumerate(label_order)})

    out = pd.DataFrame(
        {
            "structure_id": id_series,
            "label": labels_series,
            "label_index": label_index,
        }
    )
    return out
