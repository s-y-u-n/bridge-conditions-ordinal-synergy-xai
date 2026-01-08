from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def apply_baseline_mask(row: pd.Series, features: Iterable[str], baseline: Dict[str, object]) -> pd.Series:
    masked = row.copy()
    for feature in features:
        if feature in baseline:
            masked[feature] = baseline[feature]
    return masked
