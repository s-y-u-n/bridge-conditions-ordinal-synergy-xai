from __future__ import annotations

from datetime import datetime
from typing import Dict

import pandas as pd

from bci_osxai.preprocess.parse_spans import parse_span_count


def _safe_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index)


def build_features(df: pd.DataFrame, columns: Dict[str, str], inspection_year: int | None = None) -> pd.DataFrame:
    if inspection_year is None:
        inspection_year = datetime.utcnow().year

    out = pd.DataFrame(index=df.index)
    out["structure_id"] = _safe_series(df, columns["id"])

    categorical_cols = [
        "category",
        "subcategory_1",
        "type_1",
        "material_1",
        "region",
        "county",
        "owner",
        "operation_status",
    ]
    for key in categorical_cols:
        out[key] = _safe_series(df, columns[key])

    year_built = pd.to_numeric(_safe_series(df, columns["year_built"]), errors="coerce")
    last_major = pd.to_numeric(_safe_series(df, columns["last_major_rehab"]), errors="coerce")
    last_minor = pd.to_numeric(_safe_series(df, columns["last_minor_rehab"]), errors="coerce")

    out["age"] = inspection_year - year_built
    out["years_since_major_rehab"] = inspection_year - last_major
    out["years_since_minor_rehab"] = inspection_year - last_minor

    out["deck_length_m"] = pd.to_numeric(_safe_series(df, columns["deck_length_m"]), errors="coerce")
    out["width_total_m"] = pd.to_numeric(_safe_series(df, columns["width_total_m"]), errors="coerce")

    span_raw = _safe_series(df, columns["number_span_cells"])
    out["span_count"] = span_raw.apply(parse_span_count)

    return out
