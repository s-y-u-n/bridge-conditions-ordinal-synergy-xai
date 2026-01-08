import re
from typing import Optional

import pandas as pd


_SPAN_RE = re.compile(r"(\d+(?:\.\d+)?)")


def parse_span_count(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    match = _SPAN_RE.search(str(value))
    if not match:
        return None
    return float(match.group(1))


def add_span_count(df: pd.DataFrame, column: str, out_column: str = "span_count") -> pd.DataFrame:
    out = df.copy()
    out[out_column] = out[column].apply(parse_span_count)
    return out
