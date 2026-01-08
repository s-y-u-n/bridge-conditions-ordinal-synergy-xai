from typing import Iterable, List, Optional

import pandas as pd


def build_bci_long(
    df: pd.DataFrame,
    id_col: str,
    year_cols: Iterable[str],
    current_bci_col: Optional[str] = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    year_cols = [col for col in year_cols if col in df.columns]
    if year_cols:
        long_years = df.melt(
            id_vars=[id_col],
            value_vars=year_cols,
            var_name="year",
            value_name="bci",
        )
        long_years["year"] = long_years["year"].astype(str).str.extract(r"(\d{4})", expand=False)
        long_years["year"] = pd.to_numeric(long_years["year"], errors="coerce")
        long_years["bci_source"] = "YEAR_COL"
        frames.append(long_years)

    if current_bci_col and current_bci_col in df.columns:
        current = df[[id_col, current_bci_col]].rename(columns={current_bci_col: "bci"})
        current["year"] = pd.NA
        current["bci_source"] = "CURRENT_BCI"
        frames.append(current)

    if not frames:
        return pd.DataFrame(columns=[id_col, "year", "bci", "bci_source"])

    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype("Int64")
    return out
