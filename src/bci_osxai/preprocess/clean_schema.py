from typing import Mapping

import pandas as pd


def normalize_column_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def standardize_columns(df: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)
