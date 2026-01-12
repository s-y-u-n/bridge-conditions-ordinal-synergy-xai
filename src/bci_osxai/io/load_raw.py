from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import pandas as pd


PathLike = Union[str, Path]


def load_raw_csv(
    path: PathLike,
    *,
    encoding: str | None = None,
    has_header: bool = True,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a raw CSV dataset.

    Some UCI-style datasets ship without a header row; set has_header=False and
    provide `columns` to name the fields.
    """
    if has_header:
        return pd.read_csv(Path(path), encoding=encoding)
    if not columns:
        raise ValueError("columns is required when has_header=False.")
    return pd.read_csv(Path(path), encoding=encoding, header=None, names=list(columns))
