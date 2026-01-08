from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def load_raw_csv(path: PathLike, encoding: str | None = None) -> pd.DataFrame:
    """Load the raw Ontario bridge conditions CSV."""
    return pd.read_csv(Path(path), encoding=encoding)
