from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def load_raw_csv(path: PathLike, encoding: str | None = None) -> pd.DataFrame:
    """Load the raw Ontario bridge conditions CSV."""
    return pd.read_csv(Path(path), encoding=encoding)


def load_raw_arff(path: PathLike) -> pd.DataFrame:
    """Load an ARFF file into a pandas DataFrame.

    Notes:
    - `scipy.io.arff.loadarff` often returns nominal columns as `bytes`; we decode them to `str`.
    - Missing values may appear as '?' strings; we normalize them to NA.
    """
    from scipy.io import arff  # noqa: PLC0415

    data, _meta = arff.loadarff(str(Path(path)))
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x)
            df[col] = df[col].replace("?", pd.NA)
    return df
