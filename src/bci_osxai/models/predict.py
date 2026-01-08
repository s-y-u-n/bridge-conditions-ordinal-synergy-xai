from __future__ import annotations

import pickle
from pathlib import Path
import pandas as pd


def load_model(path: str | Path):
    path = Path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict(model, features: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(features), index=features.index)
