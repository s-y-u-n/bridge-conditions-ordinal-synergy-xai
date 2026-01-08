from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

_NUMERIC_SUFFIX_RE = re.compile(r".*?(\\d+)$")


def _ensure_writable_caches(cache_dir: str | Path = ".cache") -> None:
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    mpl_dir = cache_dir / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")


def _build_expected_label_index_fn(
    model: Any,
    feature_columns: Sequence[str],
    feature_dtypes: pd.Series,
    label_to_index: Dict[str, int],
) -> Callable[[np.ndarray], np.ndarray]:
    numeric_cols = set(feature_dtypes[feature_dtypes.apply(lambda d: np.issubdtype(d, np.number))].index)

    def f(x: np.ndarray) -> np.ndarray:
        x_df = pd.DataFrame(x, columns=list(feature_columns))
        for col in feature_columns:
            if col in numeric_cols:
                x_df[col] = pd.to_numeric(x_df[col], errors="coerce")
            else:
                x_df[col] = x_df[col].astype("object")

        if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
            proba = model.predict_proba(x_df)
            classes = [str(c) for c in model.classes_]
            weights = np.array([label_to_index.get(c, np.nan) for c in classes], dtype=float)
            if np.isnan(weights).any():
                missing = [c for c in classes if c not in label_to_index]
                raise ValueError(f"Label index mapping missing classes: {missing}")
            return proba @ weights

        preds = model.predict(x_df)
        return np.asarray(preds, dtype=float)

    return f


def _label_order_from_thresholds(thresholds: List[Dict[str, float | str]]) -> List[str]:
    def min_key(item: Dict[str, float | str]) -> float:
        return float(item.get("min_inclusive", float("-inf")))

    return [str(item["name"]) for item in sorted(thresholds, key=min_key)]


def _label_to_index_from_model_classes(model: Any) -> Dict[str, int]:
    classes = [str(c) for c in getattr(model, "classes_", [])]
    if not classes:
        return {}

    parsed: List[tuple[int, str]] = []
    for c in classes:
        m = _NUMERIC_SUFFIX_RE.match(c)
        if not m:
            parsed = []
            break
        parsed.append((int(m.group(1)), c))

    if parsed and len(parsed) == len(classes):
        parsed.sort(key=lambda t: t[0])
        return {label: idx for idx, (_, label) in enumerate(parsed)}

    classes_sorted = sorted(classes)
    return {label: idx for idx, label in enumerate(classes_sorted)}


@dataclass(frozen=True)
class ShapiqSettings:
    index: str = "k-SII"
    max_order: int = 2
    budget: int = 2000
    background_size: int = 256
    random_state: int = 42
    verbose: bool = False


def _co_key(coalition: Sequence[str]) -> str:
    return "|".join(sorted(str(x) for x in coalition))


def _compute_interactions_table_shapiq(
    *,
    model: Any,
    background_X: pd.DataFrame,
    x: pd.DataFrame,
    thresholds: List[Dict[str, float | str]] | None,
    settings: ShapiqSettings,
) -> pd.DataFrame:
    if len(x) != 1:
        raise ValueError("x must be a single-row DataFrame")

    _ensure_writable_caches()

    import shapiq  # noqa: PLC0415

    background_X = background_X.reset_index(drop=True)
    x = x.reset_index(drop=True)

    if settings.background_size < len(background_X):
        background_X = background_X.sample(n=settings.background_size, random_state=settings.random_state)

    label_to_index: Dict[str, int] = {}
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        if thresholds:
            label_order = _label_order_from_thresholds(thresholds)
            label_to_index = {label: idx for idx, label in enumerate(label_order)}
        else:
            label_to_index = _label_to_index_from_model_classes(model)
            if not label_to_index:
                raise ValueError("Could not infer label order for classification model")

    feature_columns = list(background_X.columns)
    feature_dtypes = background_X.dtypes

    predict_fn = _build_expected_label_index_fn(model, feature_columns, feature_dtypes, label_to_index)

    explainer = shapiq.TabularExplainer(
        model=predict_fn,
        data=background_X.to_numpy(),
        index=settings.index,
        max_order=settings.max_order,
        random_state=settings.random_state,
        verbose=bool(settings.verbose),
    )
    interactions = explainer.explain(x.to_numpy()[0], budget=settings.budget)

    rows: List[Dict[str, object]] = []
    for idx_tuple, value in interactions.dict_values.items():
        if not isinstance(idx_tuple, tuple):
            continue
        order = len(idx_tuple)
        if order < 1 or order > settings.max_order:
            continue
        feature_set = [feature_columns[i] for i in idx_tuple]
        rows.append(
            {
                "coalition_key": _co_key(feature_set),
                "order": int(order),
                "value": float(value),
                "abs_value": float(abs(value)),
            }
        )
    df = pd.DataFrame(rows, columns=["coalition_key", "order", "value", "abs_value"])
    if not df.empty:
        df = df.drop_duplicates(subset=["coalition_key"], keep="first")
    return df


def explain_interactions_shapiq(
    *,
    model: Any,
    background_X: pd.DataFrame,
    x: pd.DataFrame,
    thresholds: List[Dict[str, float | str]] | None,
    settings: ShapiqSettings,
    top_k: int = 5,
    min_order: int = 2,
    cache_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        table = pd.read_parquet(cache_file)
    else:
        table = _compute_interactions_table_shapiq(model=model, background_X=background_X, x=x, thresholds=thresholds, settings=settings)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(cache_file, index=False)

    results: List[Dict[str, Any]] = []
    if not table.empty:
        filtered = table[(table["order"] >= int(min_order)) & (table["order"] <= int(settings.max_order))].copy()
        filtered = filtered.sort_values("abs_value", ascending=False).head(int(top_k))
        for _, row in filtered.iterrows():
            feat = [x for x in str(row["coalition_key"]).split("|") if x]
            results.append(
                {
                    "set": feat,
                    "order": int(row["order"]),
                    "value": float(row["value"]),
                    "abs_value": float(row["abs_value"]),
                }
            )
    return results


def all_coalition_scores_shapiq(
    *,
    model: Any,
    background_X: pd.DataFrame,
    x: pd.DataFrame,
    thresholds: List[Dict[str, float | str]] | None,
    settings: ShapiqSettings,
    min_order: int = 1,
    cache_path: str | Path | None = None,
) -> List[Dict[str, Any]]:
    """Return scores for all coalitions up to `settings.max_order` (including singletons)."""
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        table = pd.read_parquet(cache_file)
    else:
        table = _compute_interactions_table_shapiq(model=model, background_X=background_X, x=x, thresholds=thresholds, settings=settings)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(cache_file, index=False)

    results: List[Dict[str, Any]] = []
    if table.empty:
        return results

    filtered = table[(table["order"] >= int(min_order)) & (table["order"] <= int(settings.max_order))].copy()
    filtered = filtered.sort_values("abs_value", ascending=False)
    for _, row in filtered.iterrows():
        feat = [x for x in str(row["coalition_key"]).split("|") if x]
        results.append(
            {
                "set": feat,
                "order": int(row["order"]),
                "value": float(row["value"]),
                "abs_value": float(row["abs_value"]),
            }
        )
    return results
