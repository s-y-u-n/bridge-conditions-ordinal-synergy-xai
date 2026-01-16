from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from bci_osxai.utils.progress import tqdm_wrap


@dataclass(frozen=True)
class GameTableSettings:
    enabled: bool = True
    max_order: int = 2
    n_coalitions: int = 400
    test_size: float = 0.25
    metric: str = "accuracy"  # "accuracy" | "neg_mae" | "inv_mae" | "mae"
    seed: int = 42
    n_repeats: int = 1
    cv_folds: int = 0


def coalition_key(coalition: Sequence[str]) -> str:
    return "|".join(sorted(str(x) for x in coalition))


def _mode_with_tiebreak(values: pd.Series) -> str:
    """Most frequent label with deterministic tie-break by string sort."""
    vc = values.astype("object").value_counts(dropna=True)
    if vc.empty:
        raise ValueError("Cannot compute mode for empty labels.")
    max_count = int(vc.max())
    candidates = sorted([str(x) for x in vc[vc == max_count].index.tolist()])
    return candidates[0]


def build_coalition_patterns(players: Sequence[str], *, max_order: int | None = None) -> pd.DataFrame:
    """Build a wide 0/1 table of all coalition patterns up to max_order (no model training)."""
    players = [str(p) for p in players]
    n = len(players)
    if n == 0:
        return pd.DataFrame(columns=["order", "coalition_key"])

    if max_order is None:
        max_order = n
    max_order = max(1, int(max_order))
    max_order = min(max_order, n)

    from itertools import combinations  # noqa: PLC0415

    coalitions: list[list[str]] = []
    for k in range(1, max_order + 1):
        for combo in combinations(players, k):
            coalitions.append(list(combo))

    base = pd.DataFrame({"coalition_key": [coalition_key(c) for c in coalitions], "order": [len(c) for c in coalitions]})
    out = pd.DataFrame(0, index=base.index, columns=players, dtype=int)
    for i, coalition in enumerate(coalitions):
        for p in coalition:
            out.at[i, p] = 1
    out["order"] = base["order"].astype(int)
    out["coalition_key"] = base["coalition_key"].astype(str)
    return out


def _generate_coalitions(players: Sequence[str], *, max_order: int, n_coalitions: int, seed: int) -> List[List[str]]:
    players = [str(p) for p in players]
    n = len(players)
    max_order = max(1, int(max_order))
    max_order = min(max_order, n)

    # If the requested number of coalitions covers all subsets up to max_order,
    # enumerate deterministically instead of sampling.
    total = sum(math.comb(n, k) for k in range(1, max_order + 1))
    if int(n_coalitions) >= int(total):
        from itertools import combinations  # noqa: PLC0415

        coalitions: List[List[str]] = []
        for k in range(1, max_order + 1):
            for combo in combinations(players, k):
                coalitions.append(list(combo))
        return coalitions

    rng = np.random.default_rng(int(seed))
    unique: set[str] = set()
    coalitions: List[List[str]] = []

    # Always include all singletons.
    for p in players:
        k = coalition_key([p])
        if k not in unique:
            unique.add(k)
            coalitions.append([p])

    target = max(int(n_coalitions), len(coalitions))

    attempts = 0
    while len(coalitions) < target and attempts < target * 50:
        attempts += 1
        size = int(rng.integers(1, max_order + 1))
        subset = rng.choice(players, size=size, replace=False).tolist()
        k = coalition_key(subset)
        if k in unique:
            continue
        unique.add(k)
        coalitions.append(sorted(subset))

    return coalitions[:target]


def _score_predictions(y_true: pd.Series, y_pred: np.ndarray, *, metric: str) -> float:
    metric = str(metric)
    if metric == "accuracy":
        from sklearn.metrics import accuracy_score  # noqa: PLC0415

        return float(accuracy_score(y_true.astype("object"), pd.Series(y_pred, index=y_true.index).astype("object")))
    if metric == "mae":
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415

        return float(mean_absolute_error(y_true.astype(float), np.asarray(y_pred, dtype=float)))
    if metric == "neg_mae":
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415

        return -float(mean_absolute_error(y_true.astype(float), np.asarray(y_pred, dtype=float)))
    if metric == "inv_mae":
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415

        mae = float(mean_absolute_error(y_true.astype(float), np.asarray(y_pred, dtype=float)))
        return 1.0 / (1.0 + mae)
    raise ValueError(f"Unsupported metric: {metric}")


def _train_and_score(
    X: pd.DataFrame,
    y: pd.Series,
    coalition: Sequence[str],
    *,
    settings: GameTableSettings,
) -> Dict[str, Any]:
    cols = [c for c in coalition if c in X.columns]
    if not cols:
        return {
            "coalition_key": coalition_key(coalition),
            "order": int(len(coalition)),
            "value": float("nan"),
            "abs_value": float("nan"),
            "value_std": float("nan"),
            "metric": str(settings.metric),
            "n_train": 0,
            "n_test": 0,
            "n_runs": 0,
        }

    Xc = X[cols].copy()
    y_clean = y.copy()

    stratify = None
    is_regression = pd.api.types.is_numeric_dtype(y_clean)
    if is_regression:
        if str(settings.metric) not in {"neg_mae", "inv_mae", "mae"}:
            raise ValueError("Regression game table currently supports metric in {neg_mae, inv_mae, mae} only.")
    else:
        if settings.metric != "accuracy":
            raise ValueError("Classification game table currently supports metric=accuracy only.")
        if y_clean.nunique(dropna=True) > 1:
            stratify = y_clean

    # Drop rows with missing targets (should already be handled upstream, but keep it safe).
    mask = y_clean.notna()
    Xc = Xc.loc[mask].copy()
    y_clean = y_clean.loc[mask].copy()

    if len(y_clean) < 5:
        raise ValueError("Not enough labeled rows to build a game table.")

    from bci_osxai.models.train import train_xgb_classifier, train_xgb_regressor  # noqa: PLC0415

    def train_and_score_once(*, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        if is_regression:
            model = train_xgb_regressor(X_train, y_train)
            y_pred = model.predict(X_test)
            metric = str(settings.metric)
        else:
            model = train_xgb_classifier(X_train, y_train)
            y_pred = model.predict(X_test)
            metric = "accuracy"
        return _score_predictions(y_test, y_pred, metric=metric)

    scores: list[float] = []
    n_train = 0
    n_test = 0
    cv_folds = int(getattr(settings, "cv_folds", 0) or 0)
    n_repeats = max(1, int(getattr(settings, "n_repeats", 1) or 1))

    if stratify is not None:
        stratify = stratify.loc[mask]

    if cv_folds >= 2:
        from sklearn.model_selection import KFold, StratifiedKFold  # noqa: PLC0415

        for rep in range(n_repeats):
            rs = int(settings.seed) + int(rep)
            if (not is_regression) and (stratify is not None) and (stratify.nunique(dropna=True) > 1):
                splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
                split_iter = splitter.split(Xc, stratify)
            else:
                splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=rs)
                split_iter = splitter.split(Xc)

            for train_idx, test_idx in split_iter:
                X_train = Xc.iloc[train_idx].copy()
                X_test = Xc.iloc[test_idx].copy()
                y_train = y_clean.iloc[train_idx].copy()
                y_test = y_clean.iloc[test_idx].copy()
                if n_train == 0:
                    n_train = int(len(X_train))
                    n_test = int(len(X_test))
                scores.append(float(train_and_score_once(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)))
    else:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        for rep in range(n_repeats):
            rs = int(settings.seed) + int(rep)
            X_train, X_test, y_train, y_test = train_test_split(
                Xc,
                y_clean,
                test_size=float(settings.test_size),
                random_state=rs,
                stratify=stratify if (not is_regression and stratify is not None and y_clean.nunique(dropna=True) > 1) else None,
            )
            if n_train == 0:
                n_train = int(len(X_train))
                n_test = int(len(X_test))
            scores.append(float(train_and_score_once(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)))

    value = float(np.mean(scores)) if scores else float("nan")
    value_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return {
        "coalition": list(cols),
        "order": int(len(cols)),
        "value": float(value),
        "abs_value": float(abs(value)),
        "value_std": float(value_std),
        "metric": str(settings.metric) if is_regression else "accuracy",
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_runs": int(len(scores)),
    }


def compute_empty_baseline(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    settings: GameTableSettings,
) -> Dict[str, Any]:
    """Compute v(∅): uninformative predictor score using labels only.

    Classification: predict the majority class in y_train (ties broken deterministically).
    Regression: predict the median of y_train (optimal for MAE-based metrics).
    """
    y_clean = y.copy()
    mask = y_clean.notna()
    Xc = X.loc[mask].copy()
    y_clean = y_clean.loc[mask].copy()

    if len(y_clean) < 5:
        raise ValueError("Not enough labeled rows to compute an empty baseline.")

    stratify = None
    is_regression = pd.api.types.is_numeric_dtype(y_clean)
    if is_regression:
        if str(settings.metric) not in {"neg_mae", "inv_mae", "mae"}:
            raise ValueError("Regression baseline currently supports metric in {neg_mae, inv_mae, mae} only.")
    else:
        if settings.metric != "accuracy":
            raise ValueError("Classification baseline currently supports metric=accuracy only.")
        if y_clean.nunique(dropna=True) > 1:
            stratify = y_clean

    cv_folds = int(getattr(settings, "cv_folds", 0) or 0)
    n_repeats = max(1, int(getattr(settings, "n_repeats", 1) or 1))

    def baseline_score_once(*, y_train: pd.Series, y_test: pd.Series) -> float:
        if is_regression:
            pred = float(pd.to_numeric(y_train, errors="coerce").median())
            y_pred = np.full(shape=(len(y_test),), fill_value=pred, dtype=float)
            metric = str(settings.metric)
        else:
            pred = _mode_with_tiebreak(y_train)
            y_pred = np.asarray([pred] * len(y_test), dtype=object)
            metric = "accuracy"
        return _score_predictions(y_test, y_pred, metric=metric)

    scores: list[float] = []
    n_train = 0
    n_test = 0
    if cv_folds >= 2:
        from sklearn.model_selection import KFold, StratifiedKFold  # noqa: PLC0415

        for rep in range(n_repeats):
            rs = int(settings.seed) + int(rep)
            if (not is_regression) and (stratify is not None) and (stratify.nunique(dropna=True) > 1):
                splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=rs)
                split_iter = splitter.split(Xc, stratify)
            else:
                splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=rs)
                split_iter = splitter.split(Xc)

            for train_idx, test_idx in split_iter:
                y_train = y_clean.iloc[train_idx].copy()
                y_test = y_clean.iloc[test_idx].copy()
                if n_train == 0:
                    n_train = int(len(train_idx))
                    n_test = int(len(test_idx))
                scores.append(float(baseline_score_once(y_train=y_train, y_test=y_test)))
    else:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        for rep in range(n_repeats):
            rs = int(settings.seed) + int(rep)
            _, _, y_train, y_test = train_test_split(
                Xc,
                y_clean,
                test_size=float(settings.test_size),
                random_state=rs,
                stratify=stratify,
            )
            if n_train == 0:
                n_train = int(len(y_train))
                n_test = int(len(y_test))
            scores.append(float(baseline_score_once(y_train=y_train, y_test=y_test)))

    metric = str(settings.metric) if is_regression else "accuracy"
    value = float(np.mean(scores)) if scores else float("nan")
    value_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return {
        "coalition": [],
        "order": 0,
        "value": float(value),
        "abs_value": float(abs(value)),
        "value_std": float(value_std),
        "metric": str(metric),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_runs": int(len(scores)),
        "seed": int(settings.seed),
    }


def build_empty_baseline_table(*, players: Sequence[str], baseline: Dict[str, Any]) -> pd.DataFrame:
    """Build a one-row wide-format table matching game_table output schema (order=0)."""
    players = [str(p) for p in players]
    out = pd.DataFrame([{p: 0 for p in players}], dtype=int)
    out["order"] = int(baseline["order"])
    out["value"] = float(baseline["value"])
    out["abs_value"] = float(baseline["abs_value"])
    if "value_std" in baseline:
        out["value_std"] = float(baseline["value_std"])
    out["metric"] = str(baseline["metric"])
    out["n_train"] = int(baseline["n_train"])
    out["n_test"] = int(baseline["n_test"])
    if "n_runs" in baseline:
        out["n_runs"] = int(baseline["n_runs"])
    out["seed"] = int(baseline.get("seed", 42))
    return out


def build_game_table(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    players: Sequence[str] | None = None,
    settings: GameTableSettings,
    progress: bool = False,
) -> pd.DataFrame:
    if not settings.enabled:
        raise ValueError("game_table is disabled")
    if players is None:
        players = list(X.columns)

    coalitions = _generate_coalitions(players, max_order=settings.max_order, n_coalitions=settings.n_coalitions, seed=settings.seed)
    rows: List[Dict[str, Any]] = []
    for coalition in tqdm_wrap(coalitions, desc="build-game-table", enabled=bool(progress)):
        rows.append(_train_and_score(X, y, coalition, settings=settings))
    base = pd.DataFrame(rows)
    player_cols = [str(p) for p in players]
    out = pd.DataFrame(0, index=base.index, columns=player_cols, dtype=int)
    for i, coalition in enumerate(base["coalition"].tolist()):
        for p in coalition:
            if p in out.columns:
                out.at[i, p] = 1
    out["order"] = base["order"].astype(int)
    out["value"] = base["value"].astype(float)
    out["abs_value"] = base["abs_value"].astype(float)
    if "value_std" in base.columns:
        out["value_std"] = base["value_std"].astype(float)
    out["metric"] = base["metric"].astype(str)
    out["n_train"] = base["n_train"].astype(int)
    out["n_test"] = base["n_test"].astype(int)
    if "n_runs" in base.columns:
        out["n_runs"] = base["n_runs"].astype(int)
    out["seed"] = int(settings.seed)
    return out


def load_game_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Game table format is CSV only.")
    return pd.read_csv(path)


def save_game_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
