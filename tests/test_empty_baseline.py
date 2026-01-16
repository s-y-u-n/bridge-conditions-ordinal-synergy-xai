import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from bci_osxai.game_table import GameTableSettings, build_empty_baseline_table, compute_empty_baseline


def test_empty_baseline_classification_matches_majority_vote() -> None:
    X = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5, 6, 7], "b": [1, 1, 1, 1, 0, 0, 0, 0]})
    y = pd.Series(["A", "A", "A", "B", "B", "B", "B", "B"])
    settings = GameTableSettings(test_size=0.25, metric="accuracy", seed=42)

    baseline = compute_empty_baseline(X=X, y=y, settings=settings)

    _, _, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.seed,
        stratify=y,
    )
    majority = y_train.value_counts().idxmax()
    expected = float((y_test == majority).mean())
    assert baseline["order"] == 0
    assert baseline["metric"] == "accuracy"
    assert abs(baseline["value"] - expected) < 1e-12


def test_empty_baseline_regression_inv_mae_uses_train_median() -> None:
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7]})
    y = pd.Series([0.0, 0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 10.0])
    settings = GameTableSettings(test_size=0.25, metric="inv_mae", seed=0)

    baseline = compute_empty_baseline(X=X, y=y, settings=settings)

    _, _, y_train, y_test = train_test_split(X, y, test_size=settings.test_size, random_state=settings.seed)
    pred = float(y_train.median())
    mae = float((y_test - pred).abs().mean())
    expected = 1.0 / (1.0 + mae)
    assert baseline["order"] == 0
    assert baseline["metric"] == "inv_mae"
    assert abs(baseline["value"] - expected) < 1e-12


def test_empty_baseline_regression_mae_uses_train_median() -> None:
    X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7]})
    y = pd.Series([0.0, 0.0, 0.0, 1.0, 10.0, 10.0, 10.0, 10.0])
    settings = GameTableSettings(test_size=0.25, metric="mae", seed=0)

    baseline = compute_empty_baseline(X=X, y=y, settings=settings)

    _, _, y_train, y_test = train_test_split(X, y, test_size=settings.test_size, random_state=settings.seed)
    pred = float(y_train.median())
    expected = float((y_test - pred).abs().mean())
    assert baseline["order"] == 0
    assert baseline["metric"] == "mae"
    assert abs(baseline["value"] - expected) < 1e-12


def test_build_empty_baseline_table_schema() -> None:
    players = ["f1", "f2", "f3"]
    baseline = {"order": 0, "value": 0.5, "abs_value": 0.5, "metric": "accuracy", "n_train": 10, "n_test": 5, "seed": 42}
    df = build_empty_baseline_table(players=players, baseline=baseline)
    assert df.shape == (1, len(players) + 7)
    assert df["order"].iloc[0] == 0
    for p in players:
        assert int(df[p].iloc[0]) == 0


def test_empty_baseline_repeated_holdout_averages_over_seeds() -> None:
    def mode_with_tiebreak(values: pd.Series) -> str:
        vc = values.astype("object").value_counts(dropna=True)
        max_count = int(vc.max())
        candidates = sorted([str(x) for x in vc[vc == max_count].index.tolist()])
        return candidates[0]

    X = pd.DataFrame({"x": list(range(30))})
    y = pd.Series((["A"] * 14) + (["B"] * 10) + (["C"] * 6))
    settings = GameTableSettings(test_size=0.2, metric="accuracy", seed=0, n_repeats=3, cv_folds=0)

    baseline = compute_empty_baseline(X=X, y=y, settings=settings)

    scores: list[float] = []
    for rep in range(settings.n_repeats):
        _, _, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.test_size,
            random_state=settings.seed + rep,
            stratify=y,
        )
        majority = mode_with_tiebreak(y_train)
        scores.append(float((y_test == majority).mean()))

    expected_mean = float(np.mean(scores))
    expected_std = float(np.std(scores, ddof=1))
    assert baseline["n_runs"] == settings.n_repeats
    assert abs(baseline["value"] - expected_mean) < 1e-12
    assert abs(baseline["value_std"] - expected_std) < 1e-12


def test_empty_baseline_cv_averages_over_folds_and_repeats() -> None:
    X = pd.DataFrame({"x": list(range(12))})
    y = pd.Series([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0])
    settings = GameTableSettings(test_size=0.25, metric="mae", seed=42, n_repeats=2, cv_folds=3)

    baseline = compute_empty_baseline(X=X, y=y, settings=settings)

    scores: list[float] = []
    for rep in range(settings.n_repeats):
        splitter = KFold(n_splits=settings.cv_folds, shuffle=True, random_state=settings.seed + rep)
        for train_idx, test_idx in splitter.split(X):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            pred = float(y_train.median())
            scores.append(float((y_test - pred).abs().mean()))

    expected_mean = float(np.mean(scores))
    expected_std = float(np.std(scores, ddof=1))
    assert baseline["metric"] == "mae"
    assert baseline["n_runs"] == settings.n_repeats * settings.cv_folds
    assert abs(baseline["value"] - expected_mean) < 1e-12
    assert abs(baseline["value_std"] - expected_std) < 1e-12
