from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_features = features.select_dtypes(include=["number"]).columns
    categorical_features = features.select_dtypes(exclude=["number"]).columns

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


@dataclass
class EncodedClassifier:
    pipeline: Pipeline
    classes_: List[str]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred_idx = self.pipeline.predict(X)
        pred_idx = np.asarray(pred_idx, dtype=int)
        return np.asarray([self.classes_[i] for i in pred_idx], dtype=object)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)


def train_xgb_classifier(features: pd.DataFrame, labels: pd.Series) -> EncodedClassifier:
    from xgboost import XGBClassifier  # noqa: PLC0415

    y = labels.astype("object").fillna(pd.NA)
    classes = sorted([str(c) for c in pd.unique(y.dropna())])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = y.map(class_to_idx).astype(int)

    preprocessor = _build_preprocessor(features)
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=0,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(features, y_idx)
    return EncodedClassifier(pipeline=pipeline, classes_=classes)


def train_xgb_regressor(features: pd.DataFrame, targets: pd.Series) -> Pipeline:
    from xgboost import XGBRegressor  # noqa: PLC0415

    preprocessor = _build_preprocessor(features)
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=0,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(features, targets.astype(float))
    return pipeline


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)

