from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from bci_osxai.utils.progress import tqdm_wrap


Coalition = Tuple[str, ...]


@dataclass(frozen=True)
class GameTableSettings:
    enabled: bool = True
    n_samples: int = 10_000
    max_order: int = 6
    seed: int = 42
    cache_dir: str = "artifacts/game_tables"


def _safe_id(value: str) -> str:
    return str(value).replace("/", "_")


def _model_fingerprint(model_path: Path) -> str:
    data = model_path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _co_key(coalition: Sequence[str]) -> str:
    return "|".join(sorted(str(x) for x in coalition))


def _parse_rank_suffix(label: str) -> int | None:
    # e.g. R3 -> 3, L2 -> 2
    digits = ""
    for ch in reversed(label):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else None


def _class_weights_from_model(model: Any) -> np.ndarray:
    classes = [str(c) for c in getattr(model, "classes_", [])]
    if not classes:
        raise ValueError("Model has no classes_ for predict_proba weighting")

    suffixes = [_parse_rank_suffix(c) for c in classes]
    if all(s is not None for s in suffixes):
        # Map to increasing order of suffix
        order = np.argsort(np.asarray(suffixes, dtype=int))
        weights = np.empty(len(classes), dtype=float)
        for i, idx in enumerate(order):
            weights[idx] = float(i)
        return weights

    # Fallback: lexicographic
    order = np.argsort(np.asarray(classes, dtype=object))
    weights = np.empty(len(classes), dtype=float)
    for i, idx in enumerate(order):
        weights[idx] = float(i)
    return weights


def _compute_baseline(background_X: pd.DataFrame) -> Dict[str, object]:
    baseline: Dict[str, object] = {}
    for col in background_X.columns:
        s = background_X[col]
        if pd.api.types.is_numeric_dtype(s):
            baseline[col] = float(pd.to_numeric(s, errors="coerce").median())
        else:
            mode = s.dropna().mode()
            baseline[col] = mode.iloc[0] if not mode.empty else None
    return baseline


def _score_rows(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        weights = _class_weights_from_model(model)
        proba = model.predict_proba(X)
        return proba @ weights
    return np.asarray(model.predict(X), dtype=float)


def _sample_coalitions(
    players: Sequence[str],
    *,
    n_samples: int,
    max_order: int,
    seed: int,
    existing: set[str],
) -> List[Coalition]:
    rng = np.random.default_rng(seed)
    players = list(players)
    p = len(players)
    max_order = min(int(max_order), p)

    out: List[Coalition] = []
    attempts = 0
    while len(out) < n_samples and attempts < n_samples * 50:
        attempts += 1
        k = int(rng.integers(1, max_order + 1))
        idx = rng.choice(p, size=k, replace=False)
        coalition = tuple(sorted(players[i] for i in idx))
        key = _co_key(coalition)
        if key in existing:
            continue
        existing.add(key)
        out.append(coalition)
    return out


def load_or_build_game_table(
    *,
    model: Any,
    model_path: Path,
    structure_id: str,
    x_row: pd.DataFrame,
    background_X: pd.DataFrame,
    settings: GameTableSettings,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Create or extend a cached game table for a specific (model, structure_id, feature-set, config).

    The table stores coalition -> scalar score for the given instance x, under a baseline-masking game:
    - Keep features in coalition from x
    - Replace all other features with baseline values derived from background_X (median/mode)
    """

    if not settings.enabled:
        raise ValueError("game_table is disabled")

    model_hash = _model_fingerprint(model_path)
    cache_dir = Path(settings.cache_dir) / "next_rank"
    cache_dir.mkdir(parents=True, exist_ok=True)

    players = list(x_row.columns)
    players_key = hashlib.sha256(("|".join(players)).encode("utf-8")).hexdigest()[:12]
    file = cache_dir / f"{_safe_id(structure_id)}__{model_hash}__p{players_key}__k{int(settings.max_order)}.parquet"

    existing_df: pd.DataFrame
    if file.exists():
        existing_df = pd.read_parquet(file)
    else:
        existing_df = pd.DataFrame(columns=["coalition_key", "order", "value", "abs_value"])

    existing_keys = set(existing_df["coalition_key"].astype(str)) if not existing_df.empty else set()

    needed = max(0, int(settings.n_samples) - len(existing_keys))
    if needed == 0:
        return existing_df

    baseline = _compute_baseline(background_X)

    coalitions = _sample_coalitions(
        players,
        n_samples=needed,
        max_order=settings.max_order,
        seed=settings.seed + len(existing_keys),
        existing=existing_keys,
    )

    if not coalitions:
        return existing_df

    masked_rows: List[pd.Series] = []
    for coalition in tqdm_wrap(coalitions, desc="game-table mask", enabled=progress):
        masked = x_row.iloc[0].copy()
        keep = set(coalition)
        for col in players:
            if col in keep:
                continue
            masked[col] = baseline.get(col)
        masked_rows.append(masked)

    X_eval = pd.DataFrame(masked_rows, columns=players)
    scores = _score_rows(model, X_eval)

    new_df = pd.DataFrame(
        {
            "coalition_key": [_co_key(c) for c in coalitions],
            "order": [len(c) for c in coalitions],
            "value": scores.astype(float),
        }
    )
    new_df["abs_value"] = new_df["value"].abs()

    if existing_df.empty:
        combined = new_df
    else:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["coalition_key"], keep="first")
    combined.to_parquet(file, index=False)
    return combined


def game_table_to_scored(
    game_table: pd.DataFrame,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for _, row in game_table.iterrows():
        key = str(row["coalition_key"])
        feat = [x for x in key.split("|") if x]
        scored.append(
            {
                "set": feat,
                "order": int(row["order"]),
                "value": float(row["value"]),
                "abs_value": float(row["abs_value"]),
            }
        )
    scored.sort(key=lambda d: d["abs_value"], reverse=True)
    return scored
