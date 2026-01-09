from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

_META_COLUMNS = {
    "coalition",
    "coalition_key",
    "order",
    "value",
    "abs_value",
    "metric",
    "n_train",
    "n_test",
    "seed",
}


def _is_binary_indicator_column(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique().tolist()
    if not vals:
        return False
    allowed = {0.0, 1.0}
    return all(float(v) in allowed for v in vals)


def infer_players_from_table(table: pd.DataFrame, *, meta_columns: Iterable[str] = _META_COLUMNS) -> list[str]:
    meta = {str(c) for c in meta_columns}
    if "coalition_key" in table.columns and table["coalition_key"].notna().any():
        players: set[str] = set()
        for key in table["coalition_key"].dropna().astype(str).tolist():
            for p in key.split("|"):
                if p:
                    players.add(str(p))
        return sorted(players)

    candidates = [c for c in table.columns if str(c) not in meta]
    players = [str(c) for c in candidates if _is_binary_indicator_column(table[c])]
    if not players:
        raise ValueError("Could not infer players from game table (no coalition_key and no binary indicator columns).")
    return players


def _mask_from_row_indicators(row: Mapping[str, object], players: Sequence[str]) -> int:
    mask = 0
    for idx, p in enumerate(players):
        try:
            if int(row.get(p, 0) or 0) == 1:
                mask |= 1 << idx
        except Exception:
            continue
    return int(mask)


def _mask_from_coalition_key(coalition_key: str, players_to_index: Mapping[str, int]) -> int:
    mask = 0
    for p in str(coalition_key).split("|"):
        if not p:
            continue
        idx = players_to_index.get(str(p))
        if idx is None:
            continue
        mask |= 1 << int(idx)
    return int(mask)


@dataclass(frozen=True)
class CooperativeGame:
    players: list[str]
    values_by_mask: np.ndarray  # shape=(2^n,)
    v_empty: float
    v_full: float

    @property
    def n_players(self) -> int:
        return int(len(self.players))


def game_from_table(
    table: pd.DataFrame,
    *,
    players: Sequence[str] | None = None,
    value_col: str = "value",
    v_empty: float | None = 0.0,
    strict: bool = True,
) -> CooperativeGame:
    if table is None or table.empty:
        raise ValueError("Game table is empty.")

    if players is None:
        players = infer_players_from_table(table)
    players = [str(p) for p in players]
    n = len(players)
    if n < 1:
        raise ValueError("At least one player is required.")

    if value_col not in table.columns:
        raise ValueError(f"Missing value column: {value_col}")

    size = 1 << n
    values = np.full(size, np.nan, dtype=float)

    if "coalition_key" in table.columns and table["coalition_key"].notna().any():
        p2i = {p: i for i, p in enumerate(players)}
        for coalition_key, v in zip(table["coalition_key"].tolist(), table[value_col].tolist()):
            if coalition_key is None or (isinstance(coalition_key, float) and np.isnan(coalition_key)):
                continue
            mask = _mask_from_coalition_key(str(coalition_key), p2i)
            fv = float(pd.to_numeric(v, errors="coerce"))
            if np.isnan(fv):
                continue
            if not np.isnan(values[mask]) and abs(values[mask] - fv) > 0:
                raise ValueError(f"Duplicate coalition with conflicting values: coalition_key={coalition_key!r}")
            values[mask] = fv
    else:
        for _, row in table.iterrows():
            mask = _mask_from_row_indicators(row, players)
            fv = float(pd.to_numeric(row.get(value_col), errors="coerce"))
            if np.isnan(fv):
                continue
            if not np.isnan(values[mask]) and abs(values[mask] - fv) > 0:
                raise ValueError(f"Duplicate coalition with conflicting values: mask={mask}")
            values[mask] = fv

    if np.isnan(values[0]):
        if v_empty is None:
            raise ValueError("v(empty) is missing from the game table; pass v_empty explicitly.")
        values[0] = float(v_empty)
    v0 = float(values[0])

    full_mask = (1 << n) - 1
    if np.isnan(values[full_mask]):
        raise ValueError("v(N) is missing from the game table (full coalition).")
    vN = float(values[full_mask])

    if strict:
        missing = np.isnan(values).sum()
        if missing:
            raise ValueError(
                f"Game table is incomplete for exact computation: missing {int(missing)} / {int(size)} coalitions."
            )

    return CooperativeGame(players=list(players), values_by_mask=values, v_empty=v0, v_full=vN)


class PowerIndex(Protocol):
    name: str

    def compute(self, game: CooperativeGame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class ShapleyValueIndex:
    name: str = "shapley"

    def compute(self, game: CooperativeGame) -> pd.DataFrame:
        n = game.n_players
        values = game.values_by_mask
        denom = float(math.factorial(n))
        weights = np.array(
            [float(math.factorial(k) * math.factorial(n - k - 1)) / denom for k in range(n)],
            dtype=float,
        )

        phi = np.zeros(n, dtype=float)
        for i in range(n):
            bit = 1 << i
            acc = 0.0
            for mask in range(1 << n):
                if mask & bit:
                    continue
                k = int(mask.bit_count())
                v_s = float(values[mask])
                v_si = float(values[mask | bit])
                acc += float(weights[k]) * (v_si - v_s)
            phi[i] = acc

        df = pd.DataFrame({"player": game.players, "index": self.name, "value": phi})
        sum_phi = float(phi.sum())
        df["n_players"] = int(n)
        df["v_empty"] = float(game.v_empty)
        df["v_full"] = float(game.v_full)
        df["sum_values"] = float(sum_phi)
        df["efficiency_gap"] = float(sum_phi - (float(game.v_full) - float(game.v_empty)))
        return df


@dataclass(frozen=True)
class BanzhafIndex:
    name: str = "banzhaf"

    def compute(self, game: CooperativeGame) -> pd.DataFrame:
        n = game.n_players
        values = game.values_by_mask
        norm = float(1 << (n - 1))

        beta = np.zeros(n, dtype=float)
        for i in range(n):
            bit = 1 << i
            acc = 0.0
            for mask in range(1 << n):
                if mask & bit:
                    continue
                acc += float(values[mask | bit] - values[mask])
            beta[i] = acc / norm

        df = pd.DataFrame({"player": game.players, "index": self.name, "value": beta})
        df["n_players"] = int(n)
        df["v_empty"] = float(game.v_empty)
        df["v_full"] = float(game.v_full)
        df["sum_values"] = float(beta.sum())
        df["efficiency_gap"] = float(beta.sum() - (float(game.v_full) - float(game.v_empty)))
        return df


def get_power_index(name: str) -> PowerIndex:
    key = str(name).strip().lower()
    if key in {"shapley", "shapley-value", "shapley_value"}:
        return ShapleyValueIndex()
    if key in {"banzhaf", "banzhaf-index", "banzhaf_index"}:
        return BanzhafIndex()
    raise ValueError(f"Unknown power index: {name}")


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    return path
