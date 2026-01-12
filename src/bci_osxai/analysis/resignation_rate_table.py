from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class ResignationRateSettings:
    include_empty_coalition: bool = True


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_resignation_risk_factors(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Employee_ID",
        "Age",
        "Monthly_Salary",
        "Overtime_Hours",
        "Remote_Work_Frequency",
        "Team_Size",
        "Promotions",
        "Resigned",
    }
    missing = sorted([c for c in required if c not in df.columns])
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    age = _to_num(df["Age"])
    salary = _to_num(df["Monthly_Salary"])
    overtime = _to_num(df["Overtime_Hours"])
    remote = _to_num(df["Remote_Work_Frequency"])
    team = _to_num(df["Team_Size"])
    promotions = _to_num(df["Promotions"])

    salary_median_by_age = salary.groupby(age).transform("median")
    promotions_median_by_age = promotions.groupby(age).transform("median")
    overtime_median = float(overtime.median())
    team_median = float(team.median())

    out = pd.DataFrame(index=df.index)
    out["structure_id"] = df["Employee_ID"].astype(str)
    out["Resigned"] = df["Resigned"].astype(bool)

    out["low_income"] = (salary < salary_median_by_age).fillna(False).astype(int)
    out["long_overtime"] = (overtime >= overtime_median).fillna(False).astype(int)
    out["low_remote"] = (remote == 0).fillna(False).astype(int)
    out["large_team"] = (team > team_median).fillna(False).astype(int)
    out["low_promotion"] = (promotions <= promotions_median_by_age).fillna(False).astype(int)
    return out


def _all_coalitions(players: Sequence[str], *, include_empty: bool) -> Iterable[list[str]]:
    players = [str(p) for p in players]
    start = 0 if include_empty else 1
    for k in range(start, len(players) + 1):
        for combo in combinations(players, k):
            yield list(combo)


def coalition_key(coalition: Sequence[str]) -> str:
    return "|".join(sorted(str(x) for x in coalition))


def build_resignation_rate_table(
    *,
    factors: pd.DataFrame,
    players: Sequence[str],
    settings: ResignationRateSettings = ResignationRateSettings(),
) -> pd.DataFrame:
    players = [str(p) for p in players]
    for col in ["Resigned", *players]:
        if col not in factors.columns:
            raise ValueError(f"Missing column in factors: {col}")

    rows: list[dict[str, object]] = []
    for coalition in _all_coalitions(players, include_empty=bool(settings.include_empty_coalition)):
        mask = pd.Series(True, index=factors.index)
        for p in coalition:
            mask &= factors[p].astype(int) == 1

        sub = factors.loc[mask]
        n = int(len(sub))
        n_resigned = int(sub["Resigned"].astype(bool).sum())
        rate = float(n_resigned / n) if n else float("nan")

        rows.append(
            {
                "coalition": coalition,
                "coalition_key": coalition_key(coalition),
                "order": int(len(coalition)),
                "n_rows": n,
                "n_resigned": n_resigned,
                "resignation_rate": rate,
            }
        )

    base = pd.DataFrame(rows)
    out = pd.DataFrame(0, index=base.index, columns=players, dtype=int)
    for i, coalition in enumerate(base["coalition"].tolist()):
        for p in coalition:
            out.at[i, p] = 1
    out["order"] = base["order"].astype(int)
    out["n_rows"] = base["n_rows"].astype(int)
    out["n_resigned"] = base["n_resigned"].astype(int)
    out["resignation_rate"] = base["resignation_rate"].astype(float)
    out["coalition_key"] = base["coalition_key"].astype(str)
    return out.sort_values(by=["order", "coalition_key"], ignore_index=True)

