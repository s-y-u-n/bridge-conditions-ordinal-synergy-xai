from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class LexcelSettings:
    enabled: bool = True
    score_key: str = "abs_value"  # "abs_value" or "value"
    tie_tol: float = 1.0e-12
    min_order: int = 1
    max_order: int | None = None
    max_items: int = 20
    theta_head: int = 12
    theta_nonzero_max: int = 40


def _iter_players_in_coalition_key(coalition_key: str) -> List[str]:
    return [p for p in str(coalition_key).split("|") if p]


def lexcel_ranking(
    interactions_table: pd.DataFrame,
    *,
    players: Sequence[str],
    settings: LexcelSettings,
) -> Dict[str, Any]:
    if not settings.enabled:
        return {"enabled": False}

    if interactions_table is None or interactions_table.empty:
        return {
            "enabled": True,
            "score_key": settings.score_key,
            "tie_tol": float(settings.tie_tol),
            "min_order": int(settings.min_order),
            "max_order": int(settings.max_order) if settings.max_order is not None else None,
            "n_ranks": 0,
            "ranking": [],
        }

    required = {"coalition_key", "order", settings.score_key}
    missing = [c for c in required if c not in interactions_table.columns]
    if missing:
        raise ValueError(f"Lex-cel requires columns {sorted(required)}; missing={missing}")

    df = interactions_table.copy()
    df = df[df["order"].astype(int) >= int(settings.min_order)]
    if settings.max_order is not None:
        df = df[df["order"].astype(int) <= int(settings.max_order)]
    df = df[df["coalition_key"].notna()].copy()
    if df.empty:
        return {
            "enabled": True,
            "score_key": settings.score_key,
            "tie_tol": float(settings.tie_tol),
            "min_order": int(settings.min_order),
            "max_order": int(settings.max_order) if settings.max_order is not None else None,
            "n_ranks": 0,
            "ranking": [],
        }

    df["_score"] = pd.to_numeric(df[settings.score_key], errors="coerce")
    df = df[df["_score"].notna()].copy()
    df = df.sort_values("_score", ascending=False, kind="mergesort").reset_index(drop=True)

    # Build score-tie equivalence classes Σ_k by scanning sorted scores.
    rank_ids: List[int] = []
    current_rank = -1
    prev_score: float | None = None
    for v in df["_score"].tolist():
        fv = float(v)
        if prev_score is None or abs(fv - prev_score) > float(settings.tie_tol):
            current_rank += 1
            prev_score = fv
        rank_ids.append(current_rank)
    df["_rank_id"] = rank_ids

    n_ranks = int(current_rank + 1)
    player_set = {str(p) for p in players}
    theta_by_player: Dict[str, List[int]] = {p: [0] * n_ranks for p in player_set}

    for coalition_key, rank_id in zip(df["coalition_key"].tolist(), df["_rank_id"].tolist()):
        k = int(rank_id)
        for p in _iter_players_in_coalition_key(coalition_key):
            if p in theta_by_player:
                theta_by_player[p][k] += 1

    def theta_key(p: str) -> tuple:
        return tuple(theta_by_player.get(p, [0] * n_ranks))

    ranked_players = sorted(player_set, key=theta_key, reverse=True)

    def summarize_theta(theta: Iterable[int]) -> tuple[list[int], list[list[int]]]:
        theta_list = list(theta)
        head = theta_list[: int(settings.theta_head)]
        nonzero: list[list[int]] = []
        for idx, c in enumerate(theta_list, start=1):
            if c:
                nonzero.append([int(idx), int(c)])
                if len(nonzero) >= int(settings.theta_nonzero_max):
                    break
        return head, nonzero

    ranking: List[Dict[str, Any]] = []
    for p in ranked_players[: int(settings.max_items)]:
        head, nonzero = summarize_theta(theta_by_player.get(p, [0] * n_ranks))
        ranking.append(
            {
                "player": p,
                "theta_head": head,
                "theta_nonzero": nonzero,
            }
        )

    return {
        "enabled": True,
        "score_key": settings.score_key,
        "tie_tol": float(settings.tie_tol),
        "min_order": int(settings.min_order),
        "max_order": int(settings.max_order) if settings.max_order is not None else None,
        "n_ranks": int(n_ranks),
        "ranking": ranking,
    }

