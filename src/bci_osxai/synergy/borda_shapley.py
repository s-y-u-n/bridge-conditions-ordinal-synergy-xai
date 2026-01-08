from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from bci_osxai.utils.progress import tqdm_wrap

Coalition = Tuple[str, ...]


@dataclass(frozen=True)
class BordaShapleySettings:
    n_players: int = 10
    max_order: int = 2
    domain_max_size: int = 2
    score_key: str = "abs_value"
    tie_tol: float = 1e-12
    missing_score: float = 0.0
    top_k: int = 20
    progress: bool = False


def _to_coalition(item: Dict[str, Any]) -> Coalition:
    raw = item.get("set", [])
    if isinstance(raw, (list, tuple)):
        return tuple(sorted(str(x) for x in raw))
    return (str(raw),)


def _pick_players(scored: Sequence[Dict[str, Any]], n_players: int) -> List[str]:
    singles: List[Tuple[str, float]] = []
    for item in scored:
        c = _to_coalition(item)
        if len(c) != 1:
            continue
        singles.append((c[0], float(item.get("abs_value", 0.0))))
    singles.sort(key=lambda t: t[1], reverse=True)
    players = [name for name, _ in singles]

    if len(players) < n_players:
        # Backfill from any coalition names
        for item in scored:
            for name in _to_coalition(item):
                if name not in players:
                    players.append(name)
                if len(players) >= n_players:
                    break
            if len(players) >= n_players:
                break

    return players[:n_players]


def _coalition_mask(coalition: Coalition, player_to_idx: Dict[str, int]) -> int:
    mask = 0
    for p in coalition:
        mask |= 1 << player_to_idx[p]
    return mask


def _mask_to_coalition(mask: int, players: Sequence[str]) -> Coalition:
    return tuple(players[i] for i in range(len(players)) if (mask >> i) & 1)


def _induce_weak_order(scores: np.ndarray, tie_tol: float) -> Tuple[np.ndarray, List[List[int]]]:
    order = np.argsort(-scores)
    classes: List[List[int]] = []
    if len(order) == 0:
        return order, classes

    current = [int(order[0])]
    current_score = float(scores[order[0]])
    for idx in order[1:]:
        s = float(scores[idx])
        if abs(s - current_score) <= tie_tol:
            current.append(int(idx))
        else:
            classes.append(current)
            current = [int(idx)]
            current_score = s
    classes.append(current)
    return order, classes


def _borda_scores_from_classes(n_classes: int) -> List[int]:
    # class k (0-based from top) gets: (#below - #above) = (n_classes-1-k) - k
    return [int((n_classes - 1 - k) - k) for k in range(n_classes)]


def borda_shapley_index(
    scored: Sequence[Dict[str, Any]],
    *,
    settings: BordaShapleySettings,
) -> Dict[str, Any]:
    players = _pick_players(scored, settings.n_players)
    n = len(players)
    if n == 0:
        return {"players": [], "results": []}

    player_to_idx = {p: i for i, p in enumerate(players)}

    score_map: Dict[int, float] = {}
    for item in scored:
        c = _to_coalition(item)
        if any(p not in player_to_idx for p in c):
            continue
        if len(c) > settings.max_order:
            continue
        mask = _coalition_mask(c, player_to_idx)
        score_map[mask] = float(item.get(settings.score_key, 0.0))

    if settings.domain_max_size < settings.max_order:
        raise ValueError("domain_max_size must be >= max_order")

    all_masks = [m for m in range(1, (1 << n)) if int(m).bit_count() <= settings.domain_max_size]
    masks = np.array(all_masks, dtype=int)
    scores = np.array([score_map.get(int(m), float(settings.missing_score)) for m in masks], dtype=float)

    _, classes = _induce_weak_order(scores, settings.tie_tol)
    # Signed Borda score for a weak order:
    # For x in equivalence class Σ_k, define
    #   s(x) = (# of alternatives strictly below Σ_k) - (# strictly above Σ_k).
    # This uses alternative-counts (not just class-counts), which matches the
    # usual signed-Borda generalization to weak orders and yields both signs.
    layer_sizes = [len(layer) for layer in classes]
    prefix_counts: List[int] = []
    running = 0
    for sz in layer_sizes:
        running += int(sz)
        prefix_counts.append(running)
    total_alts = int(prefix_counts[-1]) if prefix_counts else 0

    borda_score_by_mask: Dict[int, int] = {}
    above = 0
    for k, layer in enumerate(classes):
        below = total_alts - above - len(layer)
        s_val = int(below - above)
        for idx in layer:
            borda_score_by_mask[int(masks[idx])] = s_val
        above += len(layer)

    denom_cache: Dict[int, int] = {}

    def coeff(n_: int, s_: int, t_: int) -> float:
        # (n - t - s)! t! / (n - s + 1)!
        return factorial(n_ - t_ - s_) * factorial(t_) / factorial(n_ - s_ + 1)

    results: List[Dict[str, Any]] = []
    for Smask in tqdm_wrap(masks, desc="borda-shapley", enabled=settings.progress):
        s_size = int(Smask.bit_count())
        if s_size > settings.max_order:
            continue

        remaining_mask = ((1 << n) - 1) & (~Smask)
        max_t_size = int(settings.domain_max_size - s_size)
        total = 0.0
        # enumerate T ⊆ N\\S by iterating submasks of remaining_mask
        Tmask = remaining_mask
        while True:
            t_size = int(Tmask.bit_count())
            if t_size <= max_t_size:
                w = coeff(n, s_size, t_size)
                inner = 0.0
                # enumerate L ⊆ S
                Lmask = Smask
                while True:
                    l_size = int(Lmask.bit_count())
                    sign = -1.0 if ((s_size - l_size) % 2 == 1) else 1.0
                    U = int(Lmask | Tmask)
                    if U != 0:
                        inner += sign * float(borda_score_by_mask.get(U, 0))
                    if Lmask == 0:
                        break
                    Lmask = (Lmask - 1) & Smask
                total += w * inner

            if Tmask == 0:
                break
            Tmask = (Tmask - 1) & remaining_mask

        results.append(
            {
                "set": list(_mask_to_coalition(int(Smask), players)),
                "order": s_size,
                "value": float(total),
                "abs_value": float(abs(total)),
            }
        )

    results.sort(key=lambda d: float(d["abs_value"]), reverse=True)

    positives = sorted([r for r in results if r["value"] > 0], key=lambda r: r["value"], reverse=True)
    negatives = sorted([r for r in results if r["value"] < 0], key=lambda r: r["value"])

    # To avoid outputs that look "all negative" (or "all positive") when selecting by abs_value,
    # we emit a balanced top-k when possible.
    k_pos = settings.top_k // 2
    k_neg = settings.top_k - k_pos
    balanced = positives[:k_pos] + negatives[:k_neg]
    balanced.sort(key=lambda d: float(d["abs_value"]), reverse=True)

    return {
        "players": players,
        "n_players": n,
        "max_order": settings.max_order,
        "domain_max_size": settings.domain_max_size,
        "score_key": settings.score_key,
        "tie_tol": settings.tie_tol,
        "missing_score": settings.missing_score,
        "n_classes": len(classes),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "results": balanced,
        "results_top_abs": results[: settings.top_k],
        "results_top_positive": positives[: settings.top_k],
        "results_top_negative": negatives[: settings.top_k],
    }
