from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from bci_osxai.utils.progress import tqdm_wrap

Coalition = Tuple[str, ...]


@dataclass(frozen=True)
class GroupOrdinalBanzhafSettings:
    score_key: str = "abs_value"
    tie_tol: float = 1e-12
    max_items: int = 20
    fixed_order: int | None = 2
    progress: bool = False


def _to_coalition(item: Dict[str, Any]) -> Coalition:
    raw = item.get("set", [])
    if isinstance(raw, (list, tuple)):
        return tuple(sorted(str(x) for x in raw))
    return (str(raw),)


def _build_layers(
    scored: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    tie_tol: float,
) -> tuple[list[list[Coalition]], dict[Coalition, int]]:
    items: list[tuple[Coalition, float]] = []
    for item in scored:
        c = _to_coalition(item)
        if not c:
            continue
        items.append((c, float(item.get(score_key, 0.0))))

    items.sort(key=lambda t: t[1], reverse=True)
    if not items:
        return [], {}

    layers: list[list[Coalition]] = []
    layer_index: dict[Coalition, int] = {}
    current: list[Coalition] = [items[0][0]]
    current_score = items[0][1]

    for c, score in items[1:]:
        if abs(score - current_score) <= tie_tol:
            current.append(c)
        else:
            layers.append(current)
            current = [c]
            current_score = score
    layers.append(current)

    for idx, layer in enumerate(layers):
        for c in layer:
            if c not in layer_index:
                layer_index[c] = idx

    return layers, layer_index


def group_ordinal_banzhaf(
    scored: Sequence[Dict[str, Any]],
    *,
    settings: GroupOrdinalBanzhafSettings = GroupOrdinalBanzhafSettings(),
) -> Dict[str, Any]:
    """
    Compute Group Ordinal Banzhaf scores from a coalition preorder over the given universe.

    Practical note:
    - We only evaluate m_T^S when both S and S∪T exist in the provided `scored` universe.
    - The preorder is induced by sorting by `settings.score_key` (descending) with ties
      within `settings.tie_tol` as equivalence classes.
    """

    layers, layer_index = _build_layers(scored, score_key=settings.score_key, tie_tol=settings.tie_tol)
    if not layer_index:
        return {"score_key": settings.score_key, "tie_tol": settings.tie_tol, "n_layers": 0, "ranking": []}

    # unique coalitions in input order
    uniq: list[Coalition] = []
    seen = set()
    score_lookup: dict[Coalition, Dict[str, Any]] = {}
    for item in scored:
        c = _to_coalition(item)
        if not c:
            continue
        if c not in seen:
            seen.add(c)
            uniq.append(c)
            score_lookup[c] = item

    universe = set(uniq)

    results: list[dict[str, Any]] = []
    for T in tqdm_wrap(uniq, desc="group-ordinal-banzhaf", enabled=settings.progress):
        if settings.fixed_order is not None and len(T) != int(settings.fixed_order):
            continue
        plus = 0
        minus = 0
        comparable = 0
        Tset = set(T)

        for S in uniq:
            Sset = set(S)
            if Tset & Sset:
                continue
            U = tuple(sorted(Tset | Sset))
            if U not in universe or S not in universe:
                continue
            # Compare U vs S using layer order (lower layer index = better).
            rU = layer_index.get(U)
            rS = layer_index.get(S)
            if rU is None or rS is None:
                continue
            comparable += 1
            if rU < rS:
                plus += 1
            elif rS < rU:
                minus += 1

        score = plus - minus
        base = score_lookup.get(T, {})
        results.append(
            {
                "set": list(T),
                "order": len(T),
                "u_plus": plus,
                "u_minus": minus,
                "score": score,
                "comparisons": comparable,
                "abs_value": base.get("abs_value"),
                "value": base.get("value"),
            }
        )

    results.sort(key=lambda d: (d["score"], float(d.get("abs_value") or 0.0)), reverse=True)
    return {
        "score_key": settings.score_key,
        "tie_tol": settings.tie_tol,
        "n_layers": len(layers),
        "ranking": results[: settings.max_items],
    }
