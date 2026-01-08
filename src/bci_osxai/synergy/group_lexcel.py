from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from bci_osxai.utils.progress import tqdm_wrap


Coalition = Tuple[str, ...]


@dataclass(frozen=True)
class GroupLexcelSettings:
    score_key: str = "abs_value"
    tie_tol: float = 1e-12
    max_items: int = 20
    theta_head_len: int = 12
    theta_nonzero_limit: int = 30
    fixed_order: int | None = 2
    progress: bool = False


def _to_coalition(item: Dict[str, Any]) -> Coalition:
    raw = item.get("set", [])
    if isinstance(raw, (list, tuple)):
        return tuple(sorted(str(x) for x in raw))
    return (str(raw),)


def _build_equivalence_classes(
    scored: Sequence[Dict[str, Any]],
    *,
    score_key: str,
    tie_tol: float,
) -> List[List[Coalition]]:
    items = []
    for item in scored:
        coalition = _to_coalition(item)
        if not coalition:
            continue
        score = float(item.get(score_key, 0.0))
        items.append((coalition, score))

    items.sort(key=lambda t: t[1], reverse=True)
    if not items:
        return []

    classes: List[List[Coalition]] = []
    current: List[Coalition] = [items[0][0]]
    current_score = items[0][1]
    for coalition, score in items[1:]:
        if abs(score - current_score) <= tie_tol:
            current.append(coalition)
        else:
            classes.append(current)
            current = [coalition]
            current_score = score
    classes.append(current)
    return classes


def _theta(classes: Sequence[Sequence[Coalition]], coalition: Coalition) -> Tuple[int, ...]:
    tset = set(coalition)
    out: List[int] = []
    for layer in classes:
        count = 0
        for s in layer:
            if tset.issubset(set(s)):
                count += 1
        out.append(count)
    return tuple(out)


def group_lexcel_ranking(
    scored: Sequence[Dict[str, Any]],
    *,
    settings: GroupLexcelSettings = GroupLexcelSettings(),
) -> Dict[str, Any]:
    """
    Compute the Group lex-cel solution induced by a total preorder over coalitions.

    Practical note: we assume the preorder is represented by sorting `scored` by
    `settings.score_key` (descending) with ties within `settings.tie_tol`, which induces
    equivalence classes Σ1 ≻ Σ2 ≻ ... as in the definition.
    """

    classes = _build_equivalence_classes(scored, score_key=settings.score_key, tie_tol=settings.tie_tol)
    if not classes:
        return {"equivalence_classes": [], "ranking": []}

    coalitions = [_to_coalition(item) for item in scored if _to_coalition(item)]
    # unique while preserving order
    seen = set()
    uniq: List[Coalition] = []
    for c in coalitions:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    candidates = uniq
    if settings.fixed_order is not None:
        candidates = [c for c in uniq if len(c) == int(settings.fixed_order)]

    theta_map: Dict[Coalition, Tuple[int, ...]] = {}
    for c in tqdm_wrap(candidates, desc="group-lexcel theta", enabled=settings.progress):
        theta_map[c] = _theta(classes, c)
    candidates.sort(key=lambda c: theta_map[c], reverse=True)

    # Attach metadata if present in scored list.
    score_lookup: Dict[Coalition, Dict[str, Any]] = {}
    for item in scored:
        c = _to_coalition(item)
        if c and c not in score_lookup:
            score_lookup[c] = item

    ranking_items: List[Dict[str, Any]] = []
    for c in candidates[: settings.max_items]:
        base = score_lookup.get(c, {})
        theta_vec = theta_map[c]
        theta_head = list(theta_vec[: settings.theta_head_len])
        theta_nonzero = [(i + 1, v) for i, v in enumerate(theta_vec) if v != 0][: settings.theta_nonzero_limit]
        ranking_items.append(
            {
                "set": list(c),
                "theta_head": theta_head,
                "theta_nonzero": theta_nonzero,
                "order": len(c),
                "value": base.get("value"),
                "abs_value": base.get("abs_value"),
            }
        )

    eq_out = [ [list(c) for c in layer] for layer in classes[:10] ]
    return {
        "score_key": settings.score_key,
        "tie_tol": settings.tie_tol,
        "n_layers": len(classes),
        "equivalence_classes_head": eq_out,
        "ranking": ranking_items,
    }
