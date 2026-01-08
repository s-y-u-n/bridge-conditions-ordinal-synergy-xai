from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple


def generate_candidate_sets(features: Sequence[str], max_size: int = 3) -> List[Tuple[str, ...]]:
    candidates: List[Tuple[str, ...]] = []
    for size in range(2, max_size + 1):
        candidates.extend(combinations(features, size))
    return candidates
