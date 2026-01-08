from __future__ import annotations

from typing import Literal


def compare_scores(score_a: float, score_b: float, tie_tol: float = 1e-6) -> Literal[">", "=", "<"]:
    if abs(score_a - score_b) <= tie_tol:
        return "="
    return ">" if score_a > score_b else "<"
