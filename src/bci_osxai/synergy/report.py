from __future__ import annotations

from typing import Dict, List, Sequence


def build_synergy_report(
    structure_id: str,
    predicted_label: str,
    top_sets: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    return {
        "structure_id": structure_id,
        "predicted_label": predicted_label,
        "synergy_top_k": list(top_sets),
    }
