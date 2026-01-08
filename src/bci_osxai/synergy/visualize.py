from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence


def plot_interactions_bar(
    *,
    structure_id: str,
    predicted_label: str,
    interactions: Sequence[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    from bci_osxai.utils.caches import ensure_writable_caches  # noqa: PLC0415

    ensure_writable_caches()

    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: List[str] = []
    values: List[float] = []
    for item in interactions:
        feature_set = item.get("set", [])
        if isinstance(feature_set, list):
            labels.append(" × ".join(str(x) for x in feature_set))
        else:
            labels.append(str(feature_set))
        values.append(float(item.get("value", 0.0)))

    if not labels:
        labels = ["(no interactions)"]
        values = [0.0]

    fig_h = max(3.0, 0.45 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#444444", linewidth=1)
    ax.set_xlabel("Interaction value (expected ordinal index shift)")
    ax.set_title(f"{structure_id}  predicted={predicted_label}")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path
