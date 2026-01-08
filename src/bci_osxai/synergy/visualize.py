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
    from bci_osxai.synergy.shapiq_explain import _ensure_writable_caches  # noqa: PLC0415

    _ensure_writable_caches()

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


def plot_group_lexcel_bar(
    *,
    structure_id: str,
    predicted_label: str,
    group_lexcel: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    from bci_osxai.synergy.shapiq_explain import _ensure_writable_caches  # noqa: PLC0415

    _ensure_writable_caches()

    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranking = group_lexcel.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        labels = ["(no ranking)"]
        values = [0.0]
    else:
        labels = []
        values = []
        for item in ranking[:20]:
            feature_set = item.get("set", [])
            if isinstance(feature_set, list):
                labels.append(" × ".join(str(x) for x in feature_set))
            else:
                labels.append(str(feature_set))
            values.append(float(item.get("abs_value") or 0.0))

    fig_h = max(3.0, 0.45 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(labels)), values, color="#9467bd")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"Score used for preorder ({group_lexcel.get('score_key', 'abs_value')})")
    ax.set_title(f"{structure_id}  predicted={predicted_label}  Group lex-cel (top)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_group_ordinal_banzhaf_bar(
    *,
    structure_id: str,
    predicted_label: str,
    banzhaf: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    from bci_osxai.synergy.shapiq_explain import _ensure_writable_caches  # noqa: PLC0415

    _ensure_writable_caches()

    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranking = banzhaf.get("ranking", [])
    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    if isinstance(ranking, list) and ranking:
        for item in ranking[:20]:
            feature_set = item.get("set", [])
            if isinstance(feature_set, list):
                labels.append(" × ".join(str(x) for x in feature_set))
            else:
                labels.append(str(feature_set))
            score = float(item.get("score", 0.0))
            values.append(score)
            colors.append("#2ca02c" if score >= 0 else "#d62728")
    else:
        labels = ["(no ranking)"]
        values = [0.0]
        colors = ["#777777"]

    fig_h = max(3.0, 0.45 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#444444", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Group Ordinal Banzhaf score (u_plus - u_minus)")
    ax.set_title(f"{structure_id}  predicted={predicted_label}  Group Ordinal Banzhaf (top)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_borda_shapley_bar(
    *,
    structure_id: str,
    predicted_label: str,
    borda_shapley: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    from bci_osxai.synergy.shapiq_explain import _ensure_writable_caches  # noqa: PLC0415

    _ensure_writable_caches()

    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = borda_shapley.get("results", [])
    labels: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    if isinstance(results, list) and results:
        for item in results[:20]:
            feature_set = item.get("set", [])
            if isinstance(feature_set, list):
                labels.append(" × ".join(str(x) for x in feature_set))
            else:
                labels.append(str(feature_set))
            v = float(item.get("value", 0.0))
            values.append(v)
            colors.append("#2ca02c" if v >= 0 else "#d62728")
    else:
        labels = ["(no results)"]
        values = [0.0]
        colors = ["#777777"]

    fig_h = max(3.0, 0.45 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axvline(0.0, color="#444444", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Borda-based Shapley-type interaction index (I^B)")
    ax.set_title(f"{structure_id}  predicted={predicted_label}  Borda-Shapley (top)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_group_lexcel_table(
    *,
    structure_id: str,
    predicted_label: str,
    group_lexcel: Dict[str, Any],
    output_path: str | Path,
    max_rows: int = 20,
) -> Path:
    from bci_osxai.synergy.shapiq_explain import _ensure_writable_caches  # noqa: PLC0415

    _ensure_writable_caches()

    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ranking = group_lexcel.get("ranking", [])
    rows: List[List[str]] = []
    if isinstance(ranking, list) and ranking:
        for i, item in enumerate(ranking[:max_rows], start=1):
            feature_set = item.get("set", [])
            if isinstance(feature_set, list):
                set_text = " × ".join(str(x) for x in feature_set)
            else:
                set_text = str(feature_set)

            theta_head = item.get("theta_head", [])
            theta_head_text = "[" + ",".join(str(int(v)) for v in theta_head) + "]" if isinstance(theta_head, list) else str(theta_head)

            theta_nonzero = item.get("theta_nonzero", [])
            if isinstance(theta_nonzero, list):
                theta_nz_text = " ".join(f"{int(k)}:{int(v)}" for k, v in theta_nonzero[:10])
                if len(theta_nonzero) > 10:
                    theta_nz_text += " ..."
            else:
                theta_nz_text = str(theta_nonzero)

            rows.append([str(i), set_text, theta_head_text, theta_nz_text])
    else:
        rows.append(["-", "(no ranking)", "[]", ""])

    col_labels = ["rank", "set", "theta_head", "theta_nonzero (first 10)"]

    fig_h = max(3.0, 0.45 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    ax.set_title(f"{structure_id}  predicted={predicted_label}  Group lex-cel (Θ(T) lex order)")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    # Widen columns a bit for readability.
    try:
        table.auto_set_column_width(col=list(range(len(col_labels))))
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
