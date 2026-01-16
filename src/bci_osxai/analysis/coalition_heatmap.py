from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from bci_osxai.analysis.feature_labels import feature_label_map, infer_dataset_key_from_outputs_path


_META_COLS = {
    "coalition_id",
    "coalition_key",
    "order",
    "value",
    "abs_value",
    "metric",
    "n_train",
    "n_test",
    "seed",
    "class_id",
    "k_selected",
    "class_score_max",
    "class_score_min",
    "class_size",
}


@dataclass(frozen=True)
class HeatmapSummary:
    n_rows_total: int
    n_rows_used: int
    n_features: int
    metric: str
    score_col: str
    higher_is_better: bool


def infer_feature_indicator_columns(df: pd.DataFrame, *, extra_exclude: Sequence[str] = ()) -> list[str]:
    exclude = set(_META_COLS) | {str(c) for c in extra_exclude}
    cols: list[str] = []
    for c in df.columns:
        sc = str(c)
        if sc in exclude:
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        if vals.issubset({0, 1}):
            cols.append(sc)
    return cols


def _ensure_coalition_id(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    if "coalition_id" in df.columns:
        return df
    work = df.copy()

    def row_id(row: pd.Series) -> str:
        members = [c for c in feature_cols if int(row[c]) == 1]
        return "|".join(members) if members else "EMPTY"

    work["coalition_id"] = work.apply(row_id, axis=1)
    return work


def _infer_higher_is_better(df: pd.DataFrame) -> bool:
    if "metric" not in df.columns:
        return True
    metrics = df["metric"].astype(str).dropna().unique().tolist()
    if len(metrics) != 1:
        return True
    if metrics[0] == "mae":
        return False
    return True


def plot_coalition_feature_heatmap(
    *,
    game_table_csv: str | Path,
    out_path: str | Path,
    score_col: str = "value",
    top_n: int | None = 200,
    title: str | None = None,
    ranked_csv: str | Path | None = None,
    class_gap_rows: int = 3,
) -> Dict[str, Any]:
    """Plot a coalition×feature heatmap (red if feature used).

    Rows are ordered by prediction quality:
      - higher score is better, except metric=mae where lower is better.

    If ranked_csv is provided and contains class_id, horizontal separators are drawn
    between classes (in the row order used for plotting).
    """
    game_table_csv = Path(game_table_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(game_table_csv)
    if score_col not in df.columns:
        raise ValueError(f"Missing score column {score_col!r} in {game_table_csv}")

    n_total = int(len(df))
    scores = pd.to_numeric(df[score_col], errors="coerce")
    df = df.loc[scores.notna()].copy()
    scores = pd.to_numeric(df[score_col], errors="coerce").astype(float)

    higher_is_better = _infer_higher_is_better(df)
    asc = not higher_is_better

    feature_cols = infer_feature_indicator_columns(df, extra_exclude=[score_col])
    if not feature_cols:
        raise ValueError("Could not infer 0/1 feature indicator columns for heatmap.")

    df = _ensure_coalition_id(df, feature_cols)
    # Tie-break: among equal scores, show larger coalitions first for readability.
    order_cols = [score_col]
    ascending = [asc]
    if "order" in df.columns:
        order_cols.append("order")
        ascending.append(False)
    order_cols.append("coalition_id")
    ascending.append(True)
    df = df.sort_values(by=order_cols, ascending=ascending, kind="mergesort").copy()

    if top_n is not None:
        top_n = int(top_n)
        if top_n > 0:
            df = df.head(top_n).copy()

    mat = df[feature_cols].to_numpy(dtype=int)

    ds_key = infer_dataset_key_from_outputs_path(game_table_csv)
    labels = feature_label_map(ds_key)
    x_labels = [labels.get(c, c) for c in feature_cols]

    # Optional class boundaries
    boundaries: list[int] = []
    if ranked_csv is not None:
        rdf = pd.read_csv(Path(ranked_csv))
        if "class_id" in rdf.columns:
            if "coalition_id" not in rdf.columns:
                rdf = _ensure_coalition_id(rdf, feature_cols)
            # Map coalition_id -> class_id
            m = rdf.set_index("coalition_id")["class_id"]
            class_ids = df["coalition_id"].map(m).fillna(pd.NA)
            df["_class_id_"] = class_ids
            # boundaries are indices where class changes
            prev = None
            for i, cid in enumerate(df["_class_id_"].tolist()):
                if prev is None:
                    prev = cid
                    continue
                if cid != prev:
                    boundaries.append(i)
                    prev = cid

    gap_line_positions: list[float] = []
    if boundaries and int(class_gap_rows) > 0:
        gap = int(class_gap_rows)
        expanded: list[np.ndarray] = []
        boundary_set = set(boundaries)
        row_idx = 0
        for i in range(mat.shape[0]):
            if i in boundary_set:
                for _ in range(gap):
                    expanded.append(np.full((mat.shape[1],), -1, dtype=int))
                    row_idx += 1
                # Draw a bold separator through the middle of the gap block.
                gap_line_positions.append(row_idx - gap / 2.0 - 0.5)
            expanded.append(mat[i].astype(int))
            row_idx += 1
        mat = np.vstack(expanded) if expanded else mat

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import japanize_matplotlib  # noqa: F401, PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.colors import ListedColormap  # noqa: PLC0415

    # -1: gap, 0: off, 1: on
    cmap = ListedColormap(["#E0E0E0", "white", "#E45756"])

    # Dynamic height for readability
    fig_h = max(3.0, min(18.0, 0.18 * mat.shape[0] + 2.0))
    fig_w = max(6.0, min(18.0, 0.45 * mat.shape[1] + 3.0))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)

    ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(x_labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticks([])

    ax.set_xlabel("特徴量")
    ax.set_ylabel("予測性能（上が良い順）" if higher_is_better else "予測性能（下が良い順）")

    if title is None:
        title = "特徴量セット×特徴量の使用状況（上位順）"
    ax.set_title(str(title))

    # Make boundaries clearer with bold lines (in addition to gap rows).
    for y in gap_line_positions:
        ax.axhline(y, color="#222222", linewidth=2.0, alpha=0.9)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    metric = str(df["metric"].iloc[0]) if "metric" in df.columns and not df.empty else ""
    summary = HeatmapSummary(
        n_rows_total=n_total,
        n_rows_used=int(len(df)),
        n_features=int(len(feature_cols)),
        metric=metric,
        score_col=str(score_col),
        higher_is_better=bool(higher_is_better),
    )
    return {"game_table_csv": str(game_table_csv), "out_path": str(out_path), "summary": summary.__dict__}
