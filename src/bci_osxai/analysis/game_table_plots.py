from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreDistributionSummary:
    n_rows: int
    n_used: int
    n_missing: int
    mean: float
    std: float
    min: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    max: float


def summarize_scores(scores: pd.Series) -> ScoreDistributionSummary:
    s = pd.to_numeric(scores, errors="coerce")
    n_rows = int(len(s))
    n_missing = int(s.isna().sum())
    s = s.dropna()
    if s.empty:
        raise ValueError("No non-missing scores found.")
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()
    return ScoreDistributionSummary(
        n_rows=n_rows,
        n_used=int(len(s)),
        n_missing=n_missing,
        mean=float(s.mean()),
        std=float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        min=float(s.min()),
        p10=float(q.get(0.10, np.nan)),
        p25=float(q.get(0.25, np.nan)),
        p50=float(q.get(0.50, np.nan)),
        p75=float(q.get(0.75, np.nan)),
        p90=float(q.get(0.90, np.nan)),
        max=float(s.max()),
    )


def plot_score_distribution(
    *,
    game_table_csv: str | Path,
    out_path: str | Path,
    score_col: str = "value",
    bins: int = 50,
    title: str | None = None,
    empty_baseline_csv: str | Path | None = None,
    cut_points: Sequence[float] | None = None,
    xlim: tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Plot score distribution for a game table CSV (histogram).

    Returns a JSONable summary dict including quantiles.
    """
    game_table_csv = Path(game_table_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(game_table_csv)
    if score_col not in df.columns:
        raise ValueError(f"Missing score column {score_col!r} in {game_table_csv}")

    scores = pd.to_numeric(df[score_col], errors="coerce")
    summary = summarize_scores(scores)

    baseline_value: Optional[float] = None
    baseline_metric: Optional[str] = None
    if empty_baseline_csv is not None:
        eb = pd.read_csv(Path(empty_baseline_csv))
        if score_col not in eb.columns:
            raise ValueError(f"Missing score column {score_col!r} in {empty_baseline_csv}")
        baseline_value = float(pd.to_numeric(eb[score_col], errors="coerce").dropna().iloc[0])
        if "metric" in eb.columns:
            baseline_metric = str(eb["metric"].iloc[0])

    # Force headless backend at call time (safe if already set).
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg", force=True)
    import japanize_matplotlib  # noqa: F401, PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    s = scores.dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8), constrained_layout=True)

    ax.hist(s, bins=int(bins), color="#4C78A8", alpha=0.85, edgecolor="white")
    x_label = "正解率/MAE"
    inferred_metric: str | None = None
    if "metric" in df.columns:
        metrics = df["metric"].astype(str).dropna().unique().tolist()
        if len(metrics) == 1:
            inferred_metric = metrics[0]
            if metrics[0] == "accuracy":
                x_label = "正解率"
            elif metrics[0] == "mae":
                x_label = "MAE"
            elif metrics[0] == "neg_mae":
                x_label = "-MAE"
            elif metrics[0] == "inv_mae":
                x_label = "1/(1+MAE)"
            elif metrics[0] == "mean_yield":
                x_label = "平均収穫量"
    if inferred_metric == "mean_yield":
        x_label = "平均収穫量 [tons/ha]"

    ax.set_xlabel(x_label)
    ax.set_ylabel("度数")
    ax.grid(True, alpha=0.2)

    if xlim is not None:
        ax.set_xlim(*xlim)

    if baseline_value is not None:
        label = "空集合ベースライン"
        if baseline_metric:
            label += f" ({baseline_metric})"
        ax.axvline(baseline_value, color="#54A24B", linestyle="--", linewidth=2, label=label)

    if cut_points:
        for i, cp in enumerate(cut_points, start=1):
            ax.axvline(float(cp), color="#B279A2", linestyle=":", linewidth=2, label="ランキング境界" if i == 1 else None)

    if baseline_value is not None or cut_points:
        ax.legend(loc="best")

    if title is None:
        if inferred_metric == "mean_yield":
            title = "政策パターンごとの平均収穫量の分布"
        elif inferred_metric in {"accuracy", "mae", "neg_mae", "inv_mae"}:
            title = "特徴量セットごとの予測精度の分布"
        else:
            title = "スコア分布"
    ax.set_title(str(title))

    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    out: Dict[str, Any] = {
        "game_table_csv": str(game_table_csv),
        "out_path": str(out_path),
        "score_col": str(score_col),
        "bins": int(bins),
        "summary": summary.__dict__,
    }
    if baseline_value is not None:
        out["empty_baseline_value"] = float(baseline_value)
        if baseline_metric is not None:
            out["empty_baseline_metric"] = str(baseline_metric)
    if cut_points:
        out["cut_points"] = [float(x) for x in cut_points]
    return out
