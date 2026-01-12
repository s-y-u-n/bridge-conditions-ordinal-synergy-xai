from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from bci_osxai.analysis.stats import correlation_ratio, cramers_v, safe_jsonable


@dataclass(frozen=True)
class AnalysisOutputs:
    out_dir: Path
    overview_json: Path
    features_summary_csv: Path
    target_summary_csv: Path
    feature_target_assoc_csv: Path
    numeric_correlations_spearman_csv: Path
    report_md: Path


def infer_dataset_key(dataset_config: str | Path, *, configs_root: str | Path = "configs/datasets") -> str:
    dataset_config = Path(dataset_config)
    configs_root = Path(configs_root)
    try:
        rel = dataset_config.resolve().relative_to(configs_root.resolve())
    except Exception:
        rel = dataset_config

    parts = [p for p in rel.parts if p not in {"dataset.yml"}]
    key = "__".join(parts) if parts else dataset_config.stem
    key = key.replace(" ", "_")
    return key


def _is_bool_dtype(series: pd.Series) -> bool:
    return pd.api.types.is_bool_dtype(series) or str(series.dtype).lower() in {"boolean"}


def infer_problem_type(y: pd.Series) -> str:
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return "unknown"
    if pd.api.types.is_numeric_dtype(y_nonnull):
        nunique = int(y_nonnull.nunique(dropna=True))
        # Few unique numeric values are more likely a coded classification/ordinal target.
        if nunique <= 15:
            return "classification"
        return "regression"
    return "classification"


def _feature_kinds(X: pd.DataFrame) -> Dict[str, str]:
    kinds: Dict[str, str] = {}
    for col in X.columns:
        s = X[col]
        if _is_bool_dtype(s):
            kinds[str(col)] = "categorical"
        elif pd.api.types.is_numeric_dtype(s):
            kinds[str(col)] = "numeric"
        else:
            kinds[str(col)] = "categorical"
    return kinds


def summarize_features(X: pd.DataFrame) -> pd.DataFrame:
    n = int(len(X))
    rows: List[Dict[str, Any]] = []
    for col in X.columns:
        s = X[col]
        n_missing = int(s.isna().sum())
        missing_rate = float(n_missing / n) if n else float("nan")
        nunique = int(s.nunique(dropna=True))
        unique_rate = float(nunique / n) if n else float("nan")

        top1 = None
        top1_rate = float("nan")
        vc = s.value_counts(dropna=True)
        if not vc.empty:
            top1 = vc.index[0]
            top1_rate = float(vc.iloc[0] / max(n - n_missing, 1))

        row: Dict[str, Any] = {
            "feature": str(col),
            "dtype": str(s.dtype),
            "kind": "numeric" if (pd.api.types.is_numeric_dtype(s) and not _is_bool_dtype(s)) else "categorical",
            "n_rows": n,
            "n_missing": n_missing,
            "missing_rate": missing_rate,
            "n_unique": nunique,
            "unique_rate": unique_rate,
            "top1": top1,
            "top1_rate": top1_rate,
        }

        if row["kind"] == "numeric":
            s_num = pd.to_numeric(s, errors="coerce")
            desc = s_num.describe(percentiles=[0.25, 0.5, 0.75])
            row.update(
                {
                    "mean": float(desc.get("mean", np.nan)),
                    "std": float(desc.get("std", np.nan)),
                    "min": float(desc.get("min", np.nan)),
                    "p25": float(desc.get("25%", np.nan)),
                    "p50": float(desc.get("50%", np.nan)),
                    "p75": float(desc.get("75%", np.nan)),
                    "max": float(desc.get("max", np.nan)),
                }
            )
        else:
            row.update({"mean": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "max": np.nan})

        rows.append(row)
    return pd.DataFrame(rows)


def summarize_target(y: pd.Series, *, problem_type: str) -> pd.DataFrame:
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return pd.DataFrame([{"n_rows": int(len(y)), "n_nonnull": 0}])

    if problem_type == "regression":
        s = pd.to_numeric(y, errors="coerce")
        desc = s.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        desc = {k: safe_jsonable(v) for k, v in desc.items()}
        desc["n_rows"] = int(len(y))
        desc["n_nonnull"] = int(s.notna().sum())
        return pd.DataFrame([desc])

    counts = y_nonnull.astype("object").value_counts(dropna=True)
    total = float(counts.sum())
    out = pd.DataFrame({"class": counts.index.astype(str), "count": counts.values.astype(int)})
    out["rate"] = out["count"].astype(float) / total if total else np.nan
    return out


def top_spearman_pairs(X: pd.DataFrame, *, top_k: int = 200) -> pd.DataFrame:
    X_num = X.select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "spearman_r"])

    corr = X_num.corr(method="spearman", numeric_only=True)
    pairs: List[Tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iat[i, j]
            if pd.isna(r):
                continue
            pairs.append((str(cols[i]), str(cols[j]), float(r)))

    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    pairs = pairs[: int(top_k)]
    return pd.DataFrame(pairs, columns=["feature_a", "feature_b", "spearman_r"])


def feature_target_association(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    problem_type: str,
    max_contingency_cells: int = 20_000,
) -> pd.DataFrame:
    df = X.copy()
    df["_y_"] = y
    df = df[df["_y_"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["feature", "kind", "method", "score", "p_value", "n_used", "note"])

    y_clean = df.pop("_y_")
    kinds = _feature_kinds(df)

    rows: List[Dict[str, Any]] = []
    if problem_type == "classification":
        # Numeric features: ANOVA F
        from sklearn.feature_selection import f_classif  # noqa: PLC0415

        y_code = y_clean.astype("category").cat.codes.to_numpy()
        numeric_cols = [c for c, k in kinds.items() if k == "numeric"]
        if numeric_cols:
            X_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
            X_num = X_num.fillna(X_num.median(numeric_only=True))
            try:
                f_vals, p_vals = f_classif(X_num.to_numpy(dtype=float), y_code)
                for col, f, p in zip(numeric_cols, f_vals, p_vals):
                    rows.append(
                        {
                            "feature": str(col),
                            "kind": "numeric",
                            "method": "f_classif",
                            "score": float(f),
                            "p_value": float(p),
                            "n_used": int(len(y_clean)),
                            "note": "",
                        }
                    )
            except Exception as exc:  # keep going
                for col in numeric_cols:
                    rows.append(
                        {
                            "feature": str(col),
                            "kind": "numeric",
                            "method": "f_classif",
                            "score": float("nan"),
                            "p_value": float("nan"),
                            "n_used": int(len(y_clean)),
                            "note": f"failed: {type(exc).__name__}",
                        }
                    )

        # Categorical features: Cramér's V
        cat_cols = [c for c, k in kinds.items() if k == "categorical"]
        for col in cat_cols:
            s = df[col]
            # Guard against huge cross-tabs.
            n_cells = int(s.nunique(dropna=True)) * int(y_clean.nunique(dropna=True))
            if n_cells > int(max_contingency_cells):
                rows.append(
                    {
                        "feature": str(col),
                        "kind": "categorical",
                        "method": "cramers_v",
                        "score": float("nan"),
                        "p_value": float("nan"),
                        "n_used": int(len(y_clean)),
                        "note": f"skipped (contingency cells={n_cells})",
                    }
                )
                continue
            rows.append(
                {
                    "feature": str(col),
                    "kind": "categorical",
                    "method": "cramers_v",
                    "score": float(cramers_v(s, y_clean)),
                    "p_value": float("nan"),
                    "n_used": int(len(y_clean)),
                    "note": "",
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(by="score", ascending=False, key=lambda s: s.abs(), ignore_index=True)
        return out

    # Regression
    y_num = pd.to_numeric(y_clean, errors="coerce")
    mask = y_num.notna()
    df = df.loc[mask].copy()
    y_num = y_num.loc[mask]
    if df.empty:
        return pd.DataFrame(columns=["feature", "kind", "method", "score", "p_value", "n_used", "note"])

    numeric_cols = [c for c, k in kinds.items() if k == "numeric"]
    for col in numeric_cols:
        x = pd.to_numeric(df[col], errors="coerce")
        joined = pd.DataFrame({"x": x, "y": y_num}).dropna()
        if joined.empty:
            rows.append({"feature": str(col), "kind": "numeric", "method": "pearson_r", "score": float("nan"), "p_value": float("nan"), "n_used": 0, "note": ""})
            rows.append({"feature": str(col), "kind": "numeric", "method": "spearman_r", "score": float("nan"), "p_value": float("nan"), "n_used": 0, "note": ""})
            continue
        rows.append(
            {
                "feature": str(col),
                "kind": "numeric",
                "method": "pearson_r",
                "score": float(joined["x"].corr(joined["y"], method="pearson")),
                "p_value": float("nan"),
                "n_used": int(len(joined)),
                "note": "",
            }
        )
        rows.append(
            {
                "feature": str(col),
                "kind": "numeric",
                "method": "spearman_r",
                "score": float(joined["x"].corr(joined["y"], method="spearman")),
                "p_value": float("nan"),
                "n_used": int(len(joined)),
                "note": "",
            }
        )

    cat_cols = [c for c, k in kinds.items() if k == "categorical"]
    for col in cat_cols:
        rows.append(
            {
                "feature": str(col),
                "kind": "categorical",
                "method": "correlation_ratio_eta",
                "score": float(correlation_ratio(df[col], y_num)),
                "p_value": float("nan"),
                "n_used": int(len(y_num)),
                "note": "",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by="score", ascending=False, key=lambda s: s.abs(), ignore_index=True)
    return out


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml  # noqa: PLC0415

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML: {path}")
    return data


def _load_task_dataset(dataset_config: Path, *, task: str) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    dataset_cfg = _load_yaml(dataset_config)
    paths = dataset_cfg.get("paths", {}) or {}

    if task == "next-rank":
        next_path = paths.get("next_rank_dataset_csv")
        if not next_path:
            raise FileNotFoundError("dataset.paths.next_rank_dataset_csv is required for task=next-rank.")
        ds = pd.read_csv(next_path)
        ds = ds[ds["label"].notna()].copy()
        drop_cols = {
            "structure_id",
            "label",
            "label_index",
            "target_rank",
            "target_rank_index",
            "target_bci",
            "year_next",
            "current_rank",
        }
        X = ds.drop(columns=[c for c in drop_cols if c in ds.columns], errors="ignore")
        y = ds["label"].astype("object")
        meta = {"input_path": str(next_path)}
        return X, y, meta

    if task == "baseline":
        features_path = paths.get("features_csv")
        labels_path = paths.get("labels_csv")
        if not features_path or not labels_path:
            raise FileNotFoundError("dataset.paths.features_csv and dataset.paths.labels_csv are required for task=baseline.")

        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)
        merged = features.merge(labels[["structure_id", "label"]], on="structure_id", how="inner")
        merged = merged[merged["label"].notna()].copy()
        X = merged.drop(columns=["structure_id", "label"], errors="ignore")
        y = merged["label"]
        meta = {"features_path": str(features_path), "labels_path": str(labels_path)}
        return X, y, meta

    raise ValueError(f"Unknown task: {task}")


def write_report(
    *,
    out_path: Path,
    dataset_key: str,
    task: str,
    overview: Dict[str, Any],
    features_summary: pd.DataFrame,
    feature_target_assoc: pd.DataFrame,
    spearman_pairs: pd.DataFrame,
) -> None:
    missing_top = features_summary.sort_values("missing_rate", ascending=False).head(10)
    assoc_top = feature_target_assoc.head(15) if not feature_target_assoc.empty else feature_target_assoc
    corr_top = spearman_pairs.head(15) if not spearman_pairs.empty else spearman_pairs

    lines: List[str] = []
    lines.append(f"# Feature analysis report: `{dataset_key}` ({task})")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- n_rows: {overview.get('n_rows')}")
    lines.append(f"- n_features: {overview.get('n_features')}")
    lines.append(f"- problem_type: {overview.get('problem_type')}")
    lines.append("")
    lines.append("## Files")
    for k in ["overview.json", "features_summary.csv", "target_summary.csv", "feature_target_assoc.csv", "numeric_correlations_spearman.csv"]:
        lines.append(f"- `{k}`")
    lines.append("")
    lines.append("## Top missing features")
    if missing_top.empty:
        lines.append("- (none)")
    else:
        for _, r in missing_top.iterrows():
            lines.append(f"- `{r['feature']}` missing_rate={float(r['missing_rate']):.3f} n_missing={int(r['n_missing'])}")
    lines.append("")
    lines.append("## Top feature↔target associations")
    if assoc_top.empty:
        lines.append("- (none)")
    else:
        for _, r in assoc_top.iterrows():
            score = r["score"]
            score_s = "nan" if pd.isna(score) else f"{float(score):.4f}"
            lines.append(f"- `{r['feature']}` {r['method']}={score_s} ({r['kind']})")
    lines.append("")
    lines.append("## Top numeric feature correlations (Spearman)")
    if corr_top.empty:
        lines.append("- (none or <2 numeric features)")
    else:
        for _, r in corr_top.iterrows():
            lines.append(f"- `{r['feature_a']}` ↔ `{r['feature_b']}` r={float(r['spearman_r']):.4f}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def analyze_dataset(
    *,
    dataset_config: str | Path,
    task: str = "baseline",
    out_dir: str | Path | None = None,
    configs_root: str | Path = "configs/datasets",
    top_k_corr_pairs: int = 200,
) -> AnalysisOutputs:
    dataset_config = Path(dataset_config)
    dataset_key = infer_dataset_key(dataset_config, configs_root=configs_root)
    task = str(task)

    base_out = Path(out_dir) if out_dir else Path("outputs") / dataset_key / "analysis" / task
    base_out.mkdir(parents=True, exist_ok=True)

    X, y, meta = _load_task_dataset(dataset_config, task=task)
    problem_type = infer_problem_type(y)

    features_summary = summarize_features(X)
    target_summary = summarize_target(y, problem_type=problem_type)
    assoc = feature_target_association(X, y, problem_type=problem_type)
    spearman_pairs = top_spearman_pairs(X, top_k=top_k_corr_pairs)

    overview: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "task": task,
        "problem_type": problem_type,
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_numeric_features": int(len([c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not _is_bool_dtype(X[c])])),
        "n_categorical_features": int(len([c for c in X.columns if not (pd.api.types.is_numeric_dtype(X[c]) and not _is_bool_dtype(X[c]))])),
        "input": meta,
    }

    overview_path = base_out / "overview.json"
    features_summary_path = base_out / "features_summary.csv"
    target_summary_path = base_out / "target_summary.csv"
    assoc_path = base_out / "feature_target_assoc.csv"
    corr_path = base_out / "numeric_correlations_spearman.csv"
    report_path = base_out / "report.md"

    overview_path.write_text(json.dumps({k: safe_jsonable(v) for k, v in overview.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    features_summary.to_csv(features_summary_path, index=False)
    target_summary.to_csv(target_summary_path, index=False)
    assoc.to_csv(assoc_path, index=False)
    spearman_pairs.to_csv(corr_path, index=False)
    write_report(
        out_path=report_path,
        dataset_key=dataset_key,
        task=task,
        overview=overview,
        features_summary=features_summary,
        feature_target_assoc=assoc,
        spearman_pairs=spearman_pairs,
    )

    return AnalysisOutputs(
        out_dir=base_out,
        overview_json=overview_path,
        features_summary_csv=features_summary_path,
        target_summary_csv=target_summary_path,
        feature_target_assoc_csv=assoc_path,
        numeric_correlations_spearman_csv=corr_path,
        report_md=report_path,
    )


def discover_dataset_configs(*, configs_root: str | Path = "configs/datasets") -> List[Path]:
    root = Path(configs_root)
    paths = sorted(root.rglob("dataset.yml"))
    out: List[Path] = []
    for p in paths:
        # Ignore templates (e.g., _template_tabular)
        if any(part.startswith("_template") for part in p.parts):
            continue
        out.append(p)
    return out
