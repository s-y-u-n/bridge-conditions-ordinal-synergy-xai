from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from bci_osxai.features.build_features import build_features
from bci_osxai.game_table import (
    GameTableSettings,
    build_coalition_patterns,
    build_empty_baseline_table,
    build_game_table,
    compute_empty_baseline,
    save_game_table,
)
from bci_osxai.io.load_raw import load_raw_csv
from bci_osxai.labels.make_column_labels import make_column_labels
from bci_osxai.labels.make_continuous_targets import make_continuous_targets
from bci_osxai.labels.make_ordinal_labels import make_ordinal_labels
from bci_osxai.labels.make_rehab_floor_rank_labels import RehabFloorRankConfig, make_rehab_floor_rank_labels
from bci_osxai.analysis.dataset_analysis import analyze_dataset, discover_dataset_configs, infer_dataset_key
from bci_osxai.analysis.coalition_heatmap import plot_coalition_feature_heatmap
from bci_osxai.analysis.game_table_plots import plot_score_distribution
from bci_osxai.pipeline.dataset_pipeline import infer_pipeline_paths
from bci_osxai.ranking.weak_order import Criterion, dumps_json, rank_to_weak_order
from bci_osxai.preprocess.build_next_rank_dataset import NextRankConfig, build_next_rank_dataset
from bci_osxai.preprocess.clean_raw_csv import clean_multiline_csv, load_rename_map_from_data_dictionary
from bci_osxai.preprocess.clean_schema import normalize_column_whitespace
from bci_osxai.preprocess.reshape_bci import build_bci_long
from bci_osxai.models.train import save_model, train_xgb_classifier, train_xgb_regressor
from bci_osxai.analysis.resignation_rate_table import (
    ResignationRateSettings,
    build_resignation_rate_table,
    build_resignation_risk_factors,
)
from bci_osxai.analysis.crop_policy_game_table import (
    CropPolicySpec,
    build_crop_policy_flags,
    build_crop_policy_game_table,
)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML: {path}")
    return data


def load_task_dataset(dataset_config: Path, *, task: str) -> tuple[pd.DataFrame, pd.Series]:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]

    if task == "next-rank":
        next_path = paths.get("next_rank_dataset_csv")
        if not next_path:
            raise SystemExit("dataset.paths.next_rank_dataset_csv is required for --task next-rank.")
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
        return X, y

    if task == "baseline":
        features_path = paths.get("features_csv")
        labels_path = paths.get("labels_csv")
        if not features_path or not labels_path:
            raise SystemExit("dataset.paths.features_csv and dataset.paths.labels_csv are required for --task baseline.")
        features = pd.read_csv(features_path)
        labels = pd.read_csv(labels_path)

        merged = features.merge(labels[["structure_id", "label"]], on="structure_id", how="inner")
        merged = merged[merged["label"].notna()].copy()
        X = merged.drop(columns=["structure_id", "label"], errors="ignore")
        y = merged["label"]
        return X, y

    raise ValueError(f"Unknown task: {task}")


def run_preprocess(dataset_config: Path, labeling_config: Path) -> None:
    dataset_cfg = load_yaml(dataset_config)
    labeling_cfg = load_yaml(labeling_config)

    paths = dataset_cfg["paths"]
    columns = (dataset_cfg.get("columns") or {}) if isinstance(dataset_cfg, dict) else {}
    encoding = (dataset_cfg.get("io") or {}).get("encoding")
    rename_columns = bool((dataset_cfg.get("io") or {}).get("rename_columns", False))
    io_format = str((dataset_cfg.get("io") or {}).get("format") or "csv").strip().lower()
    csv_has_header = bool((dataset_cfg.get("io") or {}).get("has_header", True))
    csv_columns = (dataset_cfg.get("io") or {}).get("columns")

    raw_path = paths.get("raw_csv") or paths.get("raw_path")
    if not raw_path:
        raise SystemExit("dataset.paths.raw_csv (or raw_path) is required.")
    clean_csv = paths.get("raw_clean_csv")
    if io_format == "csv" and clean_csv:
        rename_map = None
        if rename_columns:
            year_cfg = dataset_cfg.get("bci_years", {}) or {}
            rename_map = load_rename_map_from_data_dictionary(
                paths.get("data_dictionary", "docs/data_dictionary_bridge_conditions.md"),
                year_start=int(year_cfg.get("start", 2000)),
                year_end=int(year_cfg.get("end", 2020)),
            )
        clean_multiline_csv(raw_path, clean_csv, encoding=encoding or "utf-8", rename_map=rename_map)
        raw_path = clean_csv

    if io_format != "csv":
        raise SystemExit(f"Unsupported io.format: {io_format!r} (expected: csv only)")
    df = load_raw_csv(raw_path, encoding=encoding, has_header=csv_has_header, columns=csv_columns)
    df = normalize_column_whitespace(df)

    filters = dataset_cfg.get("filters", {}) or {}
    for key, value in filters.items():
        column_name = columns.get(key)
        if column_name and column_name in df.columns:
            df = df[df[column_name] == value].copy()
    # Ensure stable row indexing after filtering; avoids accidental index-alignment
    # issues when creating new Series (e.g., structure_id) downstream.
    df = df.reset_index(drop=True)

    id_source = columns.get("id")
    df = df.copy()
    if id_source is not None and str(id_source).strip() != "" and str(id_source) in df.columns:
        df["structure_id"] = df[str(id_source)].astype(str)
    else:
        df["structure_id"] = pd.Series(range(1, len(df) + 1), index=df.index, dtype="int64").astype(str)

    is_bridge_conditions = bool(dataset_cfg.get("bci_years")) and bool(columns.get("current_bci"))
    bci_long = None
    year_cols: list[str] = []
    if is_bridge_conditions:
        year_cfg = dataset_cfg.get("bci_years", {}) or {}
        start = int(year_cfg.get("start", 2000))
        end = int(year_cfg.get("end", 2020))
        col_format = str(year_cfg.get("col_format", "{year}"))
        year_cols = [col_format.format(year=year) for year in range(start, end + 1)]
        bci_long = build_bci_long(df, "structure_id", year_cols, str(columns["current_bci"]))

    inspection_year = (dataset_cfg.get("inspection_year") or {}).get("default")
    if is_bridge_conditions:
        features = build_features(df, columns, inspection_year=inspection_year)
    else:
        target_fallback = columns.get("target")
        target_cfg = labeling_cfg.get("target", {}) or {}
        target_col = target_cfg.get("column") or target_fallback
        drop = {"structure_id"}
        if target_col and str(target_col) in df.columns:
            drop.add(str(target_col))
        features = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore").copy()
        features.insert(0, "structure_id", df["structure_id"].astype(str))

    include = (dataset_cfg.get("feature_columns") or {}).get("include")
    if include:
        include_set = {str(c) for c in include}
        include_set.add("structure_id")
        features = features[[c for c in features.columns if c in include_set]].copy()

    scheme = labeling_cfg["active_scheme"]
    scheme_cfg = labeling_cfg["schemes"][scheme]
    scheme_type = scheme_cfg.get("type", "fixed_thresholds")
    target_cfg = labeling_cfg.get("target", {}) or {}

    if target_cfg.get("source") == "year":
        if not is_bridge_conditions:
            raise SystemExit("target.source=year is only supported for bridge_conditions-style datasets.")
        year_cfg = dataset_cfg.get("bci_years", {}) or {}
        bci_col = str(year_cfg.get("col_format", "{year}")).format(year=int(target_cfg.get("year")))
    elif target_cfg.get("source") == "column":
        bci_col = str(target_cfg.get("column"))
    else:
        if columns.get("current_bci") is None:
            raise SystemExit("labeling.target.column is required for non-bridge datasets (set target.source: column).")
        bci_col = str(target_cfg.get("column", columns["current_bci"]))

    if scheme_type == "continuous_bci":
        labels = make_continuous_targets(df, id_col="structure_id", bci_col=bci_col)
    elif scheme_type == "column":
        labels = make_column_labels(
            df,
            id_col="structure_id",
            target_col=bci_col,
            as_categorical=bool(target_cfg.get("as_categorical", False)),
        )
    elif scheme_type == "rehab_floor_ranks":
        dist = scheme_cfg.get("distribution", {}) or {}
        config = RehabFloorRankConfig(
            n_ranks=int(scheme_cfg.get("n_ranks", 5)),
            increase_delta=float(scheme_cfg.get("increase_delta", 0.5)),
            label_prefix=str(scheme_cfg.get("label_prefix", "R")),
            distribution_name=str(dist.get("name", "beta")),
            alpha=float(dist.get("alpha", 5.0)),
            beta=float(dist.get("beta", 2.0)),
        )
        labels = make_rehab_floor_rank_labels(
            df,
            id_col="structure_id",
            target_bci_col=bci_col,
            bci_year_cols=year_cols,
            config=config,
        )
    else:
        thresholds = scheme_cfg["labels"]
        labels = make_ordinal_labels(df, "structure_id", bci_col, thresholds)

    next_rank_path = paths.get("next_rank_dataset_csv")
    if next_rank_path and scheme_type == "rehab_floor_ranks":
        if bci_long is None:
            raise SystemExit("rehab_floor_ranks requires bci_long to be available.")
        dist = scheme_cfg.get("distribution", {}) or {}
        next_cfg = NextRankConfig(
            n_ranks=int(scheme_cfg.get("n_ranks", 5)),
            increase_delta=float(scheme_cfg.get("increase_delta", 0.5)),
            label_prefix=str(scheme_cfg.get("label_prefix", "R")),
            distribution_name=str(dist.get("name", "beta")),
            alpha=float(dist.get("alpha", 5.0)),
            beta=float(dist.get("beta", 2.0)),
        )
        next_dataset = build_next_rank_dataset(structures=df, bci_long=bci_long, config=next_cfg)
        Path(next_rank_path).parent.mkdir(parents=True, exist_ok=True)
        next_dataset.to_csv(next_rank_path, index=False)

    structures_path = paths.get("structures_csv")
    if not structures_path:
        raise SystemExit("dataset.paths.structures_csv is required.")
    Path(structures_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(structures_path, index=False)

    bci_long_path = paths.get("bci_long_csv")
    if bci_long is not None and bci_long_path:
        Path(bci_long_path).parent.mkdir(parents=True, exist_ok=True)
        bci_long.to_csv(bci_long_path, index=False)

    features_path = paths.get("features_csv")
    if not features_path:
        raise SystemExit("dataset.paths.features_csv is required.")
    Path(features_path).parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(features_path, index=False)

    labels_path = paths.get("labels_csv")
    if not labels_path:
        raise SystemExit("dataset.paths.labels_csv is required.")
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(labels_path, index=False)


def run_build_game_table(*, dataset_config: Path, game_table_config: Path, task: str, out: str | None, no_progress: bool) -> None:
    cfg = load_yaml(game_table_config)
    game_cfg = (cfg.get("game_table", {}) or {}) if isinstance(cfg, dict) else {}
    default_out = game_cfg.get("cache_path")
    out_path = Path(out) if out else (Path(default_out) if default_out else None)
    if out_path is None:
        raise SystemExit("Output path is required: set game_table.cache_path in game_table config or pass --out.")
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")

    X, y = load_task_dataset(dataset_config, task=str(task))
    max_rows = game_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows > 0 and len(y) > max_rows:
            from sklearn.model_selection import train_test_split  # noqa: PLC0415

            seed = int(game_cfg.get("seed", 42))
            is_regression = pd.api.types.is_numeric_dtype(y)
            if is_regression:
                X, _, y, _ = train_test_split(X, y, train_size=max_rows, random_state=seed, shuffle=True)
            else:
                stratify = y if y.nunique(dropna=True) > 1 else None
                X, _, y, _ = train_test_split(X, y, train_size=max_rows, random_state=seed, shuffle=True, stratify=stratify)

    settings = GameTableSettings(
        enabled=True,
        max_order=int(game_cfg.get("max_order", 2)),
        n_coalitions=int(game_cfg.get("n_coalitions", 400)),
        test_size=float(game_cfg.get("test_size", 0.25)),
        metric=str(game_cfg.get("metric", "accuracy")),
        seed=int(game_cfg.get("seed", 42)),
        n_repeats=int(game_cfg.get("n_repeats", 1) or 1),
        cv_folds=int(game_cfg.get("cv_folds", 0) or 0),
    )
    table = build_game_table(X=X, y=y, players=list(X.columns), settings=settings, progress=not bool(no_progress))
    save_game_table(table, out_path)
    print(str(out_path))


def run_build_empty_baseline(*, dataset_config: Path, game_table_config: Path, task: str, out: str | None) -> None:
    cfg = load_yaml(game_table_config)
    game_cfg = (cfg.get("game_table", {}) or {}) if isinstance(cfg, dict) else {}

    if out:
        out_path = Path(out)
    else:
        dataset_key = infer_dataset_key(dataset_config)
        out_path = Path("outputs") / dataset_key / "game_tables" / "empty_baseline.csv"
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")

    X, y = load_task_dataset(dataset_config, task=str(task))
    max_rows = game_cfg.get("max_rows")
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows > 0 and len(y) > max_rows:
            from sklearn.model_selection import train_test_split  # noqa: PLC0415

            seed = int(game_cfg.get("seed", 42))
            is_regression = pd.api.types.is_numeric_dtype(y)
            if is_regression:
                X, _, y, _ = train_test_split(X, y, train_size=max_rows, random_state=seed, shuffle=True)
            else:
                stratify = y if y.nunique(dropna=True) > 1 else None
                X, _, y, _ = train_test_split(X, y, train_size=max_rows, random_state=seed, shuffle=True, stratify=stratify)

    settings = GameTableSettings(
        enabled=True,
        max_order=int(game_cfg.get("max_order", 2)),
        n_coalitions=int(game_cfg.get("n_coalitions", 400)),
        test_size=float(game_cfg.get("test_size", 0.25)),
        metric=str(game_cfg.get("metric", "accuracy")),
        seed=int(game_cfg.get("seed", 42)),
        n_repeats=int(game_cfg.get("n_repeats", 1) or 1),
        cv_folds=int(game_cfg.get("cv_folds", 0) or 0),
    )
    baseline = compute_empty_baseline(X=X, y=y, settings=settings)
    table = build_empty_baseline_table(players=list(X.columns), baseline=baseline)
    save_game_table(table, out_path)
    print(str(out_path))


def run_build_crop_policy_game_table(*, dataset_config: Path, out: str | None, crop: str | None) -> None:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg.get("paths", {}) or {}
    columns = dataset_cfg.get("columns", {}) or {}
    encoding = (dataset_cfg.get("io") or {}).get("encoding")
    csv_has_header = bool((dataset_cfg.get("io") or {}).get("has_header", True))
    csv_columns = (dataset_cfg.get("io") or {}).get("columns")

    raw_path = paths.get("raw_csv") or paths.get("raw_path")
    if not raw_path:
        raise SystemExit("dataset.paths.raw_csv (or raw_path) is required.")

    df = load_raw_csv(raw_path, encoding=encoding, has_header=csv_has_header, columns=csv_columns)
    df = normalize_column_whitespace(df)

    dataset_key = infer_dataset_key(dataset_config)
    if out:
        out_path = Path(out)
    else:
        out_path = Path("outputs") / dataset_key / "game_tables" / "game_table.csv"
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    spec = CropPolicySpec(
        crop_col=str(columns.get("crop", "Crop")),
        soil_col=str(columns.get("soil", "Soil_Type")),
        rainfall_col=str(columns.get("rainfall", "Rainfall_mm")),
        irrigation_col=str(columns.get("irrigation", "Irrigation_Used")),
        fertilizer_col=str(columns.get("fertilizer", "Fertilizer_Used")),
        target_col=str(columns.get("target", "Yield_tons_per_hectare")),
    )
    factors, meta = build_crop_policy_flags(df, spec=spec, crop_value=crop)
    players = [spec.out_high_rain_flag, spec.out_irrigation_flag, spec.out_fertilizer_flag, spec.out_improved_soil_flag]
    table = build_crop_policy_game_table(factors, players=players, target_col=spec.target_col)

    # Align with crop/wine game_table format as much as possible.
    table["n_train"] = table["n_rows"].astype(int)
    table["n_test"] = 0
    table["seed"] = int(spec.rainfall_random_state)
    table = table.drop(columns=["n_rows"], errors="ignore")

    table.to_csv(out_path, index=False)
    print(str(out_path))

    meta_path = out_path.with_name(f"{out_path.stem}__meta.json")
    meta_path.write_text(dumps_json({"meta": meta}), encoding="utf-8")
    print(str(meta_path))


def run_plot_game_table_scores(
    *,
    game_table_csv: Path,
    out: str | None,
    score_col: str,
    bins: int,
    title: str | None,
    empty_baseline_csv: str | None,
    ranked_csv: str | None,
) -> None:
    if out:
        out_path = Path(out)
    else:
        out_path = game_table_csv.with_name(f"{game_table_csv.stem}__score_distribution.png")

    cut_points = None
    if ranked_csv:
        rdf = pd.read_csv(ranked_csv)
        if "class_id" not in rdf.columns:
            raise SystemExit("--ranked-csv must include class_id column (use rank-game-table first).")
        if score_col not in rdf.columns:
            raise SystemExit(f"--ranked-csv must include score column {score_col!r}.")
        higher_is_better = True
        if "metric" in rdf.columns:
            ms = rdf["metric"].astype(str).dropna().unique().tolist()
            if len(ms) == 1 and ms[0] == "mae":
                higher_is_better = False
        # Cut points are midpoints between adjacent classes in descending score order.
        rdf = rdf.sort_values(by=[score_col, "class_id"], ascending=[not higher_is_better, True], kind="mergesort").copy()
        groups = rdf.groupby("class_id")[score_col]
        class_max = groups.max().sort_index()
        class_min = groups.min().sort_index()
        cps: list[float] = []
        for cid in range(1, int(class_max.index.max())):
            if cid not in class_min.index or (cid + 1) not in class_max.index:
                continue
            if higher_is_better:
                upper_min = float(class_min.loc[cid])
                lower_max = float(class_max.loc[cid + 1])
                cps.append((upper_min + lower_max) / 2.0)
            else:
                upper_max = float(class_max.loc[cid])
                lower_min = float(class_min.loc[cid + 1])
                cps.append((upper_max + lower_min) / 2.0)
        cut_points = cps

    summary = plot_score_distribution(
        game_table_csv=game_table_csv,
        out_path=out_path,
        score_col=str(score_col),
        bins=int(bins),
        title=title,
        empty_baseline_csv=empty_baseline_csv,
        cut_points=cut_points,
    )
    print(str(out_path))
    # Also print a compact summary for quick copy/paste
    s = summary["summary"]
    print(
        f"n={s['n_used']} missing={s['n_missing']} "
        f"min={s['min']:.6g} p25={s['p25']:.6g} p50={s['p50']:.6g} p75={s['p75']:.6g} max={s['max']:.6g}"
    )


def run_rank_game_table(
    *,
    game_table_csv: Path,
    out_ranked: str | None,
    out_classes: str | None,
    out_summary: str | None,
    score_col: str,
    id_col: str,
    k_max: int | None,
    k_fixed: int | None,
    criterion: Criterion,
    ranked_format: str,
    plot_out: str | None,
    heatmap_out: str | None,
    empty_baseline_csv: str | None,
    bins: int,
    title: str | None,
) -> None:
    df = pd.read_csv(game_table_csv)

    if empty_baseline_csv:
        eb = pd.read_csv(empty_baseline_csv)
        if eb.empty:
            raise SystemExit(f"--empty-baseline-csv is empty: {empty_baseline_csv}")
        if score_col not in eb.columns:
            raise SystemExit(f"--empty-baseline-csv must include score column {score_col!r}.")

        # Infer 0/1 player columns from the game table to identify the empty coalition row.
        exclude = {
            str(score_col),
            "value",
            "abs_value",
            "metric",
            "order",
            "n_train",
            "n_test",
            "seed",
            "coalition_key",
            "coalition_id",
            "class_id",
            "k_selected",
            "class_score_max",
            "class_score_min",
            "class_size",
        }
        player_cols: list[str] = []
        for c in df.columns:
            sc = str(c)
            if sc in exclude:
                continue
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                vals = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
                if vals.issubset({0, 1}):
                    player_cols.append(sc)

        eb_row = eb.iloc[[0]].copy()
        # Align columns
        for col in df.columns:
            if col not in eb_row.columns:
                eb_row[col] = 0 if col in player_cols else pd.NA
        eb_row = eb_row[df.columns].copy()
        if "order" in eb_row.columns:
            eb_row["order"] = 0
        for col in player_cols:
            eb_row[col] = 0

        # If an empty coalition row exists in the game table, overwrite its score with the empty baseline.
        if player_cols:
            is_empty = (df[player_cols].sum(axis=1) == 0)
            if "order" in df.columns:
                is_empty = is_empty & (pd.to_numeric(df["order"], errors="coerce").fillna(0).astype(int) == 0)
            if bool(is_empty.any()):
                idx = df.index[is_empty][0]
                for col in eb_row.columns:
                    df.at[idx, col] = eb_row.iloc[0][col]
            else:
                df = pd.concat([df, eb_row], ignore_index=True)
        else:
            df = pd.concat([df, eb_row], ignore_index=True)

    ranked, classes, summary = rank_to_weak_order(
        df,
        id_col=str(id_col),
        score_col=str(score_col),
        k_max=k_max,
        k_fixed=k_fixed,
        criterion=criterion,
    )

    ranked_format = str(ranked_format).strip().lower()
    if ranked_format not in {"score-only", "full"}:
        raise SystemExit("--ranked-format must be one of: score-only, full")

    if ranked_format == "score-only":
        base_cols = list(df.columns)
        # Only add class_id (and keep any existing columns, including score_col).
        keep_cols = [c for c in base_cols if c in ranked.columns] + ["class_id"]
        ranked_to_write = ranked.loc[:, keep_cols].copy()
    else:
        ranked_to_write = ranked

    if out_ranked:
        ranked_path = Path(out_ranked)
    else:
        ranked_path = game_table_csv.with_name(f"{game_table_csv.stem}__ranked.csv")
    ranked_path.parent.mkdir(parents=True, exist_ok=True)
    ranked_to_write.to_csv(ranked_path, index=False)
    print(str(ranked_path))

    if out_classes:
        classes_path = Path(out_classes)
    else:
        classes_path = game_table_csv.with_name(f"{game_table_csv.stem}__classes.json")
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text(dumps_json({"classes": classes}), encoding="utf-8")
    print(str(classes_path))

    if out_summary:
        summary_path = Path(out_summary)
    else:
        summary_path = game_table_csv.with_name(f"{game_table_csv.stem}__rank_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(dumps_json({"summary": summary}), encoding="utf-8")
    print(str(summary_path))

    if plot_out:
        # Build cut points from ranked output and plot histogram only.
        run_plot_game_table_scores(
            game_table_csv=game_table_csv,
            out=plot_out,
            score_col=score_col,
            bins=int(bins),
            title=title,
            empty_baseline_csv=empty_baseline_csv,
            ranked_csv=str(ranked_path),
        )

    if heatmap_out:
        heatmap_path = Path(heatmap_out)
        plot_coalition_feature_heatmap(
            game_table_csv=game_table_csv,
            out_path=heatmap_path,
            score_col=str(score_col),
            ranked_csv=str(ranked_path),
        )
        print(str(heatmap_path))


def _load_feature_columns_for_task(dataset_config: Path, *, task: str) -> list[str]:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]
    task = str(task)

    if task == "baseline":
        features_path = paths.get("features_csv")
        if not features_path:
            raise SystemExit("dataset.paths.features_csv is required for --task baseline.")
        cols = list(pd.read_csv(features_path, nrows=0).columns)
        cols = [c for c in cols if c != "structure_id"]
        return [str(c) for c in cols]

    if task == "next-rank":
        next_path = paths.get("next_rank_dataset_csv")
        if not next_path:
            raise SystemExit("dataset.paths.next_rank_dataset_csv is required for --task next-rank.")
        cols = list(pd.read_csv(next_path, nrows=0).columns)
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
        cols = [c for c in cols if c not in drop_cols]
        return [str(c) for c in cols]

    raise SystemExit(f"Unknown task: {task}")


def run_build_pattern_table(*, dataset_config: Path, task: str, out: str | None, max_order: int | None) -> None:
    players = _load_feature_columns_for_task(dataset_config, task=str(task))
    patterns = build_coalition_patterns(players, max_order=max_order)

    if out:
        out_path = Path(out)
    else:
        dataset_key = infer_dataset_key(dataset_config)
        out_path = Path("outputs") / dataset_key / "game_tables" / "patterns.csv"
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    patterns.to_csv(out_path, index=False)
    print(str(out_path))


def run_train_model(
    *,
    dataset_config: Path,
    task: str,
    out_model: str | None,
    test_size: float,
    seed: int,
) -> None:
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    X, y = load_task_dataset(dataset_config, task=str(task))
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    dataset_key = infer_dataset_key(dataset_config)
    if out_model:
        out_path = Path(out_model)
    else:
        out_path = Path("outputs") / dataset_key / "models" / "model.pkl"

    stratify = None
    is_regression = pd.api.types.is_numeric_dtype(y)
    if not is_regression and y.nunique(dropna=True) > 1:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=stratify,
    )

    if is_regression:
        model = train_xgb_regressor(X_train, y_train)
        y_pred = model.predict(X_test)
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415

        mae = float(mean_absolute_error(y_test.astype(float), y_pred.astype(float)))
        metrics = {"problem_type": "regression", "mae": mae, "n_train": int(len(X_train)), "n_test": int(len(X_test))}
    else:
        model = train_xgb_classifier(X_train, y_train)
        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score  # noqa: PLC0415

        acc = float(accuracy_score(y_test.astype("object"), pd.Series(y_pred, index=y_test.index).astype("object")))
        metrics = {"problem_type": "classification", "accuracy": acc, "n_train": int(len(X_train)), "n_test": int(len(X_test))}

    save_model(model, out_path)
    metrics_path = out_path.with_suffix(".metrics.yml")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metrics, handle, sort_keys=True, allow_unicode=True)

    print(str(out_path))
    print(str(metrics_path))


def run_analyze_dataset(*, dataset_config: Path, task: str, out_dir: str | None, configs_root: str, top_k_corr_pairs: int) -> None:
    outputs = analyze_dataset(
        dataset_config=dataset_config,
        task=str(task),
        out_dir=out_dir,
        configs_root=configs_root,
        top_k_corr_pairs=int(top_k_corr_pairs),
    )
    print(str(outputs.out_dir))

def run_build_resignation_rate_table(*, dataset_config: Path, out: str | None, include_empty: bool) -> None:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]
    raw_path = paths.get("raw_csv") or paths.get("raw_path")
    if not raw_path:
        raise SystemExit("dataset.paths.raw_csv (or raw_path) is required.")

    encoding = (dataset_cfg.get("io") or {}).get("encoding")
    df = load_raw_csv(raw_path, encoding=encoding)
    df = normalize_column_whitespace(df)

    factors = build_resignation_risk_factors(df)
    players = ["low_income", "long_overtime", "low_remote", "large_team", "low_promotion"]
    table = build_resignation_rate_table(
        factors=factors,
        players=players,
        settings=ResignationRateSettings(include_empty_coalition=bool(include_empty)),
    )

    if out:
        out_path = Path(out)
    else:
        dataset_key = infer_dataset_key(dataset_config)
        out_path = Path("outputs") / dataset_key / "game_tables" / "resignation_rate_table.csv"
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(str(out_path))


def run_analyze_all_datasets(*, configs_root: Path, task: str, out_root: str | None, continue_on_error: bool, top_k_corr_pairs: int) -> None:
    failed: list[tuple[str, str]] = []
    for dataset_cfg in discover_dataset_configs(configs_root=configs_root):
        try:
            dataset_key = infer_dataset_key(dataset_cfg, configs_root=configs_root)
            out_dir = None
            if out_root:
                out_dir = str(Path(out_root) / dataset_key / "analysis" / str(task))
            outputs = analyze_dataset(
                dataset_config=dataset_cfg,
                task=str(task),
                out_dir=out_dir,
                configs_root=str(configs_root),
                top_k_corr_pairs=int(top_k_corr_pairs),
            )
            print(str(outputs.out_dir))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            failed.append((str(dataset_cfg), msg))
            if not bool(continue_on_error):
                raise

    if failed:
        lines = ["Some dataset analyses failed:"]
        for path, msg in failed:
            lines.append(f"- {path}: {msg}")
        if not bool(continue_on_error):
            raise SystemExit("\n".join(lines))
        print("\n".join(lines))


def run_dataset_pipeline(
    *,
    dataset_config: Path,
    labeling_config: Path | None,
    game_table_config: Path,
    task: str,
    k_max: int,
    criterion: str,
    ranked_format: str,
    heatmap_top_n: int,
    no_progress: bool,
    skip_preprocess: bool,
    skip_game_table: bool,
    dry_run: bool,
) -> None:
    paths = infer_pipeline_paths(dataset_config=dataset_config, game_table_config=game_table_config)

    steps: list[tuple[str, dict[str, Any]]] = []
    if not bool(skip_preprocess):
        if labeling_config is None:
            raise SystemExit("--labeling-config is required unless --skip-preprocess is set.")
        steps.append(
            (
                "preprocess",
                {"dataset_config": dataset_config, "labeling_config": labeling_config},
            )
        )

    steps.append(
        (
            "build-empty-baseline",
            {"dataset_config": dataset_config, "game_table_config": game_table_config, "task": task, "out": str(paths.empty_baseline_csv)},
        )
    )

    if not bool(skip_game_table):
        steps.append(
            (
                "build-game-table",
                {
                    "dataset_config": dataset_config,
                    "game_table_config": game_table_config,
                    "task": task,
                    "out": str(paths.game_table_csv),
                    "no_progress": bool(no_progress),
                },
            )
        )

    steps.append(
        (
            "rank-game-table",
            {
                "game_table_csv": Path(paths.game_table_csv),
                "out_ranked": str(paths.ranked_csv),
                "out_classes": str(paths.classes_json),
                "out_summary": str(paths.rank_summary_json),
                "score_col": "value",
                "id_col": "coalition_id",
                "k_max": int(k_max),
                "k_fixed": None,
                "criterion": str(criterion),
                "ranked_format": str(ranked_format),
                "plot_out": str(paths.score_distribution_png),
                "heatmap_out": str(paths.heatmap_png),
                "empty_baseline_csv": str(paths.empty_baseline_csv),
                "bins": 60,
                "title": None,
            },
        )
    )

    if bool(dry_run):
        print("dataset_key:", paths.dataset_key)
        print("outputs:")
        print("  game_table:", str(paths.game_table_csv))
        print("  empty_baseline:", str(paths.empty_baseline_csv))
        print("  ranked:", str(paths.ranked_csv))
        print("  score_distribution:", str(paths.score_distribution_png))
        print("  heatmap:", str(paths.heatmap_png))
        print("steps:")
        for name, kwargs in steps:
            print("-", name, {k: str(v) for k, v in kwargs.items()})
        return

    for name, kwargs in steps:
        if name == "preprocess":
            run_preprocess(Path(kwargs["dataset_config"]), Path(kwargs["labeling_config"]))
        elif name == "build-empty-baseline":
            run_build_empty_baseline(**kwargs)  # type: ignore[arg-type]
        elif name == "build-game-table":
            run_build_game_table(**kwargs)  # type: ignore[arg-type]
        elif name == "rank-game-table":
            # Allow criterion values from args (str) - validated in argparse.
            kwargs["criterion"] = kwargs["criterion"]  # type: ignore[assignment]
            run_rank_game_table(**kwargs)  # type: ignore[arg-type]
        else:
            raise SystemExit(f"Unknown pipeline step: {name}")

    # Re-render heatmap with requested top_n (rank step outputs a heatmap only if requested there).
    top_n = int(heatmap_top_n)
    plot_coalition_feature_heatmap(
        game_table_csv=paths.game_table_csv,
        out_path=paths.heatmap_png,
        score_col="value",
        top_n=None if top_n == 0 else top_n,
        ranked_csv=paths.ranked_csv,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bci-xai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_dataset_cfg = "configs/datasets/bridge_conditions/dataset.yml"
    default_labeling_cfg = "configs/datasets/bridge_conditions/labeling.yml"
    default_game_table_cfg = "configs/datasets/bridge_conditions/game_table.yml"

    preprocess = subparsers.add_parser("preprocess", help="Run preprocessing pipeline (cleansing -> features/labels)")
    preprocess.add_argument("--dataset-config", default=default_dataset_cfg)
    preprocess.add_argument("--labeling-config", default=default_labeling_cfg)

    build_gt = subparsers.add_parser("build-game-table", help="Build a coalition->score game table by retraining with feature masks")
    build_gt.add_argument("--dataset-config", default=default_dataset_cfg)
    build_gt.add_argument("--task", default="baseline", choices=["baseline", "next-rank"])
    build_gt.add_argument("--game-table-config", default=default_game_table_cfg, help="Reads game_table.* settings from this file")
    build_gt.add_argument("--out", default=None, help="Write to this path (.csv; default: game_table.cache_path)")
    build_gt.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")

    empty_baseline = subparsers.add_parser(
        "build-empty-baseline",
        help="Compute v(empty): uninformative baseline score using labels only (no feature training)",
    )
    empty_baseline.add_argument("--dataset-config", default=default_dataset_cfg)
    empty_baseline.add_argument("--task", default="baseline", choices=["baseline", "next-rank"])
    empty_baseline.add_argument("--game-table-config", default=default_game_table_cfg, help="Reads game_table.* settings from this file")
    empty_baseline.add_argument("--out", default=None, help="Write to this path (.csv; default: outputs/<dataset_key>/game_tables/empty_baseline.csv)")

    plot_scores = subparsers.add_parser("plot-game-table-scores", help="Plot distribution of scores in a game table CSV")
    plot_scores.add_argument("--game-table-csv", required=True, help="Path to game_table.csv (wide format)")
    plot_scores.add_argument("--out", default=None, help="Write plot to this path (.png; default: <game_table>__score_distribution.png)")
    plot_scores.add_argument("--score-col", default="value", help="Column to plot (default: value)")
    plot_scores.add_argument("--bins", type=int, default=50, help="Histogram bins (default: 50)")
    plot_scores.add_argument("--title", default=None, help="Figure title (default: game_table filename)")
    plot_scores.add_argument("--empty-baseline-csv", default=None, help="Optional empty_baseline.csv to overlay as a vertical line")
    plot_scores.add_argument("--ranked-csv", default=None, help="Optional ranked CSV (with class_id) to overlay cut lines")

    plot_heatmap = subparsers.add_parser("plot-game-table-heatmap", help="Plot coalition×feature heatmap (red if feature used)")
    plot_heatmap.add_argument("--game-table-csv", required=True, help="Path to game_table.csv (wide format)")
    plot_heatmap.add_argument("--out", default=None, help="Write plot to this path (.png; default: <game_table>__heatmap.png)")
    plot_heatmap.add_argument("--score-col", default="value", help="Score column used to order rows (default: value)")
    plot_heatmap.add_argument("--top-n", type=int, default=200, help="Plot top N rows (0=all; default: 200)")
    plot_heatmap.add_argument("--title", default=None, help="Figure title")
    plot_heatmap.add_argument("--ranked-csv", default=None, help="Optional ranked CSV to draw class boundaries")

    rank_gt = subparsers.add_parser("rank-game-table", help="Rank coalitions into weak order classes via optimal 1D k-means segmentation")
    rank_gt.add_argument("--game-table-csv", required=True, help="Path to game_table.csv (wide format)")
    rank_gt.add_argument("--score-col", default="value", help="Score column to use (default: value)")
    rank_gt.add_argument("--id-col", default="coalition_id", help="Coalition id column (default: coalition_id; derived if missing)")
    rank_gt.add_argument("--k-max", type=int, default=None, help="Max number of classes to consider (default: min(20, n))")
    rank_gt.add_argument("--k-fixed", type=int, default=None, help="Force k classes (overrides criterion)")
    rank_gt.add_argument("--criterion", default="bic", choices=["bic", "mdl"], help="Model selection criterion (default: bic)")
    rank_gt.add_argument(
        "--ranked-format",
        default="score-only",
        choices=["score-only", "full"],
        help="Ranked CSV output columns (default: score-only adds class_id only)",
    )
    rank_gt.add_argument("--out-ranked", default=None, help="Write ranked CSV here (default: <game_table>__ranked.csv)")
    rank_gt.add_argument("--out-classes", default=None, help="Write classes JSON here (default: <game_table>__classes.json)")
    rank_gt.add_argument("--out-summary", default=None, help="Write summary JSON here (default: <game_table>__rank_summary.json)")
    rank_gt.add_argument("--plot-out", default=None, help="Also write a plot with cut lines to this path (.png)")
    rank_gt.add_argument("--heatmap-out", default=None, help="Also write coalition×feature heatmap to this path (.png)")
    rank_gt.add_argument("--bins", type=int, default=50, help="Histogram bins for --plot-out (default: 50)")
    rank_gt.add_argument("--title", default=None, help="Figure title for --plot-out")
    rank_gt.add_argument("--empty-baseline-csv", default=None, help="Optional empty_baseline.csv to overlay as a vertical line on --plot-out")

    pipe = subparsers.add_parser("run-dataset-pipeline", help="Run baseline/ranking/visualization pipeline for one dataset")
    pipe.add_argument("--dataset-config", required=True)
    pipe.add_argument("--labeling-config", default=None, help="Required unless --skip-preprocess")
    pipe.add_argument("--game-table-config", required=True)
    pipe.add_argument("--task", default="baseline", choices=["baseline", "next-rank"])
    pipe.add_argument("--k-max", type=int, default=20)
    pipe.add_argument("--criterion", default="bic", choices=["bic", "mdl"])
    pipe.add_argument("--ranked-format", default="score-only", choices=["score-only", "full"])
    pipe.add_argument("--heatmap-top-n", type=int, default=200, help="Top N coalitions for heatmap (0=all; default: 200)")
    pipe.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output for build-game-table")
    pipe.add_argument("--skip-preprocess", action="store_true")
    pipe.add_argument("--skip-game-table", action="store_true", help="Skip build-game-table (assumes cache_path exists)")
    pipe.add_argument("--dry-run", action="store_true", help="Print planned outputs/steps only")

    build_patterns = subparsers.add_parser("build-pattern-table", help="Build a wide 0/1 table of all coalition patterns (no training)")
    build_patterns.add_argument("--dataset-config", default=default_dataset_cfg)
    build_patterns.add_argument("--task", default="baseline", choices=["baseline", "next-rank"])
    build_patterns.add_argument("--max-order", type=int, default=None, help="Max coalition size (default: all features)")
    build_patterns.add_argument("--out", default=None, help="Write to this path (.csv; default: outputs/<dataset_key>/game_tables/patterns.csv)")

    train_model = subparsers.add_parser("train", help="Train a baseline model and save it (XGBoost + preprocessing)")
    train_model.add_argument("--dataset-config", default=default_dataset_cfg)
    train_model.add_argument("--task", default="baseline", choices=["baseline", "next-rank"])
    train_model.add_argument("--out-model", default=None, help="Write model pickle to this path (default: outputs/<dataset_key>/models/model.pkl)")
    train_model.add_argument("--test-size", type=float, default=0.25)
    train_model.add_argument("--seed", type=int, default=42)

    resign_rate = subparsers.add_parser("build-resignation-rate-table", help="Build a coalition->resignation_rate table from raw data (no training)")
    resign_rate.add_argument("--dataset-config", default=default_dataset_cfg)
    resign_rate.add_argument("--out", default=None, help="Write to this path (.csv; default: outputs/<dataset_key>/game_tables/resignation_rate_table.csv)")
    resign_rate.add_argument("--include-empty", action="store_true", help="Include the empty coalition row (baseline resignation rate)")

    crop_policy = subparsers.add_parser(
        "build-crop-policy-game-table",
        help="Build a 4-policy (16 patterns) mean-yield table from crop raw data (no training)",
    )
    crop_policy.add_argument("--dataset-config", default="configs/datasets/crop/dataset.yml")
    crop_policy.add_argument(
        "--out",
        default=None,
        help="Write to this path (.csv; default: outputs/<dataset_key>/game_tables/game_table.csv)",
    )
    crop_policy.add_argument("--crop", default=None, help="Override crop selection (default: most frequent crop)")

    analyze_ds = subparsers.add_parser("analyze-dataset", help="Analyze features/target and write reports under outputs/")
    analyze_ds.add_argument("--dataset-config", default=default_dataset_cfg)
    analyze_ds.add_argument("--task", default="baseline", choices=["baseline", "next-rank"])
    analyze_ds.add_argument("--out-dir", default=None, help="Output directory (default: outputs/<dataset_key>/analysis/<task>/)")
    analyze_ds.add_argument("--configs-root", default="configs/datasets", help="Used to infer dataset_key from --dataset-config path")
    analyze_ds.add_argument("--top-k-corr-pairs", type=int, default=200, help="Max numeric Spearman correlation pairs to write")

    analyze_all = subparsers.add_parser("analyze-all-datasets", help="Analyze all datasets under configs/datasets/**/dataset.yml")
    analyze_all.add_argument("--configs-root", default="configs/datasets")
    analyze_all.add_argument("--task", default="baseline", choices=["baseline", "next-rank"])
    analyze_all.add_argument("--out-root", default=None, help="Override output root (default: outputs/)")
    analyze_all.add_argument("--continue-on-error", action="store_true", help="Continue even if one dataset fails")
    analyze_all.add_argument("--top-k-corr-pairs", type=int, default=200)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "preprocess":
        run_preprocess(Path(args.dataset_config), Path(args.labeling_config))
        return

    if args.command == "build-game-table":
        run_build_game_table(
            dataset_config=Path(args.dataset_config),
            game_table_config=Path(args.game_table_config),
            task=str(args.task),
            out=args.out,
            no_progress=bool(args.no_progress),
        )
        return

    if args.command == "build-empty-baseline":
        run_build_empty_baseline(
            dataset_config=Path(args.dataset_config),
            game_table_config=Path(args.game_table_config),
            task=str(args.task),
            out=args.out,
        )
        return

    if args.command == "plot-game-table-scores":
        run_plot_game_table_scores(
            game_table_csv=Path(args.game_table_csv),
            out=args.out,
            score_col=str(args.score_col),
            bins=int(args.bins),
            title=args.title,
            empty_baseline_csv=args.empty_baseline_csv,
            ranked_csv=args.ranked_csv,
        )
        return

    if args.command == "plot-game-table-heatmap":
        game_table_csv = Path(args.game_table_csv)
        if args.out:
            out_path = Path(args.out)
        else:
            out_path = game_table_csv.with_name(f"{game_table_csv.stem}__heatmap.png")
        top_n = int(args.top_n)
        plot_coalition_feature_heatmap(
            game_table_csv=game_table_csv,
            out_path=out_path,
            score_col=str(args.score_col),
            top_n=None if top_n == 0 else top_n,
            title=args.title,
            ranked_csv=args.ranked_csv,
        )
        print(str(out_path))
        return

    if args.command == "rank-game-table":
        run_rank_game_table(
            game_table_csv=Path(args.game_table_csv),
            out_ranked=args.out_ranked,
            out_classes=args.out_classes,
            out_summary=args.out_summary,
            score_col=str(args.score_col),
            id_col=str(args.id_col),
            k_max=args.k_max,
            k_fixed=args.k_fixed,
            criterion=args.criterion,  # type: ignore[arg-type]
            ranked_format=str(args.ranked_format),
            plot_out=args.plot_out,
            heatmap_out=args.heatmap_out,
            empty_baseline_csv=args.empty_baseline_csv,
            bins=int(args.bins),
            title=args.title,
        )
        return

    if args.command == "run-dataset-pipeline":
        run_dataset_pipeline(
            dataset_config=Path(args.dataset_config),
            labeling_config=Path(args.labeling_config) if args.labeling_config else None,
            game_table_config=Path(args.game_table_config),
            task=str(args.task),
            k_max=int(args.k_max),
            criterion=str(args.criterion),
            ranked_format=str(args.ranked_format),
            heatmap_top_n=int(args.heatmap_top_n),
            no_progress=bool(args.no_progress),
            skip_preprocess=bool(args.skip_preprocess),
            skip_game_table=bool(args.skip_game_table),
            dry_run=bool(args.dry_run),
        )
        return

    if args.command == "build-pattern-table":
        run_build_pattern_table(
            dataset_config=Path(args.dataset_config),
            task=str(args.task),
            out=args.out,
            max_order=args.max_order,
        )
        return

    if args.command == "train":
        run_train_model(
            dataset_config=Path(args.dataset_config),
            task=str(args.task),
            out_model=args.out_model,
            test_size=float(args.test_size),
            seed=int(args.seed),
        )
        return

    if args.command == "build-resignation-rate-table":
        run_build_resignation_rate_table(
            dataset_config=Path(args.dataset_config),
            out=args.out,
            include_empty=bool(args.include_empty),
        )
        return

    if args.command == "build-crop-policy-game-table":
        run_build_crop_policy_game_table(dataset_config=Path(args.dataset_config), out=args.out, crop=args.crop)
        return

    if args.command == "analyze-dataset":
        run_analyze_dataset(
            dataset_config=Path(args.dataset_config),
            task=str(args.task),
            out_dir=args.out_dir,
            configs_root=str(args.configs_root),
            top_k_corr_pairs=int(args.top_k_corr_pairs),
        )
        return

    if args.command == "analyze-all-datasets":
        run_analyze_all_datasets(
            configs_root=Path(args.configs_root),
            task=str(args.task),
            out_root=args.out_root,
            continue_on_error=bool(args.continue_on_error),
            top_k_corr_pairs=int(args.top_k_corr_pairs),
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
