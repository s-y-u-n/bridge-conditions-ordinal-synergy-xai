from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from bci_osxai.features.build_features import build_features
from bci_osxai.io.load_raw import load_raw_csv
from bci_osxai.labels.make_continuous_targets import make_continuous_targets
from bci_osxai.labels.make_ordinal_labels import make_ordinal_labels
from bci_osxai.labels.make_rehab_floor_rank_labels import RehabFloorRankConfig, make_rehab_floor_rank_labels
from bci_osxai.models.predict import load_model, predict
from bci_osxai.models.train import save_model, train_xgb_classifier, train_xgb_regressor
from bci_osxai.preprocess.clean_raw_csv import clean_multiline_csv, load_rename_map_from_data_dictionary
from bci_osxai.preprocess.build_next_rank_dataset import (
    NextRankConfig,
    build_latest_features_for_next_rank,
    build_next_rank_dataset,
)
from bci_osxai.preprocess.clean_schema import normalize_column_whitespace
from bci_osxai.preprocess.reshape_bci import build_bci_long
from bci_osxai.synergy.report import build_synergy_report
from bci_osxai.synergy.game_table import GameTableSettings, build_game_table, load_game_table, save_game_table
from bci_osxai.synergy.lexcel import LexcelSettings, lexcel_ranking
from bci_osxai.synergy.visualize import (
    plot_interactions_bar,
)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_structure_id(dataset_config: Path, structure_id: str | None) -> str:
    if structure_id is not None and str(structure_id).strip() != "":
        return str(structure_id)
    dataset_cfg = load_yaml(dataset_config)
    default_id = (dataset_cfg.get("structure_id") or {}).get("default")
    if default_id is None or str(default_id).strip() == "":
        raise SystemExit(
            "structure_id is required; pass `--id <structure_id>` or set `structure_id.default` in the dataset config."
        )
    return str(default_id)


def top_sets_from_interactions_table(table: pd.DataFrame, *, top_k: int, min_order: int, max_order: int) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if table is None or table.empty:
        return results
    filtered = table[(table["order"] >= int(min_order)) & (table["order"] <= int(max_order))].copy()
    if filtered.empty:
        return results
    filtered = filtered.sort_values("abs_value", ascending=False).head(int(top_k))
    for _, row in filtered.iterrows():
        feat = [x for x in str(row["coalition_key"]).split("|") if x]
        results.append(
            {
                "set": feat,
                "order": int(row["order"]),
                "value": float(row["value"]),
                "abs_value": float(row["abs_value"]),
            }
        )
    return results


def load_task_dataset(dataset_config: Path, *, task: str) -> tuple[pd.DataFrame, pd.Series]:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]

    if task == "next-rank":
        next_path = paths.get("next_rank_dataset_parquet")
        if not next_path:
            raise SystemExit("next_rank_dataset_parquet is not configured in configs/dataset.yml")
        ds = pd.read_parquet(next_path)
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
        features = pd.read_parquet(paths["features_parquet"])
        labels = pd.read_parquet(paths["labels_parquet"])
        merged = features.merge(labels[["structure_id", "label"]], on="structure_id", how="inner")
        merged = merged[merged["label"].notna()].copy()
        X = merged.drop(columns=["structure_id", "label"], errors="ignore")
        y = merged["label"]
        return X, y

    raise ValueError(f"Unknown task: {task}")


def build_or_load_game_table(dataset_config: Path, synergy_config: Path) -> pd.DataFrame:
    synergy_cfg = load_yaml(synergy_config)
    game_cfg = (synergy_cfg.get("game_table", {}) or {}) if isinstance(synergy_cfg, dict) else {}
    if not bool(game_cfg.get("enabled", False)):
        raise SystemExit("game_table is disabled in synergy config.")
    cache_path = game_cfg.get("cache_path")
    if not cache_path:
        raise SystemExit("game_table.cache_path is required in synergy config.")

    cache_file = Path(str(cache_path))
    if cache_file.exists():
        return load_game_table(cache_file)

    if not bool(game_cfg.get("auto_build", False)):
        raise SystemExit(f"game table not found: {cache_file} (run `bci-xai build-game-table` or set game_table.auto_build: true)")

    X, y = load_task_dataset(dataset_config, task=str(game_cfg.get("task", "next-rank")))
    settings = GameTableSettings(
        enabled=True,
        max_order=int(game_cfg.get("max_order", 2)),
        n_coalitions=int(game_cfg.get("n_coalitions", 400)),
        test_size=float(game_cfg.get("test_size", 0.25)),
        metric=str(game_cfg.get("metric", "accuracy")),
        seed=int(game_cfg.get("seed", 42)),
    )
    table = build_game_table(X=X, y=y, players=list(X.columns), settings=settings)
    save_game_table(table, cache_file)
    return table


def write_report(report: Dict[str, Any], out_path: str | Path | None, stdout: bool) -> None:
    text = yaml.safe_dump(report, sort_keys=False)
    if stdout:
        print(text)
        return
    if out_path is None:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def run_preprocess(dataset_config: Path, labeling_config: Path) -> None:
    dataset_cfg = load_yaml(dataset_config)
    labeling_cfg = load_yaml(labeling_config)

    paths = dataset_cfg["paths"]
    columns = dataset_cfg["columns"]
    encoding = dataset_cfg.get("io", {}).get("encoding")
    rename_columns = bool(dataset_cfg.get("io", {}).get("rename_columns", False))

    raw_csv = paths["raw_csv"]
    clean_csv = paths.get("raw_clean_csv")
    if clean_csv:
        rename_map = None
        if rename_columns:
            year_cfg = dataset_cfg.get("bci_years", {})
            rename_map = load_rename_map_from_data_dictionary(
                paths.get("data_dictionary", "docs/data_dictionary_bridge_conditions.md"),
                year_start=int(year_cfg.get("start", 2000)),
                year_end=int(year_cfg.get("end", 2020)),
            )
        clean_multiline_csv(raw_csv, clean_csv, encoding=encoding or "utf-8", rename_map=rename_map)
        raw_csv = clean_csv

    df = load_raw_csv(raw_csv, encoding=encoding)
    df = normalize_column_whitespace(df)

    filters = dataset_cfg.get("filters", {})
    for key, value in filters.items():
        column_name = columns.get(key)
        if column_name and column_name in df.columns:
            df = df[df[column_name] == value]

    year_cfg = dataset_cfg.get("bci_years", {})
    start = int(year_cfg.get("start", 2000))
    end = int(year_cfg.get("end", 2020))
    col_format = str(year_cfg.get("col_format", "{year}"))
    year_cols = [col_format.format(year=year) for year in range(start, end + 1)]

    bci_long = build_bci_long(df, columns["id"], year_cols, columns["current_bci"])

    inspection_year = dataset_cfg.get("inspection_year", {}).get("default")
    features = build_features(df, columns, inspection_year=inspection_year)
    include = (dataset_cfg.get("feature_columns") or {}).get("include")
    if include:
        include_set = {str(c) for c in include}
        include_set.add("structure_id")
        features = features[[c for c in features.columns if c in include_set]].copy()

    scheme = labeling_cfg["active_scheme"]
    scheme_cfg = labeling_cfg["schemes"][scheme]
    scheme_type = scheme_cfg.get("type", "fixed_thresholds")
    target_cfg = labeling_cfg.get("target", {})
    if target_cfg.get("source") == "year":
        bci_col = year_cfg.get("col_format", "{year}").format(year=int(target_cfg.get("year")))
    else:
        bci_col = target_cfg.get("column", columns["current_bci"])

    if scheme_type == "continuous_bci":
        labels = make_continuous_targets(df, id_col=columns["id"], bci_col=bci_col)
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
            id_col=columns["id"],
            target_bci_col=bci_col,
            bci_year_cols=year_cols,
            config=config,
        )
    else:
        thresholds = scheme_cfg["labels"]
        labels = make_ordinal_labels(df, columns["id"], bci_col, thresholds)

    # Optional: build next-inspection rank dataset when a rank scheme is active.
    next_rank_path = paths.get("next_rank_dataset_parquet")
    if next_rank_path and scheme_type == "rehab_floor_ranks":
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
        next_dataset.to_parquet(next_rank_path, index=False)

    Path(paths["structures_parquet"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths["structures_parquet"], index=False)
    bci_long.to_parquet(paths["bci_long_parquet"], index=False)
    features.to_parquet(paths["features_parquet"], index=False)
    labels.to_parquet(paths["labels_parquet"], index=False)


def run_train(dataset_config: Path, model_out: Path) -> None:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]

    features = pd.read_parquet(paths["features_parquet"])
    labels = pd.read_parquet(paths["labels_parquet"])

    merged = features.merge(labels[["structure_id", "label"]], on="structure_id", how="inner")
    merged = merged[merged["label"].notna()].copy()
    X = merged.drop(columns=["structure_id", "label"], errors="ignore")
    y = merged["label"]

    if pd.api.types.is_numeric_dtype(y):
        model = train_xgb_regressor(X, y)
    else:
        model = train_xgb_classifier(X, y)
    save_model(model, model_out)


def run_train_next_rank(dataset_config: Path, model_out: Path) -> None:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]
    next_path = paths.get("next_rank_dataset_parquet")
    if not next_path:
        raise SystemExit("next_rank_dataset_parquet is not configured in configs/dataset.yml")

    ds = pd.read_parquet(next_path)
    ds = ds[ds["label"].notna()].copy()
    drop_cols = {
        "structure_id",
        "label",
        "label_index",
        "target_rank",
        "target_rank_index",
        "target_bci",
        "year_next",
        "current_rank",  # redundant with current_rank_index
    }
    X = ds.drop(columns=[c for c in drop_cols if c in ds.columns], errors="ignore")
    y = ds["label"]

    model = train_xgb_classifier(X, y)
    save_model(model, model_out)


def _plot_next_rank_proba(classes: list[str], proba: list[float], output_path: Path, title: str) -> None:
    from bci_osxai.utils.caches import ensure_writable_caches  # noqa: PLC0415

    ensure_writable_caches()

    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(classes)), key=lambda i: proba[i], reverse=True)
    classes_sorted = [classes[i] for i in order]
    proba_sorted = [proba[i] for i in order]

    fig_h = max(3.0, 0.45 * len(classes_sorted) + 1.0)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(range(len(classes_sorted)), proba_sorted, color="#1f77b4")
    ax.set_yticks(range(len(classes_sorted)))
    ax.set_yticklabels(classes_sorted)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted probability")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_predict_next_rank(
    dataset_config: Path,
    model_path: Path,
    structure_id: str,
    *,
    plot: bool = False,
    out: str | Path | None = None,
    stdout: bool = False,
) -> None:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]

    structures = pd.read_parquet(paths["structures_parquet"])
    bci_long = pd.read_parquet(paths["bci_long_parquet"])

    labeling_cfg = load_yaml("configs/labeling_rank5.yml") if Path("configs/labeling_rank5.yml").exists() else load_yaml("configs/labeling.yml")
    scheme = labeling_cfg["active_scheme"]
    scheme_cfg = labeling_cfg["schemes"][scheme]
    if scheme_cfg.get("type") != "rehab_floor_ranks":
        raise SystemExit("Next-rank prediction requires a rehab_floor_ranks labeling scheme (use configs/labeling_rank5.yml).")

    dist = scheme_cfg.get("distribution", {}) or {}
    next_cfg = NextRankConfig(
        n_ranks=int(scheme_cfg.get("n_ranks", 5)),
        increase_delta=float(scheme_cfg.get("increase_delta", 0.5)),
        label_prefix=str(scheme_cfg.get("label_prefix", "R")),
        distribution_name=str(dist.get("name", "beta")),
        alpha=float(dist.get("alpha", 5.0)),
        beta=float(dist.get("beta", 2.0)),
    )

    X = build_latest_features_for_next_rank(structures=structures, bci_long=bci_long, structure_id=str(structure_id), config=next_cfg)
    X_model = X.drop(columns=["structure_id", "current_rank"], errors="ignore")

    model = load_model(model_path)
    pred = str(predict(model, X_model).iloc[0])

    safe_id = str(structure_id).replace("/", "_")
    proba = None
    classes = None
    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        proba = model.predict_proba(X_model)[0].tolist()
        classes = [str(c) for c in model.classes_]
        if plot:
            _plot_next_rank_proba(
                classes=classes,
                proba=proba,
                output_path=Path("artifacts") / "reports" / f"{safe_id}_next_rank_proba.png",
                title=f"{structure_id} next-rank probabilities",
            )

    report = {
        "structure_id": str(structure_id),
        "current_year": int(X["year"].iloc[0]),
        "current_bci": float(X["current_bci"].iloc[0]),
        "current_rank": str(X["current_rank"].iloc[0]),
        "predicted_next_rank": pred,
    }
    if proba is not None and classes is not None:
        report["predicted_next_rank_proba"] = {c: float(p) for c, p in zip(classes, proba)}

    if out is None and not stdout:
        out = Path("artifacts") / "reports" / f"{safe_id}_next_rank_prediction.yml"
    write_report(report, out, stdout)


def run_explain_next_rank(
    dataset_config: Path,
    model_path: Path,
    structure_id: str,
    *,
    plot: bool = False,
    synergy_config: str | Path = "configs/synergy.yml",
    out: str | Path | None = None,
    stdout: bool = False,
) -> None:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]
    next_path = paths.get("next_rank_dataset_parquet")
    if not next_path:
        raise SystemExit("next_rank_dataset_parquet is not configured in configs/dataset.yml")

    structures = pd.read_parquet(paths["structures_parquet"])
    bci_long = pd.read_parquet(paths["bci_long_parquet"])
    next_ds = pd.read_parquet(next_path)

    labeling_cfg = load_yaml("configs/labeling_rank5.yml") if Path("configs/labeling_rank5.yml").exists() else load_yaml("configs/labeling.yml")
    scheme = labeling_cfg["active_scheme"]
    scheme_cfg = labeling_cfg["schemes"][scheme]
    if scheme_cfg.get("type") != "rehab_floor_ranks":
        raise SystemExit("Next-rank explain requires a rehab_floor_ranks labeling scheme (use configs/labeling_rank5.yml).")

    dist = scheme_cfg.get("distribution", {}) or {}
    next_cfg = NextRankConfig(
        n_ranks=int(scheme_cfg.get("n_ranks", 5)),
        increase_delta=float(scheme_cfg.get("increase_delta", 0.5)),
        label_prefix=str(scheme_cfg.get("label_prefix", "R")),
        distribution_name=str(dist.get("name", "beta")),
        alpha=float(dist.get("alpha", 5.0)),
        beta=float(dist.get("beta", 2.0)),
    )

    X_row = build_latest_features_for_next_rank(structures=structures, bci_long=bci_long, structure_id=str(structure_id), config=next_cfg)
    X_model = X_row.drop(columns=["structure_id", "current_rank"], errors="ignore")

    model = load_model(model_path)
    pred = str(predict(model, X_model).iloc[0])

    safe_id = str(structure_id).replace("/", "_")
    synergy_cfg = load_yaml(synergy_config)
    top_k = int(synergy_cfg.get("report", {}).get("top_k", 5))
    game_table = build_or_load_game_table(dataset_config, Path(synergy_config))

    # Treat game_table as a coalition preorder table.
    # Keep the report shape compatible with earlier outputs.
    max_order = int(pd.to_numeric(game_table["order"], errors="coerce").max()) if not game_table.empty else 2
    top_sets = top_sets_from_interactions_table(table=game_table, top_k=top_k, min_order=2, max_order=max_order)

    lex_cfg = (synergy_cfg.get("lexcel", {}) or {}) if isinstance(synergy_cfg, dict) else {}
    lex_enabled = bool(lex_cfg.get("enabled", False))
    lexcel = None
    if lex_enabled:
        lex_max_order = lex_cfg.get("max_order")
        lex_settings = LexcelSettings(
            enabled=True,
            score_key=str(lex_cfg.get("score_key", "abs_value")),
            tie_tol=float(lex_cfg.get("tie_tol", 1.0e-12)),
            min_order=int(lex_cfg.get("min_order", 1)),
            max_order=int(lex_max_order) if lex_max_order is not None else max_order,
            max_items=int(lex_cfg.get("max_items", 20)),
        )
        lexcel = lexcel_ranking(game_table, players=list(X_model.columns), settings=lex_settings)

    if plot:
        plot_interactions_bar(
            structure_id=str(structure_id),
            predicted_label=pred,
            interactions=top_sets,
            output_path=Path("artifacts") / "reports" / f"{safe_id}_next_rank_interactions.png",
        )

    report = {
        "structure_id": str(structure_id),
        "current_year": int(X_row["year"].iloc[0]),
        "current_bci": float(X_row["current_bci"].iloc[0]),
        "current_rank": str(X_row["current_rank"].iloc[0]),
        "predicted_next_rank": pred,
        "synergy_top_k": top_sets,
    }
    if lexcel is not None:
        report["lexcel"] = lexcel
    if out is None and not stdout:
        out = Path("artifacts") / "reports" / f"{safe_id}_next_rank_explain.yml"
    write_report(report, out, stdout)


def run_explain(
    dataset_config: Path,
    model_path: Path,
    structure_id: str,
    *,
    plot: bool = False,
    synergy_config: str | Path = "configs/synergy.yml",
    out: str | Path | None = None,
    stdout: bool = False,
) -> None:
    dataset_cfg = load_yaml(dataset_config)
    paths = dataset_cfg["paths"]

    features = pd.read_parquet(paths["features_parquet"])
    row = features[features["structure_id"].astype(str) == str(structure_id)]
    if row.empty:
        raise SystemExit(f"structure_id not found: {structure_id}")

    model = load_model(model_path)
    X = row.drop(columns=["structure_id"], errors="ignore")
    prediction = str(predict(model, X).iloc[0])

    top_sets: list[dict[str, object]] = []
    lexcel = None
    try:
        synergy_cfg = load_yaml(synergy_config)
        top_k = int(synergy_cfg.get("report", {}).get("top_k", 5))
        game_table = build_or_load_game_table(dataset_config, Path(synergy_config))
        max_order = int(pd.to_numeric(game_table["order"], errors="coerce").max()) if not game_table.empty else 2
        top_sets = top_sets_from_interactions_table(table=game_table, top_k=top_k, min_order=2, max_order=max_order)

        lex_cfg = (synergy_cfg.get("lexcel", {}) or {}) if isinstance(synergy_cfg, dict) else {}
        lex_enabled = bool(lex_cfg.get("enabled", False))
        if lex_enabled:
            lex_max_order = lex_cfg.get("max_order")
            lex_settings = LexcelSettings(
                enabled=True,
                score_key=str(lex_cfg.get("score_key", "abs_value")),
                tie_tol=float(lex_cfg.get("tie_tol", 1.0e-12)),
                min_order=int(lex_cfg.get("min_order", 1)),
                max_order=int(lex_max_order) if lex_max_order is not None else max_order,
                max_items=int(lex_cfg.get("max_items", 20)),
            )
            lexcel = lexcel_ranking(game_table, players=list(X.columns), settings=lex_settings)
    except Exception:
        top_sets = []

    report = build_synergy_report(structure_id=str(structure_id), predicted_label=prediction, top_sets=top_sets, lexcel=lexcel)
    if plot:
        safe_id = str(structure_id).replace("/", "_")
        plot_interactions_bar(
            structure_id=str(structure_id),
            predicted_label=prediction,
            interactions=top_sets,
            output_path=Path("artifacts") / "reports" / f"{safe_id}_interactions.png",
        )
    if out is None and not stdout:
        safe_id = str(structure_id).replace("/", "_")
        out = Path("artifacts") / "reports" / f"{safe_id}_explain.yml"
    write_report(report, out, stdout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bci-xai")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Run preprocessing pipeline")
    preprocess.add_argument("--dataset-config", default="configs/dataset.yml")
    preprocess.add_argument("--labeling-config", default="configs/labeling.yml")

    train = subparsers.add_parser("train", help="Train baseline model")
    train.add_argument("--dataset-config", default="configs/dataset.yml")
    train.add_argument("--model-out", default="artifacts/model.pkl")

    train_next = subparsers.add_parser("train-next-rank", help="Train next-inspection rank model (from next_rank_dataset.parquet)")
    train_next.add_argument("--dataset-config", default="configs/dataset.yml")
    train_next.add_argument("--model-out", default="artifacts/next_rank_model.pkl")

    predict_next = subparsers.add_parser("predict-next-rank", help="Predict next-inspection rank for a structure using latest available BCI year")
    predict_next.add_argument("--dataset-config", default="configs/dataset.yml")
    predict_next.add_argument("--model", default="artifacts/next_rank_model.pkl")
    predict_next.add_argument("--id", default=None, help="structure_id (default: dataset config `structure_id.default`)")
    predict_next.add_argument("--plot", action="store_true", help="Save predicted class-probability bar plot to artifacts/reports/")
    predict_next.add_argument("--out", default=None, help="Write YAML report to this path (default: artifacts/reports/*)")
    predict_next.add_argument("--stdout", action="store_true", help="Print YAML to stdout instead of writing a file")

    explain_next = subparsers.add_parser("explain-next-rank", help="Explain next-inspection rank prediction with a cached game table")
    explain_next.add_argument("--dataset-config", default="configs/dataset.yml")
    explain_next.add_argument("--model", default="artifacts/next_rank_model.pkl")
    explain_next.add_argument("--id", default=None, help="structure_id (default: dataset config `structure_id.default`)")
    explain_next.add_argument("--plot", action="store_true", help="Save a simple interaction bar plot to artifacts/reports/")
    explain_next.add_argument("--synergy-config", default="configs/synergy.yml", help="Synergy/explanation config (default: configs/synergy.yml)")
    explain_next.add_argument("--out", default=None, help="Write YAML report to this path (default: artifacts/reports/*)")
    explain_next.add_argument("--stdout", action="store_true", help="Print YAML to stdout instead of writing a file")

    explain = subparsers.add_parser("explain", help="Explain a single structure")
    explain.add_argument("--dataset-config", default="configs/dataset.yml")
    explain.add_argument("--model", default="artifacts/model.pkl")
    explain.add_argument("--id", default=None, help="structure_id (default: dataset config `structure_id.default`)")
    explain.add_argument("--plot", action="store_true", help="Save a simple interaction bar plot to artifacts/reports/")
    explain.add_argument("--synergy-config", default="configs/synergy.yml", help="Synergy/explanation config (default: configs/synergy.yml)")
    explain.add_argument("--out", default=None, help="Write YAML report to this path (default: artifacts/reports/*)")
    explain.add_argument("--stdout", action="store_true", help="Print YAML to stdout instead of writing a file")

    build_gt = subparsers.add_parser("build-game-table", help="Build a coalition->score game table by retraining with feature masks")
    build_gt.add_argument("--dataset-config", default="configs/dataset.yml")
    build_gt.add_argument("--task", default="next-rank", choices=["next-rank", "baseline"])
    build_gt.add_argument("--synergy-config", default="configs/synergy.yml", help="Reads game_table.* settings from this file")
    build_gt.add_argument("--out", default=None, help="Write to this path (.parquet or .csv; default: game_table.cache_path)")
    build_gt.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(Path(args.dataset_config), Path(args.labeling_config))
    elif args.command == "train":
        run_train(Path(args.dataset_config), Path(args.model_out))
    elif args.command == "train-next-rank":
        run_train_next_rank(Path(args.dataset_config), Path(args.model_out))
    elif args.command == "predict-next-rank":
        structure_id = resolve_structure_id(Path(args.dataset_config), getattr(args, "id", None))
        run_predict_next_rank(
            Path(args.dataset_config),
            Path(args.model),
            structure_id,
            plot=bool(args.plot),
            out=args.out,
            stdout=bool(args.stdout),
        )
    elif args.command == "explain-next-rank":
        structure_id = resolve_structure_id(Path(args.dataset_config), getattr(args, "id", None))
        run_explain_next_rank(
            Path(args.dataset_config),
            Path(args.model),
            structure_id,
            plot=bool(args.plot),
            synergy_config=str(args.synergy_config),
            out=args.out,
            stdout=bool(args.stdout),
        )
    elif args.command == "explain":
        structure_id = resolve_structure_id(Path(args.dataset_config), getattr(args, "id", None))
        run_explain(
            Path(args.dataset_config),
            Path(args.model),
            structure_id,
            plot=bool(args.plot),
            synergy_config=str(args.synergy_config),
            out=args.out,
            stdout=bool(args.stdout),
        )
    elif args.command == "build-game-table":
        synergy_cfg = load_yaml(Path(args.synergy_config))
        game_cfg = (synergy_cfg.get("game_table", {}) or {}) if isinstance(synergy_cfg, dict) else {}
        default_out = game_cfg.get("cache_path")
        out_path = Path(args.out) if args.out else (Path(default_out) if default_out else None)
        if out_path is None:
            raise SystemExit("Output path is required: set game_table.cache_path in synergy config or pass --out.")
        X, y = load_task_dataset(Path(args.dataset_config), task=str(args.task))
        settings = GameTableSettings(
            enabled=True,
            max_order=int(game_cfg.get("max_order", 2)),
            n_coalitions=int(game_cfg.get("n_coalitions", 400)),
            test_size=float(game_cfg.get("test_size", 0.25)),
            metric=str(game_cfg.get("metric", "accuracy")),
            seed=int(game_cfg.get("seed", 42)),
        )
        table = build_game_table(X=X, y=y, players=list(X.columns), settings=settings, progress=not bool(args.no_progress))
        save_game_table(table, out_path)
        print(str(out_path))


if __name__ == "__main__":
    main()
