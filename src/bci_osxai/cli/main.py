from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from bci_osxai.features.build_features import build_features
from bci_osxai.game_table import GameTableSettings, build_game_table, save_game_table
from bci_osxai.io.load_raw import load_raw_arff, load_raw_csv
from bci_osxai.labels.make_column_labels import make_column_labels
from bci_osxai.labels.make_continuous_targets import make_continuous_targets
from bci_osxai.labels.make_ordinal_labels import make_ordinal_labels
from bci_osxai.labels.make_rehab_floor_rank_labels import RehabFloorRankConfig, make_rehab_floor_rank_labels
from bci_osxai.preprocess.build_next_rank_dataset import NextRankConfig, build_next_rank_dataset
from bci_osxai.preprocess.clean_raw_csv import clean_multiline_csv, load_rename_map_from_data_dictionary
from bci_osxai.preprocess.clean_schema import normalize_column_whitespace
from bci_osxai.preprocess.reshape_bci import build_bci_long


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
        next_path = paths.get("next_rank_dataset_parquet")
        if not next_path:
            raise SystemExit("dataset.paths.next_rank_dataset_parquet is required for --task next-rank.")
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


def run_preprocess(dataset_config: Path, labeling_config: Path) -> None:
    dataset_cfg = load_yaml(dataset_config)
    labeling_cfg = load_yaml(labeling_config)

    paths = dataset_cfg["paths"]
    columns = (dataset_cfg.get("columns") or {}) if isinstance(dataset_cfg, dict) else {}
    encoding = (dataset_cfg.get("io") or {}).get("encoding")
    rename_columns = bool((dataset_cfg.get("io") or {}).get("rename_columns", False))
    io_format = str((dataset_cfg.get("io") or {}).get("format") or "csv").strip().lower()

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

    if io_format == "arff":
        df = load_raw_arff(raw_path)
    elif io_format == "csv":
        df = load_raw_csv(raw_path, encoding=encoding)
    else:
        raise SystemExit(f"Unsupported io.format: {io_format!r} (expected: csv|arff)")
    df = normalize_column_whitespace(df)

    filters = dataset_cfg.get("filters", {}) or {}
    for key, value in filters.items():
        column_name = columns.get(key)
        if column_name and column_name in df.columns:
            df = df[df[column_name] == value]

    id_source = columns.get("id")
    df = df.copy()
    if id_source is not None and str(id_source).strip() != "" and str(id_source) in df.columns:
        df["structure_id"] = df[str(id_source)].astype(str)
    else:
        df["structure_id"] = pd.Series(range(1, len(df) + 1), dtype="int64").astype(str)

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
        labels = make_column_labels(df, id_col="structure_id", target_col=bci_col)
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

    next_rank_path = paths.get("next_rank_dataset_parquet")
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
        next_dataset.to_parquet(next_rank_path, index=False)

    if paths.get("structures_parquet"):
        Path(paths["structures_parquet"]).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(paths["structures_parquet"], index=False)
    if bci_long is not None and paths.get("bci_long_parquet"):
        Path(paths["bci_long_parquet"]).parent.mkdir(parents=True, exist_ok=True)
        bci_long.to_parquet(paths["bci_long_parquet"], index=False)
    if paths.get("features_parquet"):
        Path(paths["features_parquet"]).parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(paths["features_parquet"], index=False)
    if paths.get("labels_parquet"):
        Path(paths["labels_parquet"]).parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(paths["labels_parquet"], index=False)


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
    settings = GameTableSettings(
        enabled=True,
        max_order=int(game_cfg.get("max_order", 2)),
        n_coalitions=int(game_cfg.get("n_coalitions", 400)),
        test_size=float(game_cfg.get("test_size", 0.25)),
        metric=str(game_cfg.get("metric", "accuracy")),
        seed=int(game_cfg.get("seed", 42)),
    )
    table = build_game_table(X=X, y=y, players=list(X.columns), settings=settings, progress=not bool(no_progress))
    save_game_table(table, out_path)
    print(str(out_path))


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

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

