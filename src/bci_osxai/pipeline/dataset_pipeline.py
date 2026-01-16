from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from bci_osxai.analysis.dataset_analysis import infer_dataset_key


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML: {path}")
    return data


@dataclass(frozen=True)
class PipelinePaths:
    dataset_key: str
    game_table_csv: Path
    empty_baseline_csv: Path
    ranked_csv: Path
    classes_json: Path
    rank_summary_json: Path
    score_distribution_png: Path
    heatmap_png: Path


def infer_pipeline_paths(*, dataset_config: str | Path, game_table_config: str | Path) -> PipelinePaths:
    dataset_config = Path(dataset_config)
    game_table_cfg = _load_yaml(game_table_config)
    game = (game_table_cfg.get("game_table") or {}) if isinstance(game_table_cfg, dict) else {}
    cache_path = game.get("cache_path")
    if not cache_path:
        raise ValueError("game_table.cache_path is required in game_table config for the pipeline.")

    dataset_key = infer_dataset_key(dataset_config)
    game_table_csv = Path(str(cache_path))
    out_dir = game_table_csv.parent

    stem = game_table_csv.stem
    return PipelinePaths(
        dataset_key=dataset_key,
        game_table_csv=game_table_csv,
        empty_baseline_csv=out_dir / "empty_baseline.csv",
        ranked_csv=out_dir / f"{stem}__ranked.csv",
        classes_json=out_dir / f"{stem}__classes.json",
        rank_summary_json=out_dir / f"{stem}__rank_summary.json",
        score_distribution_png=out_dir / f"{stem}__score_distribution.png",
        heatmap_png=out_dir / f"{stem}__heatmap.png",
    )


@dataclass(frozen=True)
class DatasetPipelineConfig:
    dataset_config: Path
    labeling_config: Optional[Path]
    game_table_config: Path
    task: str = "baseline"
    k_max: int = 20
    criterion: str = "bic"
    ranked_format: str = "score-only"
    heatmap_top_n: int = 200

