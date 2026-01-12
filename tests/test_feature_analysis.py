from __future__ import annotations

from pathlib import Path

import pandas as pd

from bci_osxai.analysis.dataset_analysis import analyze_dataset, infer_dataset_key
from bci_osxai.analysis.stats import correlation_ratio, cramers_v


def test_infer_dataset_key() -> None:
    key = infer_dataset_key("configs/datasets/bridge_conditions/experiments/baseline10/dataset.yml", configs_root="configs/datasets")
    assert key == "bridge_conditions__experiments__baseline10"


def test_cramers_v_basic() -> None:
    x = pd.Series(["a", "a", "b", "b"])
    y = pd.Series([0, 0, 1, 1])
    v = cramers_v(x, y)
    assert 0.9 <= v <= 1.0


def test_correlation_ratio_basic() -> None:
    c = pd.Series(["a", "a", "b", "b"])
    v = pd.Series([0.0, 0.0, 10.0, 10.0])
    eta = correlation_ratio(c, v)
    assert 0.9 <= eta <= 1.0


def test_analyze_dataset_writes_outputs(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs" / "datasets"
    dataset_dir = configs_root / "toy_ds"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = tmp_path / "data" / "processed" / "toy_ds"
    processed_dir.mkdir(parents=True, exist_ok=True)

    features = pd.DataFrame(
        {
            "structure_id": ["1", "2", "3", "4"],
            "x_num": [0.1, 0.2, 0.3, 0.4],
            "x_cat": ["a", "a", "b", "b"],
        }
    )
    labels = pd.DataFrame({"structure_id": ["1", "2", "3", "4"], "label": ["yes", "yes", "no", "no"], "label_index": [pd.NA] * 4})

    features_path = processed_dir / "features.csv"
    labels_path = processed_dir / "labels.csv"
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)

    dataset_cfg = dataset_dir / "dataset.yml"
    dataset_cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  features_csv: {features_path.as_posix()}",
                f"  labels_csv: {labels_path.as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs = analyze_dataset(
        dataset_config=dataset_cfg,
        task="baseline",
        out_dir=tmp_path / "out",
        configs_root=configs_root,
        top_k_corr_pairs=50,
    )
    assert outputs.out_dir.exists()
    assert outputs.overview_json.exists()
    assert outputs.features_summary_csv.exists()
    assert outputs.target_summary_csv.exists()
    assert outputs.feature_target_assoc_csv.exists()
    assert outputs.numeric_correlations_spearman_csv.exists()
    assert outputs.report_md.exists()
