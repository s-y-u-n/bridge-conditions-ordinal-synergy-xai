from pathlib import Path

import pandas as pd

from bci_osxai.analysis.coalition_heatmap import plot_coalition_feature_heatmap


def test_plot_coalition_feature_heatmap_writes_png(tmp_path: Path) -> None:
    gt = tmp_path / "game_table.csv"
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "f2": [1, 1, 0, 0],
            "order": [1, 2, 0, 1],
            "value": [0.1, 0.9, 0.05, 0.4],
            "abs_value": [0.1, 0.9, 0.05, 0.4],
            "metric": ["accuracy"] * 4,
            "n_train": [10] * 4,
            "n_test": [5] * 4,
            "seed": [42] * 4,
        }
    )
    df.to_csv(gt, index=False)

    out = tmp_path / "heat.png"
    plot_coalition_feature_heatmap(game_table_csv=gt, out_path=out, top_n=0)
    assert out.exists()
    assert out.stat().st_size > 0

