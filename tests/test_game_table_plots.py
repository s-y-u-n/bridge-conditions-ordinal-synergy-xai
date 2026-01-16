from pathlib import Path

import pandas as pd

from bci_osxai.analysis.game_table_plots import plot_score_distribution


def test_plot_score_distribution_writes_png(tmp_path: Path) -> None:
    gt = tmp_path / "game_table.csv"
    df = pd.DataFrame(
        {
            "f1": [0, 1, 0, 1],
            "order": [1, 1, 1, 1],
            "value": [0.1, 0.2, 0.3, 0.4],
            "abs_value": [0.1, 0.2, 0.3, 0.4],
            "metric": ["accuracy"] * 4,
            "n_train": [10] * 4,
            "n_test": [5] * 4,
            "seed": [42] * 4,
        }
    )
    df.to_csv(gt, index=False)

    out = tmp_path / "plot.png"
    summary = plot_score_distribution(game_table_csv=gt, out_path=out, bins=5, title="t")
    assert out.exists()
    assert out.stat().st_size > 0
    assert summary["summary"]["n_used"] == 4

