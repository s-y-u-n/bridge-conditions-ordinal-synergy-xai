import pandas as pd

from bci_osxai.ranking.weak_order import rank_to_weak_order


def test_rank_to_weak_order_partitions_and_orders() -> None:
    df = pd.DataFrame(
        {
            "coalition_id": [f"c{i}" for i in range(6)],
            "value": [10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
        }
    )
    ranked, classes, summary = rank_to_weak_order(df, id_col="coalition_id", score_col="value", k_max=3, criterion="bic")

    assert summary["k_selected"] == 2
    assert len(classes) == 2
    assert set(sum([c["members"] for c in classes], [])) == set(df["coalition_id"].tolist())

    # class_id=1 is higher (scores >= class_id=2)
    max_low = float(ranked.loc[ranked["class_id"] == 2, "value"].max())
    min_high = float(ranked.loc[ranked["class_id"] == 1, "value"].min())
    assert min_high >= max_low


def test_rank_to_weak_order_k_fixed_overrides_selection() -> None:
    df = pd.DataFrame({"coalition_id": [f"c{i}" for i in range(8)], "value": [1.0, 1.0, 0.9, 0.9, 0.5, 0.5, 0.0, 0.0]})
    ranked, classes, summary = rank_to_weak_order(df, id_col="coalition_id", score_col="value", k_max=4, k_fixed=3, criterion="bic")
    assert summary["k_selected"] == 3
    assert summary["k_fixed"] == 3
    assert len(classes) == 3
    assert ranked["class_id"].min() == 1
    assert ranked["class_id"].max() == 3
