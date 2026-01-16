from bci_osxai.pipeline.dataset_pipeline import infer_pipeline_paths


def test_infer_pipeline_paths_wine() -> None:
    paths = infer_pipeline_paths(dataset_config="configs/datasets/wine/dataset.yml", game_table_config="configs/datasets/wine/game_table.yml")
    assert paths.dataset_key == "wine"
    assert str(paths.game_table_csv).endswith("outputs/wine/game_tables/game_table.csv")
    assert str(paths.empty_baseline_csv).endswith("outputs/wine/game_tables/empty_baseline.csv")
    assert str(paths.ranked_csv).endswith("outputs/wine/game_tables/game_table__ranked.csv")

