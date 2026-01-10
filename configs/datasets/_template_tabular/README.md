# Template: tabular dataset

別データセットを追加する場合は、このフォルダを `configs/datasets/<dataset_id>/` としてコピーし、`dataset.yml` / `labeling.yml` / `game_table.yml` を埋めてください。

このテンプレートは「入力CSVがすでに 1行=1サンプル」の教師あり学習を想定します（ID列とターゲット列を指定）。
