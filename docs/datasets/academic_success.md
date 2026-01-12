# Dataset: academic_success

## 概要

Higher Education predictors of student retention / academic success（Kaggle）を、このリポジトリの標準パイプライン（`preprocess` → `analyze-dataset` → `build-game-table`）で扱えるようにする。

- デモグラフィック、社会経済要因、入学時情報、学期ごとの学業成績、地域経済指標（失業率/インフレ/GDP）などを含む
- 退学（Dropout）や学業成功（Graduate/Enrolled）に寄与する要因分析に使える

参照:

- https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

## 入力

- `data/raw/academic_success/dataset.csv`
- 目的変数: `Target`（`Dropout` / `Enrolled` / `Graduate`）

※ 多くの列はカテゴリを整数で符号化した値（例: `Marital status` など）として格納される。

## 出力（CSV固定）

`preprocess` により `data/processed/academic_success/` に以下を出力する。

- `structures.csv`: 元テーブルに `structure_id` を付与したもの
- `features.csv`: 学習用特徴量（`structure_id` を含む、`Target` は含めない）
- `labels.csv`: ラベル（`structure_id`, `label`, `label_index`）

## 設定

- `configs/datasets/academic_success/dataset.yml`
  - `io.encoding: utf-8-sig`（BOM付きCSV対策）
  - `columns.target: Target`
- `configs/datasets/academic_success/labeling.yml`
  - `scheme.type: column`（`Target` をそのままラベルとして使う）
- `configs/datasets/academic_success/game_table.yml`
  - 出力: `outputs/academic_success/game_tables/game_table.csv`

## 実行例

```bash
poetry run bci-xai preprocess \
  --dataset-config configs/datasets/academic_success/dataset.yml \
  --labeling-config configs/datasets/academic_success/labeling.yml

poetry run bci-xai analyze-dataset \
  --dataset-config configs/datasets/academic_success/dataset.yml \
  --task baseline

poetry run bci-xai build-game-table \
  --dataset-config configs/datasets/academic_success/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/academic_success/game_table.yml
```

