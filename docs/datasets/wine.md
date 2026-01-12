# Dataset: wine

## 概要

UCI Machine Learning Repository の Wine データセット（3クラス分類）。化学分析値（13特徴量）からワインの `class`（1/2/3）を予測する。

## 入力

- `data/raw/wine/wine.data`（ヘッダなしCSV）
  - 1列目: `class`（目的変数; 1/2/3 だが分類として扱う）
  - 2列目以降: 特徴量（13列）

## 変数

| Variable Name | Role | Type |
|---|---|---|
| `class` | Target | Categorical |
| `Alcohol` | Feature | Continuous |
| `Malicacid` | Feature | Continuous |
| `Ash` | Feature | Continuous |
| `Alcalinity_of_ash` | Feature | Continuous |
| `Magnesium` | Feature | Integer |
| `Total_phenols` | Feature | Continuous |
| `Flavanoids` | Feature | Continuous |
| `Nonflavanoid_phenols` | Feature | Continuous |
| `Proanthocyanins` | Feature | Continuous |
| `Color_intensity` | Feature | Continuous |
| `Hue` | Feature | Continuous |
| `0D280_0D315_of_diluted_wines` | Feature | Continuous |
| `Proline` | Feature | Integer |

## 出力（CSV固定）

`preprocess` により `data/processed/wine/` に以下を出力する。

- `structures.csv`: 元テーブルに `structure_id` を付与したもの
- `features.csv`: 学習用特徴量（`structure_id` を含む、`class` は含めない）
- `labels.csv`: ラベル（`structure_id`, `label`, `label_index`）

ゲームテーブル:

- `outputs/wine/game_tables/game_table.csv`

## 設定

- `configs/datasets/wine/dataset.yml`
  - `io.has_header: false` + `io.columns: [...]` で `wine.data` をヘッダなしCSVとして読み込む
  - `columns.target: class`
- `configs/datasets/wine/labeling.yml`
  - `scheme.type: column`
  - `target.as_categorical: true`（`class` が数値でも分類として扱う）
- `configs/datasets/wine/game_table.yml`
  - 出力: `outputs/wine/game_tables/game_table.csv`

## 実行例

```bash
poetry run bci-xai preprocess \
  --dataset-config configs/datasets/wine/dataset.yml \
  --labeling-config configs/datasets/wine/labeling.yml

poetry run bci-xai build-game-table \
  --dataset-config configs/datasets/wine/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/wine/game_table.yml
```

