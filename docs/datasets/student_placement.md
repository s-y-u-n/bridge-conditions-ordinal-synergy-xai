# Dataset: student_placement

## 概要

Student Placement Dataset（`data/raw/student_placement/train.csv`）を、このリポジトリの標準パイプライン（`preprocess` → `build-game-table` → `analyze-dataset`）で扱えるようにする。

本データセットは 1行=1学生の tabular データで、目的変数は `Placement_Status`（分類）とする。

## 入力

- `data/raw/student_placement/train.csv`
  - 列（抜粋）:
    - ID: `Student_ID`
    - target: `Placement_Status`（例: `Placed` / `Not Placed`）
    - features: `Age`, `Gender`, `Degree`, `Branch`, `CGPA`, `Internships`, ... 等

※ `test.csv` は target が無いため、現状の `preprocess` 対象外（必要なら Phase 2 で推論用データとして別扱いにする）。

## 設定

- `configs/datasets/student_placement/dataset.yml`
  - 入力: `paths.raw_csv`
  - 出力: `data/processed/student_placement/{structures,features,labels}.csv`
  - `columns.id: Student_ID`
  - `columns.target: Placement_Status`
- `configs/datasets/student_placement/labeling.yml`
  - `target.source: column`
  - `target.column: Placement_Status`
  - `schemes.PLACEMENT.type: column`（分類ラベルをそのまま利用）
- `configs/datasets/student_placement/game_table.yml`
  - 出力: `outputs/student_placement/game_tables/game_table.csv`

## 実行手順

1) 実験用データセット作成（features/labels）

```bash
poetry run bci-xai preprocess \
  --dataset-config configs/datasets/student_placement/dataset.yml \
  --labeling-config configs/datasets/student_placement/labeling.yml
```

2) 特徴量の分析（EDA）

```bash
poetry run bci-xai analyze-dataset \
  --dataset-config configs/datasets/student_placement/dataset.yml \
  --task baseline
```

3) ゲームテーブル作成（特徴マスク学習）

```bash
poetry run bci-xai build-game-table \
  --dataset-config configs/datasets/student_placement/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/student_placement/game_table.yml
```
