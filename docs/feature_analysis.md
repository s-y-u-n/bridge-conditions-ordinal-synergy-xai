# 特徴量分析パッケージ（設計書）

## 1. 目的

本リポジトリで管理している各データセット（`configs/datasets/**/dataset.yml`）に対して、
機械学習の予測モデル構築の前段として一般的に行う **特徴量の定量分析（EDA + 目的変数との関係）** を自動で実行し、
後工程（ゲームテーブル → 貢献度/シナジー分析を別ディレクトリで実施）へ渡せる成果物を `outputs/` に揃える。

## 2. スコープ

### 2.1 対象（Phase 1）

- `preprocess` 後に生成される学習用データ（`features.csv` / `labels.csv`）を入力とする分析
- `task` ごとに入力形を切り替え
  - `baseline`: `features.csv` + `labels.csv`
- `next-rank`: `next_rank_dataset.csv`（存在する場合）
- 出力は **CSV/JSON/Markdown** のみ（外部ツールで扱いやすくするため）

### 2.2 対象外（Phase 1）

- 図（ヒストグラム/散布図）生成（matplotlib 等の依存追加を避けるため）
- モデル学習を伴う重要度（Permutation importance / SHAP 等）
- ゲームテーブル（coalition→score）を入力とした貢献度/相互作用の推定

## 3. 入力

### 3.1 必須入力

- `dataset_config`: `configs/datasets/**/dataset.yml`

### 3.2 任意入力

- `labeling_config`: `configs/datasets/**/labeling.yml`
  - `baseline` の場合、`labels.csv` に `label` が存在すれば不要（存在しない場合に補助的に用いる）
- `task`: `baseline`（既定）/ `next-rank`
- `out_dir`: 出力先（既定は `outputs/<dataset_key>/analysis/<task>/`）

## 4. 出力（成果物）

出力先（既定）:

- `outputs/<dataset_key>/analysis/<task>/`

ここで `dataset_key` は `configs/datasets/` からの相対パスを安全な文字列に変換したもの。
例:

- `configs/datasets/credit_g/dataset.yml` → `credit_g`
- `configs/datasets/bridge_conditions/experiments/baseline10/dataset.yml` → `bridge_conditions__experiments__baseline10`

生成ファイル（全て任意だが、基本は常に出す）:

- `overview.json`: 行数/列数、欠損率サマリ、目的変数型、推定された特徴量型などのメタ情報
- `features_summary.csv`: 各特徴量の欠損率・一意数・代表値（数値は分位点など）
- `target_summary.csv`: 目的変数の分布（分類: 件数、回帰: describe）
- `feature_target_assoc.csv`: 目的変数との関連度（スコア + 可能なら p 値）
- `numeric_correlations_spearman.csv`: 数値特徴量間の Spearman 相関（上位ペアのみ）
- `report.md`: 主要結果の要約（リンク/パス中心。値の重複出力は最小）

## 5. 分析内容（Phase 1）

### 5.1 特徴量サマリ

各列について以下を出力する。

- `dtype`（pandas dtype）
- `n_missing`, `missing_rate`
- `n_unique`（欠損除外）, `unique_rate`
- 最頻値（カテゴリ）: `top1`, `top1_rate`（計算可能な場合）
- 数値列: `mean`, `std`, `min`, `p25`, `p50`, `p75`, `max`

### 5.2 目的変数サマリ

- 分類（`y` が非数値 or 一意数が小さいと推定）: クラス件数、比率
- 回帰（`y` が数値）: `describe()` 相当

### 5.3 目的変数との関連度（軽量）

列型とタスクに応じて、以下の「軽量指標」を優先する。

- 分類（カテゴリ y）:
  - 数値特徴量: ANOVA F (`sklearn.feature_selection.f_classif`)
  - カテゴリ特徴量: Cramér's V（分割表から計算）
- 回帰（数値 y）:
  - 数値特徴量: Pearson/Spearman 相関
  - カテゴリ特徴量: 目的変数平均との差（カテゴリごとの平均・分散を要約; もしくは相関の代替指標として扱う）

※ Phase 1 は「見通しを立てる」ことが目的のため、過度に複雑な統計モデルには踏み込まない。

## 6. 実装方針

### 6.1 パッケージ構成

- `bci_osxai/analysis/`
  - `dataset_analysis.py`: 入力ロード、集計、保存
  - `stats.py`: Cramér's V 等の小さな統計ユーティリティ

### 6.2 CLI

既存 CLI（`bci-xai`）にサブコマンドを追加する。

- `bci-xai analyze-dataset --dataset-config ... [--labeling-config ...] [--task baseline|next-rank] [--out-dir ...]`
- `bci-xai analyze-all-datasets [--task ...] [--configs-root configs/datasets]`

`analyze-all-datasets` は `_template_tabular` を除外し、見つかった `dataset.yml` を順番に処理する。

### 6.3 失敗時の扱い

- 入力ファイル（`features.csv` / `labels.csv` 等）が無い場合は、どのファイルが無いかを明示してエラー終了
- 列型が混在して計算不能な場合は、その指標だけを欠損（NaN）として継続し、`report.md` に警告を残す

## 7. 将来拡張（Phase 2 以降）

- 図生成（matplotlib/seaborn）
- データ品質検査（リーク候補列、極端な欠損、定数列の自動除外）
- モデルベース重要度（Permutation/SHAP）と、ゲームテーブル由来の貢献度分析の統合
