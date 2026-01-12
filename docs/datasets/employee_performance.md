# Dataset: employee_performance

## 概要（データセット説明の整理）

出典: Kaggle「Employee Performance and Productivity Data」  
https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data

本データセットは、企業環境における従業員の **パフォーマンス**・**生産性**・**デモグラフィック**を捉えるための tabular データで、全 100,000 行を含む。従業員の職務、勤務習慣、学歴、評価、満足度などの情報が含まれており、HR 分析、離職（churn）予測、生産性分析、評価分析などを想定している。

## 入力（生データ）

- `data/raw/employee_performance/Extended_Employee_Performance_and_Productivity_Data.csv`

## 列定義（原文の要約 + 本CSVの実体に合わせた整理）

本CSVには 20 列が存在する（`shape: (100000, 20)`）。

| 列名 | 型（想定） | 説明 | 制約・例 |
|---|---:|---|---|
| `Employee_ID` | int | 従業員ごとの一意なID | 一意（本CSVでは重複なし） |
| `Department` | category/string | 部署 | 例: Sales, HR, IT など |
| `Gender` | category/string | 性別 | 例: Male, Female, Other |
| `Age` | int | 年齢 | 22〜60 |
| `Job_Title` | category/string | 役職/職種 | 例: Manager, Analyst, Developer など |
| `Hire_Date` | datetime/string | 入社日 | 本CSVでは日時文字列（ISO風） |
| `Years_At_Company` | int | 在籍年数 | 0 以上 |
| `Education_Level` | category/string | 最終学歴 | High School, Bachelor, Master, PhD |
| `Performance_Score` | int | パフォーマンス評価 | 1〜5 |
| `Monthly_Salary` | int/float | 月給（USD） | 職種・評価と相関（とされる） |
| `Work_Hours_Per_Week` | int | 週あたり労働時間 | 正の整数想定 |
| `Projects_Handled` | int | 担当したプロジェクト数 | 0 以上 |
| `Overtime_Hours` | int | （直近1年の）残業時間 | 0 以上 |
| `Sick_Days` | int | 病欠日数 | 0 以上 |
| `Remote_Work_Frequency` | int | リモート勤務割合（%） | 0, 25, 50, 75, 100 |
| `Team_Size` | int | チーム人数 | 1 以上想定 |
| `Training_Hours` | int | 研修時間 | 0 以上 |
| `Promotions` | int | 昇進回数 | 0 以上 |
| `Employee_Satisfaction_Score` | float | 従業員満足度 | 1.0〜5.0 |
| `Resigned` | bool | 退職済みフラグ | True/False |

## 想定ユースケース（Kaggle記載の整理）

- 離職（Churn）予測: 退職に至るパターンの特定（`Resigned` を目的変数）
- 生産性分析: リモート率、残業、研修などが生産性/成果に与える影響の分析
- パフォーマンス評価分析: `Performance_Score` と給与、チーム規模、学歴等の関係分析
- HR 分析: 従業員属性・行動の傾向を把握し、戦略意思決定に活用

## 整形（前処理）設計：Phase 1（生データの正規化）

目的: 「列の型・表記ゆれ・範囲」を確定し、`data/processed/employee_performance/` 配下に **解析・学習で再利用可能なクリーンCSV** を出力する。

### 入出力

- 入力: `data/raw/employee_performance/Extended_Employee_Performance_and_Productivity_Data.csv`
- 出力（案）: `data/processed/employee_performance/employee_performance_clean.csv`

### 型の正規化ルール（案）

- `Hire_Date`: `datetime` としてパースし、出力は `YYYY-MM-DD` もしくは `YYYY-MM-DDTHH:MM:SS`（要選択）
- `Remote_Work_Frequency`: 0〜100 の整数（%）として保持（必要なら派生で `remote_work_ratio = Remote_Work_Frequency / 100` を追加）
- `Monthly_Salary`: 数値（USD）。本CSVでは小数なしのため整数扱いでも良いが、入力は float を許容
- `Resigned`: `True/False`（入力が文字列の場合は `true/false/0/1` を吸収）

### 品質チェック（案）

- 主キー: `Employee_ID` の一意性
- 値域:
  - `Age` は 22〜60
  - `Performance_Score` は 1〜5
  - `Employee_Satisfaction_Score` は 1.0〜5.0
  - `Remote_Work_Frequency` は {0, 25, 50, 75, 100}
- 欠損: 欠損がある場合は列ごとに方針を定義（現状のCSVは欠損なし）

## 整形（前処理）設計：Phase 2（目的変数の設計案）

本リポジトリの目的（序数ラベル + XAI）に合わせ、目的変数候補は以下。

- 序数分類: `Performance_Score`（1〜5）
- 二値分類: `Resigned`（離職）

どちらを採用するかで `configs/datasets/employee_performance/` の設計（features/labels の切り分け）が変わるため、次フェーズで決定する。

## 予測タスク案: 「パフォーマンスに何が効いているか」

目的: 以下の特徴量から `Performance_Score`（1〜5）を予測し、さらに **特徴量を制限したときの予測スコア**をゲームテーブルとして出力する。

- 目的変数: `Performance_Score`
- 特徴量（7個）:
  - `Years_At_Company`
  - `Education_Level`
  - `Work_Hours_Per_Week`
  - `Overtime_Hours`
  - `Projects_Handled`
  - `Remote_Work_Frequency`
  - `Training_Hours`

設定ファイル:

- `configs/datasets/employee_performance/experiments/performance_score_7/dataset.yml`
- `configs/datasets/employee_performance/labeling.yml`（`Performance_Score` を `S1`..`S5` にして分類として扱う）
- `configs/datasets/employee_performance/experiments/performance_score_7/game_table.yml`

実行例:

```bash
poetry run bci-xai preprocess \
  --dataset-config configs/datasets/employee_performance/experiments/performance_score_7/dataset.yml \
  --labeling-config configs/datasets/employee_performance/labeling.yml

poetry run bci-xai train \
  --dataset-config configs/datasets/employee_performance/experiments/performance_score_7/dataset.yml \
  --task baseline

poetry run bci-xai build-game-table \
  --dataset-config configs/datasets/employee_performance/experiments/performance_score_7/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/employee_performance/experiments/performance_score_7/game_table.yml
```

## 分析タスク案: 「どの要因があると退職しやすいか」（退職率テーブル）

目的: 以下の“条件（リスク要因）”ごとに、`Resigned=True` の割合（退職率）を集計し、条件の組合せ（coalition）→ 退職率のテーブルを作る（モデル学習はしない）。

リスク要因（5個, 0/1）:

- `low_income`: **低収入**（同じ `Age` の `Monthly_Salary` 中央値未満）
- `long_overtime`: **長時間残業**（`Overtime_Hours` が全体中央値以上）
- `low_remote`: **リモートワークが少ない**（`Remote_Work_Frequency == 0`）
- `large_team`: **大人数チーム**（`Team_Size` が全体中央値より大きい）
- `low_promotion`: **昇進が少ない**（同じ `Age` の `Promotions` 中央値以下）

スコア:

- `resignation_rate`: `Resigned=True` の割合

コマンド:

```bash
poetry run bci-xai build-resignation-rate-table \
  --dataset-config configs/datasets/employee_performance/experiments/resignation_risk_5/dataset.yml \
  --include-empty
```

出力:

- `outputs/employee_performance__experiments__resignation_risk_5/game_tables/resignation_rate_table.csv`
