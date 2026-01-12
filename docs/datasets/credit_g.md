# credit_g データセット対応: 設計メモ

## 目的

`data/raw/credit_g/credit_g.csv`（German Credit / credit-g）を入力として、
既存の共通フロー

1) データクレンジング
2) 特徴量絞り込み
3) 特徴マスクで学習・評価（ゲームテーブル作成）

を実行できるようにする。

## 入力

- ファイル: `data/raw/credit_g/credit_g.csv`
- 形式: CSV
- 目的変数: `class`（`good` / `bad`）

## クレンジング方針

- CSV を読み込む
- `?` を欠損として `NA` に正規化する（必要なら）

## 中間生成物（processed）

`data/processed/credit_g/` に次を出力する（CSV固定）:

- `structures.csv`: クレンジング済みの全テーブル（`structure_id` を付与）
- `features.csv`: 学習用特徴量（`structure_id` を含む）
- `labels.csv`: 目的変数（`structure_id`, `label`）

※ tabular データセットでは `bci_long.csv` は作らない。

## 特徴量（入力列）

`class`（目的変数）以外の 20 列を特徴量として扱う。

- `checking_status`: 当座預金口座の状態（残高レンジ/口座なし）
- `duration`: 借入期間（ヶ月）
- `credit_history`: クレジット履歴（延滞有無など）
- `purpose`: 借入目的（車/家具/教育など）
- `credit_amount`: 借入額
- `savings_status`: 貯蓄/債券の状態（残高レンジ/不明）
- `employment`: 現職の勤続年数レンジ（失業含む）
- `installment_commitment`: 可処分所得に占める返済割合
- `personal_status`: 個人属性（婚姻/性別カテゴリ）
- `other_parties`: 共同申請者/保証人の有無
- `residence_since`: 現住居の居住年数
- `property_magnitude`: 資産区分（不動産/保険/車/なし等）
- `age`: 年齢
- `other_payment_plans`: 他の分割払い計画（銀行/店舗/なし）
- `housing`: 住居形態（賃貸/持家/無償）
- `existing_credits`: 当該銀行での既存クレジット件数
- `job`: 職種カテゴリ（非熟練/熟練/管理職等）
- `num_dependents`: 扶養家族人数
- `own_telephone`: 電話保有（なし/あり）
- `foreign_worker`: 外国人労働者か（yes/no）

## 目的変数（ラベル）

- `class`: 信用リスクのクラス（`good` / `bad`）
  - `good`: 良い信用（貸倒れリスクが低い）
  - `bad`: 悪い信用（貸倒れリスクが高い）

補足:
- 学習時はカテゴリ列は one-hot 化し、数値列は中央値補完・スケーリングする: `src/bci_osxai/models/train.py:13`

## 設定ファイル

- `configs/datasets/credit_g/dataset.yml`
  - `io.format: csv` を指定
  - `paths.*_csv` の出力先を `data/processed/credit_g/` に集約
- `configs/datasets/credit_g/labeling.yml`
  - `target.source: column` / `target.column: class`
  - `scheme.type: column`（列をそのままラベルとして使う）
- `configs/datasets/credit_g/game_table.yml`
  - `game_table.metric: accuracy`
  - `game_table.cache_path: outputs/credit_g/game_tables/game_table.csv`（CSV固定）

## 実行例

```bash
poetry run bci-xai preprocess \
  --dataset-config configs/datasets/credit_g/dataset.yml \
  --labeling-config configs/datasets/credit_g/labeling.yml

poetry run bci-xai build-game-table \
  --dataset-config configs/datasets/credit_g/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/credit_g/game_table.yml
```

## 実験（top10 全列挙）

特徴量を 10 個に絞って（`2^10-1=1023` coalition）を全列挙してゲームテーブルを作る場合:

```bash
poetry run bci-xai preprocess \
  --dataset-config configs/datasets/credit_g/experiments/top10/dataset.yml \
  --labeling-config configs/datasets/credit_g/labeling.yml

poetry run bci-xai build-game-table \
  --dataset-config configs/datasets/credit_g/experiments/top10/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/credit_g/experiments/top10/game_table.yml
```
