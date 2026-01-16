# ゲームテーブル（特徴マスク学習）設計

## 目的

特徴集合（coalition）ごとに **自力で学習・予測** を行い、その予測性能を効用 `v(S)` とみなす **ゲームテーブル** を作成する。

## ゲームの定義

- プレイヤー集合 `N`: モデル入力の特徴量（列）集合
- coalition `S ⊆ N`: 使う特徴量の部分集合
- 効用関数 `v(S)`: coalition `S` のみを使ってモデルを学習し、ホールドアウトで評価した予測性能

### 学習・予測の手順（1 coalition あたり）

1. データ `(X, y)` を `train/test` に分割（分類は可能なら stratify）
2. `X_S := X[S]` を作り、`X_S` のみでモデルを学習
3. `test` 上で予測し、指定のメトリクスでスコア `v(S)` を算出

## 出力（CSV固定）

`outputs/**/game_tables/*.csv` に保存する（常にCSV）。

列:

### ワイド形式（0/1 指示変数; 現行の出力）

プレイヤー（特徴量）ごとに 0/1 の列を持つ。

- `<feature_name>`: その coalition に当該特徴量が含まれるなら `1`、含まれないなら `0`
- `order`: `|S|`（= 0/1 列の合計）
- `value`: `v(S)`（Lex-cel/Shapley が参照する値）
- `abs_value`: `abs(value)`（ランキング用途）
- `metric`: スコア名（分類: `accuracy`、回帰: `mae`, `neg_mae`（=-MAE）, `inv_mae`（=1/(1+MAE) で 0〜1））
- `n_train`, `n_test`: 分割サイズ
- `n_runs`: 平均に使った評価回数（`n_repeats` / `cv_folds` に依存）
- `value_std`: スコアの標準偏差（`n_runs>1` のとき）
- `seed`: 乱数シード

この形式は CSV での目視や、外部ツールでの集計に向く。

## スコアの安定化（複数seed平均 / CV）

単発の `train/test` 分割だと、分割の偶然で `value` がぶれやすい（特にサンプル数が小さいデータセット）。
そのため、以下のいずれかでスコアを平均して安定化できるようにする。

- 複数seed平均（repeated holdout）: `n_repeats>1` なら `seed, seed+1, ...` の分割でスコアを平均
- 交差検証（CV）: `cv_folds>=2` なら KFold/StratifiedKFold を使ってスコアを平均

どちらも指定された場合は「`n_repeats` 回の CV」の平均を取る（計算量が増えるため注意）。

## ゲームテーブル設定（追加）

`game_table.yml` の `game_table.*` に以下を追加できる。

- `n_repeats`（既定: 1）: repeated holdout / repeated CV の回数
- `cv_folds`（既定: 0）: 2 以上で CV（0 は従来どおり holdout）

## 空集合（無情報）ベースライン `v(∅)`

比較の都合上、「特徴量なし（無情報）」での理論的スコア `v(∅)` を別途計算できるようにする。

- 分割: ゲームテーブルと同じく `train/test` 分割（分類は可能なら stratify）
- 分類（`accuracy`）: `y_train` の最頻クラスを常に予測（同数タイは決定的に解消）
- 回帰（`mae` / `neg_mae` / `inv_mae`）: `y_train` の中央値を常に予測（MAE 最適）

出力はゲームテーブルと同じワイド形式で、`order=0` の 1 行だけを持つ CSV。

- 既定: `outputs/<dataset_key>/game_tables/empty_baseline.csv`

### 参考（キー形式）

必要であれば `"f1|f2|..."` のような `coalition_key`（文字列）を併記する方式もあるが、現行実装は出力しない（ワイド形式のみ）。

## coalition の生成

全列挙は不可能なので、最大次数 `max_order` と件数 `n_coalitions` を指定してサンプリングする。

- まず全 singleton（`|S|=1`）を必ず含める
- 残りは `1..max_order` のサイズをランダムに選び、重複しない coalition を追加

## CLI

`bci-xai build-game-table` で作成する（ゲームテーブルは常に `.csv`）。

- `--task baseline|next-rank` で対象データを選択
- `--game-table-config` の `game_table.*` を参照して設定する
- `--out` で保存先を上書き可能

`bci-xai build-empty-baseline` で `v(∅)` を作成する（学習なし; ラベルのみ）。

## スコアの弱順序（同値類つきランキング）

ゲームテーブルの `value` から、連続区間制約つきの最適1次元 k-means（Jenks 相当）で
弱順序 `Σ1 ≻ Σ2 ≻ … ≻ Σk` を生成し、`class_id`（1が最上位）を付与できる。

- `poetry run bci-xai rank-game-table --game-table-csv <path/to/game_table.csv> --score-col value`
- `--plot-out` と `--empty-baseline-csv` を指定すると、分割線入りのスコア分布図も出力できる
