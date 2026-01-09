# ゲームテーブル（特徴マスク学習）設計

## 目的

shapiq の内部サンプリングに依存せず、特徴集合（coalition）ごとに **自力で学習・予測** を行い、その予測性能を効用 `v(S)` とみなすことで、以降の Shapley / Lex-cel などの評価の土台となる **ゲームテーブル** を作成する。

## ゲームの定義

- プレイヤー集合 `N`: モデル入力の特徴量（列）集合
- coalition `S ⊆ N`: 使う特徴量の部分集合
- 効用関数 `v(S)`: coalition `S` のみを使ってモデルを学習し、ホールドアウトで評価した予測性能

### 学習・予測の手順（1 coalition あたり）

1. データ `(X, y)` を `train/test` に分割（分類は可能なら stratify）
2. `X_S := X[S]` を作り、`X_S` のみでモデルを学習
3. `test` 上で予測し、指定のメトリクスでスコア `v(S)` を算出

## 出力（CSV固定）

`artifacts/**/game_tables/*.csv` に保存する（常にCSV）。

列:

### ワイド形式（0/1 指示変数; 現行の出力）

プレイヤー（特徴量）ごとに 0/1 の列を持つ。

- `<feature_name>`: その coalition に当該特徴量が含まれるなら `1`、含まれないなら `0`
- `order`: `|S|`（= 0/1 列の合計）
- `value`: `v(S)`（Lex-cel/Shapley が参照する値）
- `abs_value`: `abs(value)`（ランキング用途）
- `metric`: スコア名（分類: `accuracy`、回帰: `neg_mae`（=-MAE）, `inv_mae`（=1/(1+MAE) で 0〜1））
- `n_train`, `n_test`: 分割サイズ
- `seed`: 乱数シード

この形式は CSV での目視や、外部ツールでの集計に向く。

### 参考（キー形式）

必要であれば `"f1|f2|..."` のような `coalition_key`（文字列）を併記する方式もあるが、現行実装は出力しない（ワイド形式のみ）。

## coalition の生成

全列挙は不可能なので、最大次数 `max_order` と件数 `n_coalitions` を指定してサンプリングする。

- まず全 singleton（`|S|=1`）を必ず含める
- 残りは `1..max_order` のサイズをランダムに選び、重複しない coalition を追加

## CLI

`bci-xai build-game-table` で作成する（ゲームテーブルは常に `.csv`）。

- `--task baseline|next-rank` で対象データを選択
- `--synergy-config` の `game_table.*` を参照して設定する
- `--out` で保存先を上書き可能

`explain` / `explain-next-rank` は、`game_table.cache_path` が存在する場合はそれを読み込み、存在しない場合は `game_table.auto_build: true` のときのみ自動生成する。
