# 特徴量貢献度（Power Index）計算: 設計書

## 目的

`artifacts/game_tables/baseline10.csv` のような「特徴量集合（coalition）→スコア（特性関数 v）」の表（ゲームテーブル）を入力として、
各特徴量（プレイヤー）の貢献度を **Power Index**（例: Shapley value）として計算し、CSVとして出力する。

将来的に Shapley value 以外（例: Banzhaf index）へ差し替え可能なように、計算ロジックをモジュール化する。

## 前提（入力データの意味）

- プレイヤー集合を `N`（特徴量集合）とする。
- 特性関数 `v: 2^N -> R` は「特徴量集合 S だけを使って学習・評価した指標値（例: accuracy, neg_mae, inv_mae）」として与えられる。
- ゲームテーブルは次のいずれかの形式を想定する:
  1) **インジケータ行列形式**: 各特徴量列が 0/1、行が部分集合 `S`、`value` 列が `v(S)` を表す（baseline10.csv がこれ）
  2) **coalition_key 形式**: `coalition_key="a|b|c"` のように集合を文字列で持ち、`value` 列が `v(S)` を表す

## 出力（CSV）

プレイヤーごとに 1 行のテーブルを出力する（最低限）。

必須列:
- `player`: 特徴量名
- `index`: 指標名（例: `shapley`, `banzhaf`）
- `value`: 貢献度（Power index）

任意列（デバッグ/検算用途）:
- `n_players`
- `v_empty`（`v(∅)` として採用した値）
- `v_full`（`v(N)`）
- `sum_values`（`sum_i phi(i)`）
- `efficiency_gap`（`sum_i phi(i) - (v(N)-v(∅))`）

## モジュール構成

`bci_osxai.synergy` 配下に、ゲーム表から Power index を計算するモジュールを追加する。

### 1) CooperativeGame（ゲーム表の共通表現）

- 役割: `v(S)` を高速に参照できる形（ビットマスク→値）に正規化し、計算指標から独立させる。
- 入力: pandas DataFrame（ゲームテーブル）
- 出力: `players: list[str]` と `values_by_mask: np.ndarray`（長さ `2^n`、`mask` が集合 `S` を表す）

### 2) PowerIndex インターフェース（差し替え点）

- `compute(game: CooperativeGame) -> pd.DataFrame` の形で統一する。
- 実装例:
  - `ShapleyValueIndex`
  - `BanzhafIndex`（後続の差し替え用に同居させておく）

### 3) CLI（入出力）

既存の `bci-xai` CLI にサブコマンドを追加して、次を可能にする:

- ゲームテーブル（csv/parquet）を読み込み
- 指標（Shapley/Banzhaf）を選択
- `v(∅)` が未収録の場合の扱い（`--v-empty`）を指定
- 貢献度を CSV に出力

## Shapley value の計算（実装方針）

定義（ユーザー指定）:

プレイヤー `i ∈ N` の Shapley value は

`phi(i) = Σ_{S ⊆ N\{i}} [ |S|!(n-|S|-1)! / n! ] * ( v(S∪{i}) - v(S) )`

実装では:
- `w(k) = k!(n-k-1)!/n!` を `k=0..n-1` で事前計算
- 各 `i` について `S` をビットマスクで走査して `v(S∪{i}) - v(S)` を加重和

## `v(∅)` の取り扱い

baseline10 のように空集合がゲームテーブルに存在しないケースがある。
その場合、次のルールで補う:

- デフォルト: `v(∅)=0.0`
- ただし CLI で `--v-empty` を指定可能にする（例: majority class の accuracy を入れる等）

## エラーハンドリング / 検証

- プレイヤー数 `n` を推定し、期待行数が `2^n-1`（空集合なし）または `2^n`（空集合あり）に近いかをチェック（厳密一致は `--strict` 相当で制御）。
- `v(S)` が欠損して Shapley 計算に必要な `S`/`S∪{i}` の参照ができない場合は例外にする（近似ではなく「定義通り」を優先）。
- 出力時に `efficiency_gap` を併記し、`sum_i phi(i) ≈ v(N)-v(∅)` を簡易検算できるようにする。
