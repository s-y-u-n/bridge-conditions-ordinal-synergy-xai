# Lex-cel（Lexicographic counting of extensions）

## 目的

本プロジェクトでは、ゲームテーブル（`docs/models/game_table.md`）の「coalition（特徴集合）→スコア」を **提携ランキング（coalition preorder）** とみなし、定義に従って Lex-cel による **プレイヤー（特徴量）** の貢献度ランキングを出力する。

ここでプレイヤー集合 `N` は `x` の特徴量（モデル入力の列）とし、coalition `S ⊆ N` はゲームテーブルに含まれる特徴集合とする。

## 入力（前提）

- `players`: `N`（特徴名のリスト）
- `coalitions`: `U ⊂ 2^N \ {∅}`（ゲームテーブルに含まれる coalition の集合族）
- `score(S)`: coalition のスコア（既定は `abs_value`）

ゲームテーブルを `U` とする。現行の出力はワイド形式（0/1 指示変数）であり、各特徴量列の 1/0 が coalition のメンバーシップを表す（詳細: `docs/models/game_table.md`）。

## 提携ランキング（preorder）の構成

実装では `U` 上の preorder `≿` を次で与える。

- `S ≿ T  ⇔  score(S) ≥ score(T)`

浮動小数の比較により、同値（同順位）を許すために `tie_tol` を用いて **近似的な同値類** を作る。

- `|score(S) - score(T)| ≤ tie_tol` を「同値」とみなし、`U` を商順序 `Σ₁ ≻ Σ₂ ≻ ... ≻ Σ_ℓ` に分割する  
  （`Σ_k` はスコアがほぼ等しい coalition の集合）

## Lex-cel の定義（出現ベクトルと辞書式比較）

各プレイヤー `i ∈ N` と各ランク `k=1..ℓ` に対し、出現頻度を

- `i_k := | { S ∈ Σ_k | i ∈ S } |`

と定義し、出現ベクトルを

- `θ_≿(i) := (i_1, ..., i_ℓ)`

とおく。

Lex-cel によるプレイヤー比較は

- `i R^le_≿ j  ⇔  θ_≿(i) ≥_lex θ_≿(j)`

で定義される（辞書式比較は第1成分から）。

## 出力仕様（YAML）

`explain` / `explain-next-rank` のレポートに `lexcel` セクションを追加する。

- `lexcel.score_key`: 使用したスコア（`abs_value` or `value`）
- `lexcel.tie_tol`: 同値類分割の許容誤差
- `lexcel.n_ranks`: `ℓ`（同値類の数）
- `lexcel.min_order`: 対象にした coalition の最小次数
- `lexcel.max_order`: 対象にした coalition の最大次数
- `lexcel.ranking`: 上位 `max_items` 件のプレイヤーランキング
  - `player`: プレイヤー名
  - `theta_head`: `θ` の先頭 `head` 個（表示用）
  - `theta_nonzero`: `(k, i_k)` のうち `i_k>0` のペア（表示用、`k` は 1-based）

注: `θ` の長さは `n_ranks` と一致するが、肥大化を避けるためレポートでは要約（`theta_head/theta_nonzero`）のみを出力する。

## 制約と注意

- `U` はサンプリングされた coalition の集合であり、理論の `2^N\\{∅}` 全体ではない。
- `max_order` や `n_coalitions` が小さいと `U` が小さくなるため、Lex-cel もその範囲での近似ランキングとなる。
