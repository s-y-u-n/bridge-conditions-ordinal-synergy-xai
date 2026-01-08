# 貢献度指標の設計（Group lex-cel / Group Ordinal Banzhaf）

本ドキュメントは、本リポジトリに実装している **Group lex-cel** と **Group Ordinal Banzhaf** の計算ロジック（設計方針・近似・出力仕様）をまとめたものです。

対象コマンド: `bci-xai explain-next-rank`（次点検ランク予測タスク）

関連実装:
- `src/bci_osxai/synergy/group_lexcel.py`
- `src/bci_osxai/synergy/group_ordinal_banzhaf.py`
- `src/bci_osxai/synergy/shapiq_explain.py`
- `src/bci_osxai/cli/main.py`
- `src/bci_osxai/synergy/visualize.py`

---

## 1. 入力となる「coalitional ranking（全順序・同値許容）」の作り方

理論上、Group lex-cel と Group Ordinal Banzhaf は、全ての非空部分集合 `F(N)` 上の **total preorder**（同値を許す順位付け）`≿` を入力として定義されます。

しかし、特徴数 `|N|` が大きいと `F(N)` は指数的に増えるため、実装では以下の **実用的な近似**を採用します。

### 1.1 Universe（対象とする集合族）

- shapiq で推定した相互作用の出力から、`max_order` 以下の coalition を **Universe** として採用します。
- 本実装の `explain-next-rank` では、`all_coalition_scores_shapiq(..., min_order=1)` を使い、`order=1..max_order` の coalition（特徴集合）を集めます。

要するに、理論の `F(N)` ではなく、**（1）shapiq が返した coalition** または **（2）キャッシュされた game table の coalition** の集合族 `U ⊂ F(N)` 上で計算します。

### 1.2 game table（キャッシュ）による Universe 構築（推奨）

実行コスト削減のため、`explain-next-rank` では ordinal 指標（Group lex-cel / Group Ordinal Banzhaf / Borda-Shapley）用に、
以下の **game table（coalition → スカラー score）** を作り、parquet にキャッシュします。

- 1つの「ゲーム」= `(structure_id, model, feature-set, game_table設定)` をキーにする
- 既に `n_samples` に達している場合は読み込みのみ
- 足りない場合は **重複しない coalition だけ**追加して追記

スコア（value）の定義（baseline-masking game）:
- coalition に含まれる特徴は元の `x` の値を保持
- それ以外の特徴は、背景データ（`background_X`）の中央値/最頻値で置換
- その masked 入力でモデルのスカラー出力を評価（分類は期待ラベルindex、回帰は予測値）

設定:
- `configs/synergy.yml` の `game_table.*`
- キャッシュ場所: `game_table.cache_dir`（既定 `artifacts/game_tables`）

補足（開発時の高速実行）:
- `bci-xai explain-next-rank` は `--synergy-config <path>` で設定ファイルを切り替えられます。
- 反復開発や動作確認では `configs/synergy_fast.yml`（小さい `budget/background_size/n_samples`）の利用を推奨します。
- shapiq の進捗表示は `shapiq.verbose` で制御します（デフォルトOFF。ONにすると "Evaluating game" が表示されます）。

### 1.2 preorder（順位）の誘導

Universe `U` 上で、各 coalition `S` にスコア `score(S)` を付与し、次で `≿` を誘導します。

- `score_key`: デフォルトは `abs_value`
  - `abs_value = |value|`（shapiq推定値の絶対値）
- 比較規則:
  - `score(S) > score(T)` なら `S ≻ T`
  - `|score(S) - score(T)| ≤ tie_tol` なら `S ∼ T`（同値）

この同値関係により、同値類（層）`Σ1 ≻ Σ2 ≻ ... ≻ Σℓ` を得ます（`Σ1` が最上位層）。

備考:
- `abs_value` を使うため、符号（改善方向/悪化方向）は **順位付けには使いません**（「強さ」順）。
- `tie_tol` は浮動小数の誤差吸収のために用います。

---

## 2. Group lex-cel（Group lexicographic counting of extensions）

### 2.1 定義（概要）

与えられた層構造 `Σ1 ≻ ... ≻ Σℓ` に対し、任意の非空 coalition `T` について

- `T_k := | { S ∈ Σ_k : T ⊆ S } |`
- `Θ(T) := (T_1, ..., T_ℓ)`

を定義し、`Θ(T)` を上位層からの辞書式比較で並べ替えた関係が Group lex-cel です。

### 2.2 実装上の計算

入力:
- Universe `U` 上の層 `Σ1..Σℓ`（`abs_value` によるpreorderから構成）
- 評価対象の coalition の集合（通常は Universe と同じ `U`）

出力（`group_lexcel`）:
- `equivalence_classes_head`: 先頭の層を確認できるよう、上位10層のみを列挙
- `ranking`: Group lex-cel の上位 `max_items` 件
  - `set`: coalition
  - `theta_head`: `Θ(T)` の先頭 `theta_head_len` 要素
  - `theta_nonzero`: `Θ(T)` の非ゼロ要素だけを (layer_index, value) で列挙（上限 `theta_nonzero_limit`）

実装上の注意:
- サイズの異なる coalition を混ぜると、`T ⊆ S` の成立数の差で小さい集合が有利になりやすいため、現状は **出力対象をサイズ2（ペア）に固定**しています（`fixed_order=2`）。

計算量:
- 素朴には `O(|U|^2)`（各 `T` について各層内の `S` を走査して `T ⊆ S` を数える）
- 本実装は `max_order` を小さく保つ前提で、単純実装を採用しています。

---

## 3. Group Ordinal Banzhaf（順位に基づく群Banzhaf）

### 3.1 定義（概要）

coalitional ranking `≿` に対して、任意の `T` と `S ⊆ N\\T` について

- `m_T^S(≿) = 1` if `S∪T ≻ S`
- `m_T^S(≿) = -1` if `S ≻ S∪T`
- `0` otherwise（同値など）

として、`u_T^+` と `u_T^-` の差 `s_T = u_T^+ - u_T^-` をスコアにします。

### 3.2 実装上の計算（Universe制約つき）

理論では `S` は `U_T`（`T` と交わらない全ての `S`）を走査しますが、実装では Universe `U` のみを走査し、さらに **両方が Universe に存在する場合のみ**比較します。

具体的には:
- Universe `U` の各 coalition を候補 `S` として走査
- `S∩T=∅` を満たすものだけを採用
- `S` と `S∪T` の両方が Universe `U` に存在する場合のみ比較
  - 層番号で比較（上位層ほど「良い」）:
    - `layer(S∪T) < layer(S)` → `m=1`
    - `layer(S) < layer(S∪T)` → `m=-1`
    - 同層 → `m=0`

出力（`group_ordinal_banzhaf`）:
- `ranking`: 上位 `max_items` 件
  - `set`, `order`
  - `u_plus`, `u_minus`, `score(=u_plus-u_minus)`
  - `comparisons`: 実際に比較できたペア数（Universe制約による有効比較数）

実装上の注意:
- サイズの異なる coalition を混ぜると解釈が難しくなるため、現状は **出力対象をサイズ2（ペア）に固定**しています（`fixed_order=2`）。

注意:
- Universe が小さい/偏ると `comparisons` が小さくなり、`score` の安定性が下がります。
- 現状の Universe は `max_order` 以下に限定しているため、厳密な `U_T` とは一致しません（近似）。

---

## 4. 可視化

`bci-xai explain-next-rank --plot` で以下を保存します（`artifacts/reports/`）。

- shapiq相互作用（top-k）: `*_next_rank_interactions.png`
- Group Ordinal Banzhaf（top）: `*_next_rank_group_ordinal_banzhaf.png`
- Borda-based Shapley-type Interaction Index（top）: `*_next_rank_borda_shapley.png`

Group lex-cel はスカラーの貢献度ではなく、層ごとの出現回数ベクトル `Θ(T)` を辞書式比較して得られる順位のため、グラフ化は行いません。YAMLレポートの `group_lexcel.ranking` に `theta_head/theta_nonzero` として出力します。
代わりに、`*_next_rank_group_lexcel_table.png` として「上位集合と `Θ(T)` の要約（`theta_head` / `theta_nonzero`）」を表形式で画像出力します。

---

## 5. 設定パラメータの対応

preorder/Universe（shapiq）:
- `configs/synergy.yml`
  - `shapiq.max_order`（Universeの最大次数）
  - `ordinal_metrics_max_order`（Group lex-cel / Ordinal Banzhaf 用に、上位集合を含めるための Universe 最大次数。大きくすると計算が重くなります）
  - `shapiq.budget`, `shapiq.background_size`（推定精度/計算量）
  - `report.top_k`（相互作用top-kの表示数）

Group lex-cel / Group Ordinal Banzhaf:
- 現状はコード内デフォルト（`score_key="abs_value"`, `tie_tol=1e-12`, `max_items=20`）
  - 設定ファイル化は今後の拡張ポイントです。

Borda-based Shapley-type Interaction Index:
- `configs/synergy.yml` の `borda_shapley.*` で制御します
  - `n_players`: 計算に使う特徴数（上位の単独貢献から選抜）
  - `max_order`: 計算対象の集合サイズ（`S` の最大次数）
  - `domain_max_size`: Bordaスコアと和（`T`,`L∪T`）を計算する **coalition宇宙の最大サイズ**（近似の都合で通常 `max_order` と同じにする）
  - `missing_score`: `domain_max_size` 以下でもスコアが無い coalition に与えるスコア（弱順序誘導のため）
  - 出力は `results`（正負のバランスを取ったtop-k）に加え、`results_top_abs/results_top_positive/results_top_negative` も含みます

---

## 6. Borda-based Shapley-type Interaction Index（計算プロセス）

本実装では、提示された定義

- signed Borda score `s_≽(·)`
- Borda-based Shapley-type interaction `I^B_≽(S)`

を **計算可能な有限宇宙**に落として近似計算します。

関連実装:
- `src/bci_osxai/synergy/borda_shapley.py`
- `configs/synergy.yml` の `borda_shapley.*`

### 6.1 入力

- Universe `U` 上の coalition スコア（shapiq 推定）
  - `all_coalition_scores_shapiq(...)` の出力（`set`, `order`, `value`, `abs_value` など）
- 設定（例）:
  - `n_players`（扱う特徴数）
  - `max_order`（`S` の最大サイズ）
  - `domain_max_size`（順位付け・Borda計算を行う coalition 宇宙の最大サイズ）
  - `score_key`（弱順序誘導に使うスコア: 既定 `abs_value`）
  - `tie_tol`（同値判定）
  - `missing_score`（スコア欠落の補完）

### 6.2 プレイヤー集合 N の選抜（縮約）

理論は `N` を全特徴集合として扱いますが、実装では計算量を抑えるため `n_players` に縮約します。

1. `order=1`（単独）の coalition を `abs_value` で降順に並べる
2. 上位から `n_players` 個を選び `N` とする
3. 単独が足りない場合は、出現する特徴名で補完

### 6.3 有限宇宙 F'(N) の構成（domain_max_size）

理論の `F(N)` の代わりに、

- `F'(N) := { C ⊆ N \\ {∅} | |C| ≤ domain_max_size }`

を計算宇宙として採用します。既定では `domain_max_size = max_order = 2` です（ペアまで）。

### 6.4 弱順序 ≽ の誘導（score_key + tie）

各 coalition `C ∈ F'(N)` にスコア `score(C)` を割り当て、弱順序 `≽` を誘導します。

- `score(C)` は `score_key`（既定 `abs_value`）で与える
- `C` が shapiq の Universe `U` に存在しない場合は `missing_score` を使う
- `score(C)` の降順で並べ、`|score(C)-score(D)| ≤ tie_tol` を同値 `C ∼ D` とする
- これにより同値類（層）`Σ1 ≻ Σ2 ≻ ... ≻ Σℓ` を得る

### 6.5 signed Borda score s≽(·)（弱順序版）

実装では、weak order 上の signed Borda を「クラス内同順位」とみなした一般化として計算します。

`x ∈ Σ_k`（上位から 0-based）とすると、代替（coalition）の総数を `M=|F'(N)|` として

- `#above(x) := |Σ_1| + ... + |Σ_{k-1}|`
- `#below(x) := |Σ_{k+1}| + ... + |Σ_ℓ|`
- `s_≽(x) := #below(x) - #above(x)`

（同値内は「strictly above/below」に入らない）

### 6.6 I^B_≽(S) の計算（トランケーション付き）

理論の式:

`I^B_≽(S) = Σ_{T ⊆ N\\S} w(n,s,t) · Σ_{L ⊆ S} (-1)^{|S|-|L|} s_≽(L∪T)`

を、`F'(N)` 上で計算します。

- `S` は `|S| ≤ max_order` のみ計算（出力対象）
- `T` の列挙は `|T| ≤ domain_max_size - |S|` のみ採用（`L∪T` が常に `F'(N)` に入るため）
- `s_≽(L∪T)` は 6.5 で求めた signed Borda

重みは定義通り:

- `w(n,s,t) = (n-t-s)! · t! / (n-s+1)!`

### 6.7 出力の見せ方（「全部マイナス」対策）

実データでは `results_top_abs`（絶対値上位）だけを見ると負側が並ぶことがあります。
そのため実装では:

- `results`: 正負が存在する場合に **正側/負側をバランス**して上位を提示
- `results_top_abs`: 絶対値上位（参考）
- `results_top_positive`: 正側上位
- `results_top_negative`: 負側上位

を同時に出力します（`borda_shapley` セクション）。
