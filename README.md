# Bridge Conditions Ordinal Synergy XAI

Ontarioの「Bridge conditions」CSVを読み込み、BCIを序数ラベルに変換して学習・推論し、特徴集合のシナジー説明（序数比較）を出力するためのリポジトリです。

## セットアップ

1. Poetry をインストール（https://python-poetry.org/）
2. 依存関係のインストール:

```bash
poetry install
```

## データの置き場所

- 生データ（CSV）は `data/raw/bridge_conditions.csv` に置いてください（git管理しません）。
- 前処理の最初に改行混入を除去したCSVを `data/processed/bridge_conditions_clean.csv` に作成します。
- 文字コードが合わない場合は `configs/dataset.yml` の `io.encoding` を変更してください。
- CSVの列定義（データ辞書）は `docs/data_dictionary_bridge_conditions.md` を参照してください。
- 生成物は `data/processed/` に出力されます（parquet）。
- 学習済みモデルやレポートは `artifacts/` に出力されます。

## 使い方（最小フロー）

```bash
poetry run bci-xai preprocess
poetry run bci-xai train
poetry run bci-xai explain --id <structure_id>
poetry run bci-xai explain --id <structure_id> --plot
```

## ラベル設計（2方向）

1) 連続値のBCIをそのまま使う（回帰）  
2) BCIをランク化（分類）

用途別に `--labeling-config` を切り替えます:

- 連続BCI（回帰）: `configs/labeling_continuous.yml`
- 5段階ランク（補修を考慮）: `configs/labeling_rank5.yml`

例:

```bash
poetry run bci-xai preprocess --labeling-config configs/labeling_continuous.yml
poetry run bci-xai train
```

```bash
poetry run bci-xai preprocess --labeling-config configs/labeling_rank5.yml
poetry run bci-xai train
```

`R5_REHAB_FLOOR` の考え方:
- 年次BCIで「上昇」が初めて起きる直前の値を `floor_bci` とみなす（補修の影響を想定）
- `floor_bci` から 100 までを、仮定した分布（デフォルト: Beta(5,2)）の分位点で5段階に割り当てる

## 時系列タスク（推奨）: 次の点検時ランク予測

過去の年次BCI（`bci_2000..bci_2020`）から、次に観測されるBCIのランク（R0..R4）を予測します。

```bash
poetry run bci-xai preprocess --labeling-config configs/labeling_rank5.yml
poetry run bci-xai train-next-rank
poetry run bci-xai predict-next-rank --id <structure_id>
poetry run bci-xai explain-next-rank --id <structure_id> --plot
```

学習モデルはデフォルトで XGBoost を使用します。

`explain-next-rank` は相互作用（shapiq）に加えて、Group lex-cel（`Θ(T)` の辞書順で得られる順位）も `group_lexcel` として出力します（`*_next_rank_group_lexcel_table.png` に表形式で可視化します）。
また、Group Ordinal Banzhaf も `group_ordinal_banzhaf` として出力し、`--plot` でグラフ（`*_next_rank_group_ordinal_banzhaf.png`）を保存します。
さらに、Borda-based Shapley-type Interaction Index も `borda_shapley` として出力し、`--plot` で `*_next_rank_borda_shapley.png` を保存します。

標準出力はデフォルトで出しません（`artifacts/reports/*.yml` に保存）。必要なら `--stdout` を付けてYAMLを表示できます。

## 実行時間と進捗表示

`explain-next-rank` は `configs/synergy.yml` の設定次第で計算時間が大きく変わります。

- `shapiq.budget` / `shapiq.background_size`: 大きいほど推定は安定しやすい一方、遅くなります
- `ordinal_metrics_max_order`: Group lex-cel / Ordinal Banzhaf が参照する Universe の最大次数です（大きいほど遅い）
- `progress.enabled: true` のとき、tqdm で進捗を **stderr** に表示します（stdoutはデフォルトOFFのまま）
- 開発中に高速で回したい場合は `--synergy-config configs/synergy_fast.yml` を使ってください（小さい `budget/background_size/n_samples`）。

例:

```bash
poetry run bci-xai explain-next-rank --id <structure_id> --plot --synergy-config configs/synergy_fast.yml
```

※ `shapiq.verbose: true` にすると shapiq 側の進捗表示（"Evaluating game"）が出ますが、ログが増えるためデフォルトではOFFです。

## game table キャッシュ

`explain-next-rank` の ordinal 指標（Group lex-cel / Group Ordinal Banzhaf / Borda-Shapley）は、毎回 shapiq で大きい Universe を推定すると高コストになるため、
`configs/synergy.yml` の `game_table.enabled: true` の場合、coalition → score のテーブルを parquet にキャッシュします。

- 既に `game_table.n_samples` に達していれば読み込みのみ
- 足りない場合は重複しない coalition を追加して追記
- 保存先: `game_table.cache_dir`（既定 `artifacts/game_tables`）

## shapiq キャッシュ（相互作用）

`explain-next-rank` の `synergy_top_k`（shapiq 相互作用）は `shapiq.budget` 回の評価が走るため、最も時間がかかりやすいです。
同一の `(structure_id, model, dataset, shapiq設定)` に対しては、相互作用テーブルを `artifacts/shapiq_cache/` に parquet でキャッシュし、2回目以降は読み込みのみになります。

※ `shapiq.verbose: true` の場合、初回計算時に "Evaluating game: 1000/1000 ..." のような進捗が出ますが、キャッシュヒット時は計算自体をしないため出ません。

## よくあるエラー

`ModuleNotFoundError: No module named 'bci_osxai'` が出る場合は、まず `poetry install` を実行してください。
また、Python 3.13 は依存ライブラリの対応が不十分な場合があるため、3.11 または 3.12 を推奨します。

```bash
poetry env use 3.11
poetry install
```

## 出力例（シナジーレポート）

```
structure_id: 12345
predicted_label: L2
synergy_top_k:
  - set: [material=Steel, region=West]
    relation: "S ≻ T"
```

## 設計書

初版の設計書は `docs/design.md` を参照してください。
Group lex-cel / Group Ordinal Banzhaf の設計は `docs/contribution_metrics.md` を参照してください。
