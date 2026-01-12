# プログラム開発: 設計書（初版）

## 1. 目的とスコープ

### 1.1 目的

Ontario "Bridge conditions" CSV を入力として、以下を実現する。

1. 序数目的変数の定義（BCIなど連続値を管理判断に近い序数へ変換）
2. 予測モデル（例: Ordinal classification / ranking）
3. 説明可能性: あなたの 序数上のシナジー指標 を用いて、
   - 単独要因では説明しにくい「組合せ（特徴集合）の効き方」を
   - **順位比較（total preorder）**として提示する

### 1.2 スコープ外（初版）

- 地図可視化（Phase 2）
- 画像データ（Phase 2）
- 年次BCI（2000–2020）を時系列モデルとして扱う高度化（Phase 2）

---

## 2. 入力データ仕様（現状把握）

詳細な列定義（データ辞書）は `docs/data_dictionary_bridge_conditions.md` を参照。

### 2.1 主な列（例示）

- ID / STRUCTURE NAME / HIGHWAY NAME
- LATITUDE, LONGITUDE
- CATEGORY, SUBCATEGORY 1, TYPE 1, MATERIAL 1
- YEAR BUILT, LAST MAJOR REHAB, LAST MINOR REHAB
- NUMBER OF SPAN / CELLS, SPAN DETAILS (m), DECK / CULVERTS LENGTH (m), WIDTH TOTAL (m)
- REGION, COUNTY, OPERATION STATUS, OWNER
- LAST INSPECTION DATE
- CURRENT BCI
- 年次列: 2020, 2019, …, 2000（欠損多い）

### 2.2 想定される問題

- 文字列に数値が混在（例: NUMBER OF SPAN / CELLS に Total=...）
- 年次列が wide 形式（2000..2020）で欠損が散在
- 日付形式が混在し得る（MM/DD/YYYY）
- CATEGORY（Bridge/Culvert）により意味が変わる列がある

---

## 3. 生成データ（中間テーブル）設計（CSV固定）

### 3.1 データモデル

- raw/bridge_conditions.csv（入手原本）
- `data/processed/<dataset_key>/structures.csv`（構造物単位: 最新断面）
- `data/processed/<dataset_key>/bci_long.csv`（年次BCIのlong形式; 該当データセットのみ）
- `data/processed/<dataset_key>/features.csv`（学習用特徴量）
- `data/processed/<dataset_key>/labels.csv`（ラベル; `label` 列を含む）
- `data/processed/<dataset_key>/next_rank_dataset.csv`（next-rank 用; 該当データセットのみ）

### 3.2 long 形式（bci_long）

キー:

- structure_id（ID (SITE Ndeg)）
- year（int）

列:

- bci（float, 欠損あり）
- bci_source（"CURRENT_BCI" or "YEAR_COL" など）

---

## 4. 目的変数（序数ラベル）設計

### 4.1 ラベル候補（BCI→序数）

BCIは0–100の連続値なので、管理判断に合わせて段階化する。

案A（4クラス、初版推奨）

- L3: 85–100（良好）
- L2: 70–85（注意）
- L1: 55–70（要対策検討）
- L0: 0–55（優先対策）

案B（3クラス）

- High / Mid / Low

備考:

- 閾値は固定でもよいが、後で政策基準に合わせて差し替え可能にする（設定ファイル化）。

### 4.2 学習ターゲットの選択

初版では以下の2系統を用意し、実験で選ぶ。

- 静的: CURRENT BCI の序数ラベル（1構造物=1サンプル）
- 年次: year=2020 のBCI列（同上、列があるものに限定）

（Phase 2で「劣化速度」などの序数化も可能）

---

## 5. 特徴量設計（最小コア）

### 5.1 そのまま使えるカテゴリ

- CATEGORY, SUBCATEGORY 1, TYPE 1, MATERIAL 1
- REGION, COUNTY, OWNER, OPERATION STATUS

### 5.2 数値化する項目

- YEAR BUILT → age = inspection_year - year_built
- LAST MAJOR REHAB, LAST MINOR REHAB → years_since_rehab
- DECK / CULVERTS LENGTH (m), WIDTH TOTAL (m)
- NUMBER OF SPAN / CELLS → 正規化された span_count（パーサで抽出）

### 5.3 解析対象の切り分け

- 初版は OPERATION STATUS = Open to traffic のみ（ノイズ低減）
- Bridge と Culvert はモデルを分ける or CATEGORYで分岐（設定で選べる）

---

## 6. 学習・評価の基本方針

### 6.1 ベースライン

- XGBoost / LightGBM の多クラス分類（序数無視）
- Ordinal regression（可能なら: 累積リンク等）※実装は後追いでも可

### 6.2 評価指標（序数向け）

- MAE（クラス番号差）
- Quadratic Weighted Kappa（可能なら）
- Accuracy（参考）

---

## 7. 説明可能性: 序数シナジー指標の組込み設計

### 7.1 目的

各サンプル（橋梁）について、

- 単独特徴の重要度ではなく、
- 特徴集合（例: {material=Steel, region=West, age>50}）の組合せが
- 予測ランクに与える影響を 序数比較で提示する。

### 7.2 入力（あなたの指標への入力形）

- 予測モデル f(x) を用意
- 特徴集合 S を選び、介入（mask / baseline / counterfactual）で f_S(x) の順位変化を得る
- そこから「シナジー比較規則」R^I を構成

※ここはあなたの理論（total preorder許容、辞書式・多数決等）に一致させる。

### 7.3 実装上の最小仕様（Phase 1）

- 特徴集合はサイズ2または3まで（計算量を抑える）
- 候補集合は
  - 上位頻出カテゴリ組合せ
  - 重要度上位特徴の組合せ
  - ルールベース（例: age×material×rehab）
- 出力は「比較可能な順序」:
  - S ≻ T / S ∼ T / S ⟂ T（必要なら）
- total preorderに寄せる場合は tie 許容で整列

### 7.4 生成物（レポート）

- 各構造物:
  - 予測ランク
  - 上位シナジー集合 top-k（集合→説明文テンプレ）
- 集計:
  - ランク別に頻出シナジー集合
  - Bridge/Culvert別の差

---

## 8. リポジトリ構成案（Python, Poetry前提）

### 8.1 方針（複数データセット対応）

橋梁以外のデータセットでも同じ流れ（クレンジング→特徴量絞り込み→特徴マスクで学習/評価→ゲームテーブル作成）を回すため、
設定ファイルは **データセット単位**でまとめ、実験ごとの差分は `experiments/` 配下で管理する。

  - すべての「成果物（outputs）」は `outputs/<dataset_key>/...` に集約する
- ゲームテーブルは **常に CSV** で保存する（外部ツールで扱いやすくする）
- 学習済みモデルも `outputs/<dataset_key>/models/` に保存する（再現性・再利用のため）

`dataset_key` は `bridge_conditions` のような識別子で、実験は `bridge_conditions__experiments__baseline10` のようにパス由来で一意化する。

### 8.2 ディレクトリ（推奨）

```
bridge-conditions-ordinal-synergy-xai/
  README.md
  pyproject.toml
  data/
    raw/<dataset_key>/                 # 原本CSV（git管理しない; 1データセット=1フォルダ）
    processed/<dataset_key>/           # 中間テーブル（CSV固定）
  configs/
    datasets/
      <dataset_id>/
        dataset.yml                    # 入出力パス（CSV）、列名、クレンジング/前処理設定
        labeling.yml                   # ターゲット定義（連続/序数/ランク等）
        model.yml                      # 学習設定
        game_table.yml                 # ゲームテーブル設定
        experiments/
          <experiment_id>/
            dataset.yml                # 特徴量絞り込み等の差分（任意）
            labeling.yml               # ラベル差分（任意）
            game_table.yml             # ゲームテーブル差分（任意）
  src/
    bci_osxai/
      __init__.py
      io/
        load_raw.py
      preprocess/
        clean_schema.py
        parse_spans.py
        reshape_bci.py
      features/
        build_features.py
      labels/
        make_ordinal_labels.py
      models/
        train.py
        predict.py
        evaluate.py
      cli/
        main.py
  outputs/
    <dataset_key>/
      game_tables/                     # 常に .csv
      analysis/
      models/
  notebooks/
  tests/
  .gitignore
```

### 8.3 設定ファイルの責務（要点）

- `dataset.yml`: 生データの場所、クレンジング設定、特徴量（列）選択、出力先（processed）を定義
- `labeling.yml`: 目的変数（回帰/分類/序数/ランク）を定義
- `game_table.yml`: ゲームテーブル作成設定（`metric` など）を定義

本リポジトリは **ゲームテーブル生成まで**を責務とし、シナジー計算・説明生成・Power index 等は扱わない。

**ゲームテーブル `value` は常に「大きいほど良い」スコア**とし、回帰系は `inv_mae=1/(1+MAE)` のように 0〜1 へ正規化した指標を使う。

### 8.4 入力フォーマット（CSV固定）

tabular データセットはすべて CSV とする（`io.format: csv`）。

CLI（最小）

- bci-xai preprocess
- bci-xai build-game-table

---

## 9. 運用・再現性

- rawデータは data/raw に置き、git管理しない（.gitignore）
- 生成物（CSV, models, analysis, game tables）は data/processed, outputs/ に出力
- すべての閾値・前処理は configs/ で固定し、実験再現性を担保

---

## 10. READMEに書くべき最小事項（初版）

- データ入手元（Ontario Bridge conditions）
- セットアップ（Poetry）
- 実行手順（preprocess → build-game-table）
- 出力例（game_table.csv の列定義）
