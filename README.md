# Bridge Conditions Ordinal Synergy XAI

Ontarioの「Bridge conditions」CSVを読み込み、BCIを序数ラベルに変換して学習・推論し、特徴集合のシナジー説明（序数比較）を出力するためのリポジトリです。

## データ説明

- 生データ（CSV）は `data/raw/<dataset_key>/` 配下に置いてください（git管理しません）。
  - 例: `data/raw/bridge_conditions/bridge_conditions.csv`
- 前処理で生成する中間データは `data/processed/<dataset_key>/` に **CSV固定** で出力します。
  - 例: `data/processed/bridge_conditions/bridge_conditions_clean.csv`
- CSVの列定義（データ辞書）は `docs/data_dictionary_bridge_conditions.md` を参照してください。

## 設定ファイルの置き場（複数データセット対応）

橋梁以外のデータセットも同じ流れで回せるよう、設定は `configs/datasets/<dataset_id>/` 配下にまとめます。
新規データセット追加は `configs/datasets/_template_tabular/` をコピーして開始できます。

## 登録済みデータセット

- `bridge_conditions`（Ontario bridge conditions）
- `credit_g`（CSV）
- `student_placement`（`data/raw/student_placement/train.csv`）: `docs/datasets/student_placement.md`
- `academic_success`（`data/raw/academic_success/dataset.csv`）: `docs/datasets/academic_success.md`
- `wine`（`data/raw/wine/wine.data`）: `docs/datasets/wine.md`
- `crop`（`data/raw/crop/crop_yield.csv`）

## 各データセットの特徴量メモ

### bridge_conditions（baseline10）

`configs/datasets/bridge_conditions/experiments/baseline10/dataset.yml` は baseline 用に特徴量を10個に絞ります（`current_bci` や `bci_20xx` は features に含めません）。

（`age` / `years_since_major_rehab` は `inspection_year`（既定: 2020）からの差分で作る派生特徴です。）

- `category`: 構造物の大分類（例: Bridge / Culvert / Tunnel / Retaining Wall）
- `subcategory_1`: 主構造要素の分類（橋梁なら上部構造の主桁形式など）
- `type_1`: 主構造要素の詳細タイプ（`subcategory_1` をより細かくした分類）
- `material_1`: 主構造（主要耐荷部材）の材料
- `region`: 管理主体（MTO）の地域区分
- `owner`: 所有者区分（例: Provincial / Municipal など）
- `age`: 築年数（`inspection_year - year_built`）
- `years_since_major_rehab`: 最終大規模補修（major rehab）からの経過年数（`inspection_year - last_major_rehab`）
- `deck_length_m`: デッキ（またはカルバート）長さ [m]
- `span_count`: スパン/セル数（`NUMBER OF SPAN / CELLS` から数値を抽出して数値化）

### credit_g

German Credit（信用リスク）データセット。属性から「信用リスクが良い／悪い」を分類する二値分類（`good` / `bad`）。  
誤分類コスト行列が提示されており、実務的には **「bad を good と誤判定する」方が重い**（Bad→Good が 5、Good→Bad が 1）。

- 目的変数: `class`（`good` / `bad`）
- 特徴量（`class` 以外の列をそのまま使用）:
  - 口座・資産: `checking_status`（当座預金の状態）, `savings_status`（貯蓄/債券の状態）, `property_magnitude`（資産区分）, `housing`（住居形態）
  - 借入: `duration`（借入期間/月）, `credit_amount`（借入額）, `purpose`（借入目的）, `installment_commitment`（返済負担率）, `other_payment_plans`（他の分割払い計画）, `existing_credits`（当該銀行での既存クレジット件数）
  - 履歴・属性: `credit_history`（信用履歴）, `employment`（勤続年数区分）, `residence_since`（居住年数）, `age`（年齢）
  - 追加情報: `personal_status`（婚姻/性別カテゴリ）, `other_parties`（共同申請者/保証人）, `job`（職種）, `num_dependents`（扶養人数）, `own_telephone`（電話保有）, `foreign_worker`（外国人労働者か）

### student_placement

大学での就職活動を題材に、学業・技術・適性・ソフトスキル等の属性から「就職するか／しないか」を予測する二値分類向けデータセット（初心者のEDA・特徴量エンジニアリング・モデル比較に適する）。  
各レコードは論理的制約に基づいて生成され、例えば「CGPA → スキル → インターン経験 → 就職」といった教育・就職の現実的な関係を反映する。

- 目的変数: `Placement_Status`（`Placed` / `Not Placed`）
- ID: `Student_ID`（前処理では `structure_id` として保持）
- 特徴量（`Placement_Status` 以外の列をそのまま使用）:
  - 基本属性: `Age`（年齢）, `Gender`（性別）, `Degree`（学位）, `Branch`（専攻）
  - 学業: `CGPA`（成績）
  - 経験: `Internships`（インターン回数）, `Projects`（プロジェクト数）, `Certifications`（資格数）
  - スキル: `Coding_Skills`（コーディング）, `Communication_Skills`（コミュニケーション）, `Soft_Skills_Rating`（ソフトスキル）
  - 適性: `Aptitude_Test_Score`（適性テスト）
  - 学業リスク: `Backlogs`（未修得/再履修の数）

### academic_success

高等教育機関の学生について、入学時点の情報（デモグラフィック/社会経済/入学経路/コース等）と、1・2学期の履修状況/成績、地域経済指標から、在籍状況（退学/在籍/卒業）を予測するデータセット。

- 目的変数: `Target`（`Dropout` / `Enrolled` / `Graduate`）
- 特徴量（多くはカテゴリを整数で符号化した値）:
  - `Marital status`: 婚姻状況（カテゴリ）
  - `Application mode`: 出願方法（カテゴリ）
  - `Application order`: 出願順位/希望順位（数値）
  - `Course`: 履修コース/学科（カテゴリ）
  - `Daytime/evening attendance`: 昼間/夜間の通学区分（カテゴリ）
  - `Previous qualification`: 入学前の最終学歴/資格（カテゴリ）
  - `Nacionality`: 国籍（カテゴリ）
  - `Mother's qualification`: 母の学歴/資格（カテゴリ）
  - `Father's qualification`: 父の学歴/資格（カテゴリ）
  - `Mother's occupation`: 母の職業（カテゴリ）
  - `Father's occupation`: 父の職業（カテゴリ）
  - `Displaced`: 住居移動（転居等）に関する区分（フラグ）
  - `Educational special needs`: 特別支援の必要有無（フラグ）
  - `Debtor`: 債務者（滞納等）フラグ（フラグ）
  - `Tuition fees up to date`: 授業料の支払状況（未滞納か）（フラグ）
  - `Gender`: 性別（カテゴリ）
  - `Scholarship holder`: 奨学金受給の有無（フラグ）
  - `Age at enrollment`: 入学時年齢（数値）
  - `International`: 留学生フラグ（フラグ）
  - `Curricular units 1st sem (credited)`: 1学期の認定単位数（数値）
  - `Curricular units 1st sem (enrolled)`: 1学期の履修登録単位数（数値）
  - `Curricular units 1st sem (evaluations)`: 1学期の評価（試験等）対象数（数値）
  - `Curricular units 1st sem (approved)`: 1学期の合格（承認）数（数値）
  - `Curricular units 1st sem (grade)`: 1学期の成績（平均等）（数値）
  - `Curricular units 1st sem (without evaluations)`: 1学期の未評価数（数値）
  - `Curricular units 2nd sem (credited)`: 2学期の認定単位数（数値）
  - `Curricular units 2nd sem (enrolled)`: 2学期の履修登録単位数（数値）
  - `Curricular units 2nd sem (evaluations)`: 2学期の評価（試験等）対象数（数値）
  - `Curricular units 2nd sem (approved)`: 2学期の合格（承認）数（数値）
  - `Curricular units 2nd sem (grade)`: 2学期の成績（平均等）（数値）
  - `Curricular units 2nd sem (without evaluations)`: 2学期の未評価数（数値）
  - `Unemployment rate`: 失業率（地域経済指標）（数値）
  - `Inflation rate`: インフレ率（地域経済指標）（数値）
  - `GDP`: GDP（地域経済指標）（数値）

academic_success の「10特徴量セット（フルゲームテーブル用）」:

- `Course`: 履修コース/学科（カテゴリ）
- `Previous qualification`: 入学前の最終学歴/資格（カテゴリ）
- `Mother's qualification`: 母の学歴/資格（カテゴリ）
- `Father's qualification`: 父の学歴/資格（カテゴリ）
- `Scholarship holder`: 奨学金受給の有無（フラグ）
- `Curricular units 1st sem (credited)`: 1学期の認定単位数（数値）
- `Curricular units 1st sem (grade)`: 1学期の成績（平均等）（数値）

設定ファイル:

- `configs/datasets/academic_success/experiments/top10/dataset.yml`
- `configs/datasets/academic_success/experiments/top10/game_table.yml`

### wine

UCI Wine データセット（3クラス分類）。化学分析値（13特徴量）からワインの `class`（`C1` / `C2` / `C3`）を予測する（詳細: `docs/datasets/wine.md`）。

- 目的変数: `class`（クラスラベル）
- 特徴量（`class` 以外の列をそのまま使用）:
  - `Alcohol`: アルコール
  - `Malicacid`: リンゴ酸
  - `Ash`: 灰分
  - `Alcalinity_of_ash`: 灰分アルカリ度
  - `Magnesium`: マグネシウム
  - `Total_phenols`: 総フェノール
  - `Flavanoids`: フラボノイド
  - `Nonflavanoid_phenols`: 非フラボノイドフェノール
  - `Proanthocyanins`: プロアントシアニン
  - `Color_intensity`: 色の強度
  - `Hue`: 色相
  - `0D280_0D315_of_diluted_wines`: 希釈ワインの OD280/OD315
  - `Proline`: プロリン

### crop

農業データ（1,000,000 サンプル）から収穫量 `Yield_tons_per_hectare`（トン/ヘクタール）を予測する回帰タスク用データセット。

条件を揃えるため、最頻出の `Crop`（現状: `Maize`）に限定して学習し、また収穫量要因とは言いにくい `Days_to_Harvest` は特徴量から除外する。

- 目的変数: `Yield_tons_per_hectare`（収穫量 [tons/ha]）
- 学習に使う特徴量（7個; `configs/datasets/crop/dataset.yml` で固定）:
  - `Region`: 栽培地域（North / East / South / West）
  - `Soil_Type`: 土壌タイプ（Clay / Sandy / Loam / Silt / Peaty / Chalky）
  - `Rainfall_mm`: 降雨量 [mm]
  - `Temperature_Celsius`: 平均気温 [°C]
  - `Fertilizer_Used`: 肥料使用（True=使用, False=未使用）
  - `Irrigation_Used`: 灌漑使用（True=使用, False=未使用）
  - `Weather_Condition`: 天候（Sunny / Rainy / Cloudy）

### crop_policy（政策 16 パターンの平均収穫量テーブル）

`crop_yield.csv` の観測データから、最頻出の `Crop` に絞った上で、4つの政策フラグ（0/1）の全16パターンについて **平均収穫量** を集計したテーブルを作成する（学習なし）。

- `high_rain_region`: `Rainfall_mm` を k=2 でクラスタリングし、中心が大きいクラスタを 1
- `irrigation_used`: `Irrigation_Used==True` を 1
- `fertilizer_used`: `Fertilizer_Used==True` を 1
- `improved_soil`: 平均収穫量が最大の `Soil_Type` を 1

```bash
poetry run bci-xai build-crop-policy-game-table \
  --dataset-config configs/datasets/crop_policy/dataset.yml
```

## 実験フロー（baseline10 → ゲームテーブル）

1) 実験用データセット作成（features/labels の生成）

```bash
poetry run bci-xai preprocess \
  --dataset-config configs/datasets/bridge_conditions/experiments/baseline10/dataset.yml \
  --labeling-config configs/datasets/bridge_conditions/labeling.yml
```

2) ゲームテーブル作成（特徴マスク学習; 出力は常に `.csv`）

```bash
poetry run bci-xai build-game-table \
  --dataset-config configs/datasets/bridge_conditions/experiments/baseline10/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/bridge_conditions/experiments/baseline10/game_table.yml
```

既定の保存先は `configs/datasets/bridge_conditions/experiments/baseline10/game_table.yml` の `game_table.cache_path` です。

分割によるスコアのブレを抑えたい場合は、`game_table.yml` の `game_table.n_repeats`（複数seed平均）や `game_table.cv_folds`（交差検証）を設定できます。

## 特徴量の分析（前段のEDA）

学習用データ（`features.csv` / `labels.csv`）から、欠損・分布・目的変数との関連度などを `outputs/` 配下へ出力します。

```bash
poetry run bci-xai analyze-dataset \
  --dataset-config configs/datasets/bridge_conditions/experiments/baseline10/dataset.yml \
  --task baseline
```

全データセットをまとめて実行する場合:

```bash
poetry run bci-xai analyze-all-datasets --task baseline --continue-on-error
```

## スコアのベースライン・ランキング・可視化

空集合（無情報）ベースライン `v(∅)` を計算する:

```bash
poetry run bci-xai build-empty-baseline \
  --dataset-config configs/datasets/crop/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/crop/game_table_full.yml
```

wine の場合:

```bash
poetry run bci-xai build-empty-baseline \
  --dataset-config configs/datasets/wine/dataset.yml \
  --task baseline \
  --game-table-config configs/datasets/wine/game_table.yml
```

ゲームテーブルのスコア分布（ヒストグラム）をプロットする（ベースライン線つき）:

```bash
poetry run bci-xai plot-game-table-scores \
  --game-table-csv outputs/crop/game_tables/game_table_full.csv \
  --empty-baseline-csv outputs/crop/game_tables/empty_baseline.csv \
  --out outputs/crop/game_tables/game_table_full__score_distribution.png
```

wine の場合:

```bash
poetry run bci-xai plot-game-table-scores \
  --game-table-csv outputs/wine/game_tables/game_table.csv \
  --empty-baseline-csv outputs/wine/game_tables/empty_baseline.csv \
  --out outputs/wine/game_tables/game_table__score_distribution.png
```

スコアを最適1次元区間分割（連続区間制約つきk-means/Jenks）で弱順序 `Σ1 ≻ Σ2 ≻ …` に分割し、rank（`class_id`）を付与する:

```bash
poetry run bci-xai rank-game-table \
  --game-table-csv outputs/crop/game_tables/game_table_full.csv \
  --score-col value \
  --plot-out outputs/crop/game_tables/game_table_full__score_distribution.png \
  --empty-baseline-csv outputs/crop/game_tables/empty_baseline.csv
```

wine の場合:

```bash
poetry run bci-xai rank-game-table \
  --game-table-csv outputs/wine/game_tables/game_table.csv \
  --score-col value \
  --plot-out outputs/wine/game_tables/game_table__score_distribution.png \
  --empty-baseline-csv outputs/wine/game_tables/empty_baseline.csv
```

提携×特徴量のヒートマップを出力する（上位順; 使われた特徴量を赤で表示）:

```bash
poetry run bci-xai plot-game-table-heatmap \
  --game-table-csv outputs/crop/game_tables/game_table_full.csv \
  --ranked-csv outputs/crop/game_tables/game_table_full__ranked.csv \
  --out outputs/crop/game_tables/game_table_full__heatmap.png \
  --top-n 200
```

rank の生成と同時にヒートマップも出力する場合:

```bash
poetry run bci-xai rank-game-table \
  --game-table-csv outputs/crop/game_tables/game_table_full.csv \
  --score-col value \
  --empty-baseline-csv outputs/crop/game_tables/empty_baseline.csv \
  --ranked-format score-only \
  --heatmap-out outputs/crop/game_tables/game_table_full__heatmap.png
```

### データセットごとの一括パイプライン

上記（空集合ベースライン → ゲームテーブル（任意） → ランク付与 → 分布プロット → ヒートマップ）をまとめて実行する:

```bash
poetry run bci-xai run-dataset-pipeline \
  --dataset-config configs/datasets/crop/dataset.yml \
  --labeling-config configs/datasets/crop/labeling.yml \
  --game-table-config configs/datasets/crop/game_table_full.yml \
  --task baseline
```

既に `game_table.cache_path` が存在する場合は、学習（build-game-table）をスキップして可視化だけ行う:

```bash
poetry run bci-xai run-dataset-pipeline \
  --dataset-config configs/datasets/wine/dataset.yml \
  --skip-preprocess \
  --skip-game-table \
  --game-table-config configs/datasets/wine/game_table.yml \
  --task baseline
```

実行せずに、出力先と手順だけ確認する（dry-run）:

```bash
poetry run bci-xai run-dataset-pipeline \
  --dataset-config configs/datasets/wine/dataset.yml \
  --skip-preprocess \
  --skip-game-table \
  --game-table-config configs/datasets/wine/game_table.yml \
  --task baseline \
  --dry-run
```
