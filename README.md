# Bridge Conditions Ordinal Synergy XAI

Ontarioの「Bridge conditions」CSVを読み込み、BCIを序数ラベルに変換して学習・推論し、特徴集合のシナジー説明（序数比較）を出力するためのリポジトリです。

## データ説明

- 生データ（CSV）は `data/raw/bridge_conditions.csv` に置いてください（git管理しません）。
- 前処理の最初に改行混入を除去したCSVを `data/processed/bridge_conditions_clean.csv` に作成します。
- CSVの列定義（データ辞書）は `docs/data_dictionary_bridge_conditions.md` を参照してください。

## 設定ファイルの置き場（複数データセット対応）

橋梁以外のデータセットも同じ流れで回せるよう、設定は `configs/datasets/<dataset_id>/` 配下にまとめます。
新規データセット追加は `configs/datasets/_template_tabular/` をコピーして開始できます。

## 特徴量説明（baseline10）

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
