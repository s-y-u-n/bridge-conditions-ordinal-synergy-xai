# Bridge Conditions Ordinal Synergy XAI

Ontarioの「Bridge conditions」CSVを読み込み、BCIを序数ラベルに変換して学習・推論し、特徴集合のシナジー説明（序数比較）を出力するためのリポジトリです。

## データ説明

- 生データ（CSV）は `data/raw/bridge_conditions.csv` に置いてください（git管理しません）。
- 前処理の最初に改行混入を除去したCSVを `data/processed/bridge_conditions_clean.csv` に作成します。
- CSVの列定義（データ辞書）は `docs/data_dictionary_bridge_conditions.md` を参照してください。

## 特徴量説明（baseline10）

`configs/dataset_baseline10.yml` は baseline 用に特徴量を10個に絞ります（`current_bci` や `bci_20xx` は features に含めません）。

- `category`
- `subcategory_1`
- `type_1`
- `material_1`
- `region`
- `owner`
- `age`
- `years_since_major_rehab`
- `deck_length_m`
- `span_count`

## 実験フロー（baseline10 → ゲームテーブル → 指標）

1) 実験用データセット作成（features/labels の生成）

```bash
poetry run bci-xai preprocess --dataset-config configs/dataset_baseline10.yml
```

2) ゲームテーブル作成（特徴マスク学習; 出力は `.parquet`/`.csv`）

```bash
poetry run bci-xai build-game-table --dataset-config configs/dataset_baseline10.yml --task baseline --synergy-config configs/synergy_baseline.yml
poetry run bci-xai build-game-table --dataset-config configs/dataset_baseline10.yml --task baseline --synergy-config configs/synergy_baseline.yml --out artifacts/game_tables/baseline10.csv
```

既定の保存先は `configs/synergy_baseline.yml` の `game_table.cache_path` です。

3) 指標計算（Lex-cel）

`configs/synergy_baseline.yml` の `lexcel.*` を使い、ゲームテーブルから `lexcel` を計算して出力します。

```bash
poetry run bci-xai explain --dataset-config configs/dataset_baseline10.yml --synergy-config configs/synergy_baseline.yml --id <structure_id> --stdout
```
