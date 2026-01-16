# crop: 政策ゲームテーブル（平均収穫量）設計

## 目的

`crop_yield.csv` の観測データから、4つの「政策」フラグ（0/1）の **全16パターン** について、
条件付き平均収穫量（`Yield_tons_per_hectare` の平均）を計算したテーブルを作る。

注意: これは観測データ上の集計であり、因果的な介入効果（do演算）を保証するものではない。

## 前処理（対象の絞り込み）

- `Crop` が最頻出の穀物のみを対象にする（現状のデータでは `Maize`）。

## 政策（プレイヤー）

出力では以下4列がプレイヤー（0/1）になる。

1. `high_rain_region`
   - `Rainfall_mm` を k=2 でクラスタリングし、中心が大きいクラスタを `1` とする
2. `irrigation_used`
   - `Irrigation_Used==True` を `1`
3. `fertilizer_used`
   - `Fertilizer_Used==True` を `1`
4. `improved_soil`
   - 対象データ（最頻出Crop）で `Yield_tons_per_hectare` の平均が最大の `Soil_Type` を「改良土壌」とし、
     `Soil_Type` がそれに一致する場合 `1`、それ以外 `0`

## 出力（CSV）

`outputs/<dataset_key>/game_tables/game_table.csv`

- 4つの政策列（0/1）
- `order`: 4政策の合計（=1の個数）
- `value`: 平均収穫量（`Yield_tons_per_hectare` の平均）
- `abs_value`: `abs(value)`（互換用）
- `metric`: `mean_yield`
- `n_train`: 当該パターンに一致したサンプル数（学習はしないが、互換のため `n_train` に格納）
- `n_test`: 常に `0`
- `seed`: `Rainfall_mm` のクラスタリング乱数（互換のため）

メタ情報（選ばれた Crop / best_soil / 雨量クラスタ中心など）は
`outputs/<dataset_key>/game_tables/game_table__meta.json` に保存する。

## CLI

```bash
poetry run bci-xai build-crop-policy-game-table \
  --dataset-config configs/datasets/crop_policy/dataset.yml
```

任意で Crop を固定できる:

```bash
poetry run bci-xai build-crop-policy-game-table \
  --dataset-config configs/datasets/crop_policy/dataset.yml \
  --crop Maize
```
