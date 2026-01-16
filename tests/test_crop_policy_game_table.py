import pandas as pd

from bci_osxai.analysis.crop_policy_game_table import CropPolicySpec, build_crop_policy_flags, build_crop_policy_game_table


def test_build_crop_policy_game_table_has_16_patterns_and_means() -> None:
    df = pd.DataFrame(
        {
            "Crop": ["Maize"] * 6 + ["Rice"] * 2,
            "Soil_Type": ["Clay", "Clay", "Sandy", "Sandy", "Clay", "Sandy", "Clay", "Sandy"],
            "Rainfall_mm": [10, 12, 100, 110, 9, 105, 50, 60],
            "Irrigation_Used": [True, False, True, False, True, False, True, False],
            "Fertilizer_Used": [False, False, True, True, False, True, False, True],
            "Yield_tons_per_hectare": [1.0, 2.0, 10.0, 20.0, 3.0, 30.0, 999.0, 999.0],
        }
    )
    spec = CropPolicySpec(rainfall_sample_size=0, rainfall_random_state=0)
    factors, meta = build_crop_policy_flags(df, spec=spec, crop_value="Maize")
    assert meta["crop_selected"] == "Maize"

    players = [spec.out_high_rain_flag, spec.out_irrigation_flag, spec.out_fertilizer_flag, spec.out_improved_soil_flag]
    table = build_crop_policy_game_table(factors, players=players, target_col=spec.target_col)
    assert table.shape[0] == 16

    # best_soil should be Sandy (mean of Maize: Sandy=(10+20+30)/3=20; Clay=(1+2+3)/3=2)
    assert meta["best_soil"] == "Sandy"

    # For a specific pattern: high_rain_region=1, irrigation_used=0, fertilizer_used=1, improved_soil=1
    # In the constructed Maize rows, this corresponds to (Rainfall=110, yield=20) and (Rainfall=105, yield=30).
    row = table[
        (table[spec.out_high_rain_flag] == 1)
        & (table[spec.out_irrigation_flag] == 0)
        & (table[spec.out_fertilizer_flag] == 1)
        & (table[spec.out_improved_soil_flag] == 1)
    ].iloc[0]
    assert int(row["n_rows"]) == 2
    assert abs(float(row["value"]) - 25.0) < 1e-12
