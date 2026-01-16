from __future__ import annotations

from pathlib import Path
from typing import Dict


def infer_dataset_key_from_outputs_path(path: str | Path) -> str | None:
    """Infer dataset key from outputs/<dataset_key>/... style paths."""
    parts = list(Path(path).parts)
    if "outputs" not in parts:
        return None
    i = parts.index("outputs")
    if i + 1 >= len(parts):
        return None
    return str(parts[i + 1])


def feature_label_map(dataset_key: str | None) -> Dict[str, str]:
    """Japanese labels for known datasets (fallback: original name)."""
    if not dataset_key:
        return {}

    key = str(dataset_key)
    # Support experiment keys like crop__experiments__... by using prefix.
    base = key.split("__", 1)[0]

    if base == "wine":
        return {
            "Alcohol": "アルコール",
            "Malicacid": "リンゴ酸",
            "Ash": "灰分",
            "Alcalinity_of_ash": "灰分アルカリ度",
            "Magnesium": "マグネシウム",
            "Total_phenols": "総フェノール",
            "Flavanoids": "フラボノイド",
            "Nonflavanoid_phenols": "非フラボノイドフェノール",
            "Proanthocyanins": "プロアントシアニン",
            "Color_intensity": "色の強度",
            "Hue": "色相",
            "0D280_0D315_of_diluted_wines": "希釈ワイン OD280/OD315",
            "Proline": "プロリン",
        }

    if base == "crop":
        return {
            "Region": "地域",
            "Soil_Type": "土壌タイプ",
            "Rainfall_mm": "降雨量(mm)",
            "Temperature_Celsius": "平均気温(℃)",
            "Fertilizer_Used": "肥料使用",
            "Irrigation_Used": "灌漑使用",
            "Weather_Condition": "天候",
        }

    return {}

