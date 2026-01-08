from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]

def _snake_case(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[()\\[\\]{}]", "", text)
    text = text.replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def load_rename_map_from_data_dictionary(md_path: PathLike, *, year_start: int = 2000, year_end: int = 2020) -> dict[str, str]:
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")

    lines = [line.rstrip("\n") for line in text.splitlines()]
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not (stripped.startswith("|") and "|" in stripped[1:]):
            continue
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        rows.append(parts)

    if not rows:
        raise ValueError(f"Could not find a markdown table in {md_path}")

    header = rows[0]
    try:
        name_idx = header.index("Variable Name")
        label_idx = header.index("Variable Label")
    except ValueError as exc:
        raise ValueError(f"Data dictionary table must include 'Variable Name' and 'Variable Label' columns: {md_path}") from exc

    rename_map: dict[str, str] = {}
    for row in rows[2:]:
        if len(row) <= max(name_idx, label_idx):
            continue
        var_name = row[name_idx]
        var_label = row[label_idx]
        if not var_name or var_name == "---":
            continue
        rename_map[var_name] = _snake_case(var_label) or _snake_case(var_name)

    # Stable, more readable overrides for core fields (used throughout the pipeline).
    overrides = {
        "_id": "row_id",
        "ID (SITE Ndeg)": "structure_id",
        "STRUCTURE NAME": "structure_name",
        "FRENCH NAME": "french_name",
        "HIGHWAY NAME": "highway_name",
        "LATITUDE": "latitude",
        "LONGITUDE": "longitude",
        "CATEGORY": "category",
        "SUBCATEGORY 1": "subcategory_1",
        "TYPE 1": "type_1",
        "MATERIAL 1": "material_1",
        "YEAR BUILT": "year_built",
        "LAST MAJOR REHAB": "last_major_rehab",
        "LAST MINOR REHAB": "last_minor_rehab",
        "NUMBER OF SPAN / CELLS": "span_cells_count",
        "SPAN DETAILS (m)": "span_details_m",
        "DECK / CULVERTS LENGTH (m)": "deck_or_culvert_length_m",
        "WIDTH TOTAL (m)": "width_total_m",
        "REGION": "region",
        "COUNTY": "county",
        "OPERATION STATUS": "operation_status",
        "OWNER": "owner",
        "LAST INSPECTION DATE": "last_inspection_date",
        "CURRENT BCI": "current_bci",
    }
    rename_map.update(overrides)

    for year in range(int(year_start), int(year_end) + 1):
        rename_map[str(year)] = f"bci_{year}"

    return rename_map


def clean_multiline_csv(
    input_path: PathLike,
    output_path: PathLike,
    encoding: str = "utf-8",
    rename_map: dict[str, str] | None = None,
) -> None:
    """Rewrite CSV so each record is on one line (remove newlines inside fields)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding=encoding, newline="") as infile:
        reader = csv.reader(infile)
        with output_path.open("w", encoding=encoding, newline="") as outfile:
            writer = csv.writer(outfile)
            for i, row in enumerate(reader):
                cleaned_row = [field.replace("\r", " ").replace("\n", " ").strip() for field in row]
                if i == 0 and rename_map:
                    cleaned_row = [col.lstrip("\ufeff") for col in cleaned_row]
                    cleaned_row = [rename_map.get(col, col) for col in cleaned_row]
                writer.writerow(cleaned_row)
