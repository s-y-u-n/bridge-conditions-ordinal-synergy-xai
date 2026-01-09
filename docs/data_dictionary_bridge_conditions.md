# Bridge conditions CSV: データ仕様（データ辞書）

対象: `data/raw/bridge_conditions.csv`（前処理後: `data/processed/bridge_conditions_clean.csv`）

## 1. 概要

- 本ファイルは Ontario “Bridge conditions” CSV の列定義（データ辞書）です。
- `configs/datasets/bridge_conditions/dataset.yml` では、このCSVの列名をキーとして前処理・特徴量作成に利用します。
- `_id` は行番号（連番）で、公開されているデータ辞書には載っていない補助列です（本リポジトリでは入力に含まれているため列として扱います）。

## 2. 列定義

| Variable Name | Variable Label | Variable Definition | Additional Notes | Data Type | Values/Codes |
|---|---|---|---|---|---|
| `_id` | Row ID | CSV内の行番号（1始まりの連番） | 元データ辞書外の補助列 | Integer | 1..N |
| ID (SITE Ndeg) | Site Number | A structure id consists of a county number and a sequential number assigned for that county. This sequential number is known as the site number. |  | Text: Character |  |
| STRUCTURE NAME | Structure Name | Name of the structure |  | Text: Character |  |
| FRENCH NAME | French Name | Name of Structure |  | Text: Character |  |
| HIGHWAY NAME | Highway Name | Name of Highway |  | Text: character |  |
| LATITUDE | Latitude | Identifies the location of the structure based on the coordinate system (Latitude/Longitude) used to locate points on Earth's surface. Measurements are taken at the centre of the structure. |  | Decimal |  |
| LONGITUDE | Longitude | Identifies the location of the structure based on the coordinate system (Latitude/Longitude) used to locate points on Earth's surface. Measurements are taken at the centre of the structure. |  | Decimal |  |
| CATEGORY | Structure Category | A basic description of a structure |  | Text: character | Bridge, Culvert, Tunnel, Retaining Wall |
| SUBCATEGORY 1 | Structure Sub-category 1 | Describes the main element of the structure. In case of a bridge it describes the main longitudinal element of the superstructure. |  | Text:Character | Slab, Frame, Beam/Girder, Truss, Temporary Modular Bridge, Arches, Cable Bridge, Moveable Bridge, Culverts, Tunnel, Retaining Wall, Hybrid, Other |
| TYPE 1 | Type 1 | Describes the main element of the structure in more detail. In case of a bridge it describes in more detail the main longitudinal element of the superstructure. |  | Text:Character | Bowstring Arch, Earth Filled Arch, Sapndrel Arch, Other Arch, Box Beam, Hollow Slab/Adjacent Box Beams, Trapezoidal Girders, Curved Web I Girder (Fluted), AASHTO Girder, CPCI Girder, Rolled I Beam, Plate I Girder, Haunched Plate I Girders, T Beam, T Beam (Boat Type), Rectangular Beam, Other Beam, Cable Stayed, Suspension, Other Cable Bridge, Rigid Frame - Inclined Leg, Rigid Frame - Slab, Rigd Frame - T Beam, Rigid Frame - Box Girder, Rigid Frame - Other, Double Leaf Bascule, Sigle Leaf Bascule, Lift, Swing, Other Moveable Bridge, Circular Voided Slab, Solid Slab, Other Slab Bridge, Acrow Panel, Bailey Panel, Mabey Panel, Other TMB, A Truss, Deck Truss, Half Through Truss (Pony), Through Truss, Other Bridge, Arch Culvert (Open) |
| MATERIAL 1 | Structure Material 1 | Material of the main load carrying memebers of the main structure type. |  | Text:Character | Cast Iron, Corrugated Steel, Steel, Weathering Steel, Mass Concrete, Post-Tensioned Cast-In-Place Concrete, Post-Tensioned Precast Concrete, Prestressed Precast Concrete |
| YEAR BUILT | Year Built | Year the existing structe was built |  | Integer |  |
| LAST MAJOR REHAB | Last Major Rehabilitation | The latest year the structure received major repairs and maintenance work. |  | Integer |  |
| LAST MINOR REHAB | Last Minor Rehabilitation | The latest year the structure received minor maintenance repairs. |  | Integer |  |
| NUMBER OF SPAN / CELLS | Number of Spans/Cells | Amount of spans or cells(unsupported portions of the bridge/culvert between piers and abutments) |  | Integer |  |
| SPAN DETAILS (m) | Span Lengths (m) | This is the length of the spans |  | Text:Character |  |
| DECK / CULVERTS LENGTH (m) | Deck/Culvert Length (m) | Length of the bridge deck or culvert length |  | Decimal |  |
| WIDTH TOTAL (m) | Overall Structure Width (m) | Total width of the structure |  | Decimal |  |
| REGION | Region | MTO's Region Name |  | Text: Character | Central, West, Eastern, Northeasernt, Northwestern |
| COUNTY | County | County Name |  | Text: Character | Brant, Carleton, Elgin, Frontenac, Haldimand, Halton, Hastings, Lanark, Leeds and Grenville, Lennox and Addington, Lincoln, Middlesex, Northumberland and Durham, Ontario, Oxford, Peel, Perth, Prince Edward, Renfrew, Waterloo, Welland, Wellington, Wentworth, York, Cochrane, Kenora, Nipissing, Parry Sound, Rainy River, Sudbury, Timiskaming, Thunder Bay |
| OPERATION STATUS | Operation Status | Operational Status of the Structure |  | Text: Character | Closed to Traffic, Demolished, Not Built Cancelled, Not built pending, Open to pedestrians only, Open to Traffic, Posted for load limit |
| OWNER | Owner | Owner of the Structure |  | Text: Character | Provincial, Municipal, Local Road Board, Ministry of Natural Resources, Federal or International, Other |
| LAST INSPECTION DATE | Last Inspection Date | Date of Last Structural Inspection |  | Date |  |
| CURRENT BCI | Current BCI | BCI (Bridge Condition Index) is a performance measure for bridges. It is a single-number assessment of bridge condition based on the remaining economic worth of a bridge. The BCI is a number from 0 to 100, the higher the number, the better the overall condition. |  | Decimal |  |

## 3. 年次BCI列（2000–2020）

各年の列（`2000`〜`2020`）は、当該年のBCI（0–100）を表します。

- Variable Label: `BCI in Year {YYYY}`
- Variable Definition: `BCI (Bridge Condition Index) ... 0 to 100, the higher the number, the better the overall condition.`
- Data Type: Decimal
