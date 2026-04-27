#
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_PROCESSED_DATA_DIR
from app.utils.dongne_paths import DONGNE_RAW_DATA_DIR

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = DONGNE_RAW_DATA_DIR
OUTPUT_DIR = DONGNE_PROCESSED_DATA_DIR
FULLWIDTH_SPACE = "\u3000"

MARKERS = {
    "interest": "\uad00\uc2ec\uc9d1\ub2e8\uc218",
    "telecom": "\ud1b5\uc2e0\uc815\ubcf4",
    "commerce": "\uc0c1\uad8c\ubd84\uc11d",
    "admin": "\uc11c\uc6b8\uc2dc_\ud589\uc815\ub3d9",
    "population": "5\uc138\ubcc4_\uc778\uad6c",
}

SPECIAL_SOURCE_TO_TARGETS: Dict[Tuple[str, str], List[Tuple[str, str]]] = {
    ("\uac15\ub0a8\uad6c", "\uc77c\uc6d02\ub3d9"): [("\uac15\ub0a8\uad6c", "\uac1c\ud3ec3\ub3d9")],
    ("\uac15\ub3d9\uad6c", "\uc0c1\uc77c\ub3d9"): [
        ("\uac15\ub3d9\uad6c", "\uc0c1\uc77c\uc81c1\ub3d9"),
        ("\uac15\ub3d9\uad6c", "\uc0c1\uc77c\uc81c2\ub3d9"),
    ],
    ("\ub3d9\ub300\ubb38\uad6c", "\uc6a9\uc2e0\ub3d9"): [
        ("\ub3d9\ub300\ubb38\uad6c", "\uc2e0\uc124\ub3d9"),
        ("\ub3d9\ub300\ubb38\uad6c", "\uc6a9\ub450\ub3d9"),
    ],
}


def find_source_file(marker: str) -> Path:
    for path in DATA_DIR.glob("*.csv"):
        if marker in path.name and not path.name.startswith("new_"):
            return path
    raise FileNotFoundError(marker)


INTEREST_FILE = find_source_file(MARKERS["interest"])
TELECOM_FILE = find_source_file(MARKERS["telecom"])
COMMERCE_FILE = find_source_file(MARKERS["commerce"])
ADMIN_FILE = find_source_file(MARKERS["admin"])
POPULATION_FILE = find_source_file(MARKERS["population"])
SUBWAY_FILE = DATA_DIR / "subway.csv"


def clean_text(value: str | None) -> str:
    return (value or "").strip()


def parse_number(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    return float(text)


def format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def normalize_dong_name(name: str) -> str:
    text = clean_text(name)
    text = text.replace(FULLWIDTH_SPACE, "")
    text = text.replace(" ", "")
    text = text.replace("\u00b7", "")
    text = text.replace("\u318d", "")
    text = text.replace("?", "")
    text = text.replace(".", "")
    text = text.replace(",", "")
    for digit in "0123456789":
        text = text.replace("\uc81c" + digit, digit)
    return text


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def stringify_rows(rows: List[Dict[str, object]], fieldnames: List[str]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    for row in rows:
        converted: Dict[str, str] = {}
        for field in fieldnames:
            value = row.get(field, "")
            if isinstance(value, float):
                converted[field] = format_number(value)
            else:
                converted[field] = str(value) if value is not None else ""
        output.append(converted)
    return output


def load_admin_master() -> List[Dict[str, str]]:
    rows = read_csv_rows(ADMIN_FILE)
    return sorted(rows, key=lambda r: (r["district_name"], r["admin_dong_name"]))


def build_admin_lookup(admin_rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    lookup: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in admin_rows:
        key = (row["district_name"], normalize_dong_name(row["admin_dong_name"]))
        lookup[key].append(row)
    return lookup


def admin_key(row: Dict[str, str]) -> Tuple[str, str]:
    return (row["district_name"], row["admin_dong_name"])


def match_targets(
    district_name: str,
    dong_name: str,
    admin_lookup: Dict[Tuple[str, str], List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    special = SPECIAL_SOURCE_TO_TARGETS.get((district_name, dong_name))
    if special:
        matched: List[Dict[str, str]] = []
        for dist, dong in special:
            matched.extend(admin_lookup.get((dist, normalize_dong_name(dong)), []))
        return matched
    return admin_lookup.get((district_name, normalize_dong_name(dong_name)), [])


def build_population_output(admin_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    population_map: Dict[Tuple[str, str], Dict[str, object]] = {}
    current_district = ""
    with POPULATION_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            raw_name = row[0]
            indent = raw_name.count(FULLWIDTH_SPACE)
            name = raw_name.replace(FULLWIDTH_SPACE, "").strip()
            if indent == 3:
                current_district = name
                continue
            if indent != 6 or not current_district:
                continue
            population_map[(current_district, name)] = {
                "district_name": current_district,
                "admin_dong_name": name,
                "metric_name": clean_text(row[1]),
                "metric_value": clean_text(row[2]),
                "total_population": parse_number(row[2]),
            }

    output: List[Dict[str, object]] = []
    for row in admin_rows:
        key = admin_key(row)
        src = population_map[key]
        output.append(
            {
                "admin_dong_code": row["admin_dong_code"],
                "city_name": row["city_name"],
                "district_name": row["district_name"],
                "admin_dong_name": row["admin_dong_name"],
                "metric_name": src["metric_name"],
                "metric_value": src["metric_value"],
                "total_population": src["total_population"],
                "fill_strategy": "source_exact",
            }
        )
    return output


def build_subway_output(admin_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = read_csv_rows(SUBWAY_FILE)
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    line_sets: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    station_sets: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        district = clean_text(row["district_name"])
        dong = clean_text(row["admin_dong_name"])
        if not district or not dong:
            continue
        key = (district, dong)
        target = grouped.setdefault(
            key,
            {
                "district_name": district,
                "admin_dong_name": dong,
                "subway_station_count": 0.0,
                "subway_line_count": 0.0,
                "subway_commute_congestion_sum": 0.0,
                "subway_evening_congestion_sum": 0.0,
                "subway_commute_congestion_max": 0.0,
                "subway_evening_congestion_max": 0.0,
            },
        )
        station_name = clean_text(row["출발역"])
        line_name = clean_text(row["호선"])
        station_sets[key].add(station_name)
        line_sets[key].add(line_name)
        commute = parse_number(row["출근시간_혼잡도합"])
        evening = parse_number(row["퇴근시간_혼잡도합"])
        target["subway_commute_congestion_sum"] += commute
        target["subway_evening_congestion_sum"] += evening
        target["subway_commute_congestion_max"] = max(target["subway_commute_congestion_max"], commute)
        target["subway_evening_congestion_max"] = max(target["subway_evening_congestion_max"], evening)

    output: List[Dict[str, object]] = []
    for row in admin_rows:
        key = admin_key(row)
        src = grouped.get(
            key,
            {
                "district_name": row["district_name"],
                "admin_dong_name": row["admin_dong_name"],
                "subway_commute_congestion_sum": 0.0,
                "subway_evening_congestion_sum": 0.0,
                "subway_commute_congestion_max": 0.0,
                "subway_evening_congestion_max": 0.0,
            },
        )
        station_count = float(len(station_sets.get(key, set())))
        line_count = float(len(line_sets.get(key, set())))
        output.append(
            {
                "admin_dong_code": row["admin_dong_code"],
                "city_name": row["city_name"],
                "district_name": row["district_name"],
                "admin_dong_name": row["admin_dong_name"],
                "subway_station_count": station_count,
                "subway_line_count": line_count,
                "subway_commute_congestion_sum": src["subway_commute_congestion_sum"],
                "subway_commute_congestion_avg": src["subway_commute_congestion_sum"] / station_count if station_count else 0.0,
                "subway_commute_congestion_max": src["subway_commute_congestion_max"],
                "subway_evening_congestion_sum": src["subway_evening_congestion_sum"],
                "subway_evening_congestion_avg": src["subway_evening_congestion_sum"] / station_count if station_count else 0.0,
                "subway_evening_congestion_max": src["subway_evening_congestion_max"],
                "fill_strategy": "source_exact",
            }
        )
    return output


def build_interest_output(admin_rows: List[Dict[str, str]], admin_lookup: Dict[Tuple[str, str], List[Dict[str, str]]]) -> List[Dict[str, object]]:
    rows = read_csv_rows(INTEREST_FILE)
    excluded = {"행정동코드", "자치구", "행정동명", "성별", "연령대", ""}
    numeric_cols = [col for col in rows[0].keys() if col not in excluded]
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    mapping_type: Dict[Tuple[str, str], str] = {}

    for row in rows:
        district = clean_text(row["자치구"])
        dong = clean_text(row["행정동명"])
        targets = match_targets(district, dong, admin_lookup)
        for target_row in targets:
            key = admin_key(target_row)
            target = grouped.setdefault(
                key,
                {
                    "admin_dong_code": target_row["admin_dong_code"],
                    "city_name": target_row["city_name"],
                    "district_name": target_row["district_name"],
                    "admin_dong_name": target_row["admin_dong_name"],
                    "source_admin_dong_name": dong,
                    "source_code_7digit": clean_text(row["행정동코드"]),
                },
            )
            mapping_type[key] = "source_split_alias" if len(targets) > 1 or dong != target_row["admin_dong_name"] else "source_exact_or_normalized"
            for col in numeric_cols:
                target[col] = target.get(col, 0.0) + parse_number(row[col])

    district_means: Dict[str, Dict[str, float]] = defaultdict(dict)
    for district in sorted({row["district_name"] for row in admin_rows}):
        district_rows = [grouped[admin_key(r)] for r in admin_rows if r["district_name"] == district and admin_key(r) in grouped]
        for col in numeric_cols:
            district_means[district][col] = sum(parse_number(r.get(col)) for r in district_rows) / len(district_rows)

    output: List[Dict[str, object]] = []
    for row in admin_rows:
        key = admin_key(row)
        if key in grouped:
            out_row = dict(grouped[key])
            out_row["fill_strategy"] = mapping_type[key]
        else:
            out_row = {
                "admin_dong_code": row["admin_dong_code"],
                "city_name": row["city_name"],
                "district_name": row["district_name"],
                "admin_dong_name": row["admin_dong_name"],
                "source_admin_dong_name": "",
                "source_code_7digit": "",
                "fill_strategy": "district_mean_fill",
            }
            for col in numeric_cols:
                out_row[col] = district_means[row["district_name"]][col]
        output.append(out_row)
    return output


def build_telecom_output(admin_rows: List[Dict[str, str]], admin_lookup: Dict[Tuple[str, str], List[Dict[str, str]]]) -> List[Dict[str, object]]:
    rows = read_csv_rows(TELECOM_FILE)
    excluded = {"행정동코드", "자치구", "행정동", "성별", "연령대"}
    numeric_cols = [col for col in rows[0].keys() if col not in excluded]
    sum_cols = {
        col
        for col in numeric_cols
        if col in {"총인구수", "1인가구수"} or "인구수" in col or "인구 수" in col
    }
    avg_cols = [col for col in numeric_cols if col not in sum_cols]

    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    weighted_sums: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: Dict[Tuple[str, str], float] = defaultdict(float)
    mapping_type: Dict[Tuple[str, str], str] = {}

    for row in rows:
        district = clean_text(row["자치구"])
        dong = clean_text(row["행정동"])
        targets = match_targets(district, dong, admin_lookup)
        for target_row in targets:
            key = admin_key(target_row)
            target = grouped.setdefault(
                key,
                {
                    "admin_dong_code": target_row["admin_dong_code"],
                    "city_name": target_row["city_name"],
                    "district_name": target_row["district_name"],
                    "admin_dong_name": target_row["admin_dong_name"],
                    "source_admin_dong_name": dong,
                    "source_code_7digit": clean_text(row["행정동코드"]),
                },
            )
            mapping_type[key] = "source_split_alias" if len(targets) > 1 or dong != target_row["admin_dong_name"] else "source_exact_or_normalized"
            weight = parse_number(row["총인구수"])
            weights[key] += weight
            for col in sum_cols:
                target[col] = target.get(col, 0.0) + parse_number(row[col])
            for col in avg_cols:
                weighted_sums[key][col] += parse_number(row[col]) * weight

    for key, target in grouped.items():
        weight = weights[key]
        for col in avg_cols:
            target[col] = weighted_sums[key][col] / weight if weight else 0.0

    district_means: Dict[str, Dict[str, float]] = defaultdict(dict)
    for district in sorted({row["district_name"] for row in admin_rows}):
        district_rows = [grouped[admin_key(r)] for r in admin_rows if r["district_name"] == district and admin_key(r) in grouped]
        for col in numeric_cols:
            district_means[district][col] = sum(parse_number(r.get(col)) for r in district_rows) / len(district_rows)

    output: List[Dict[str, object]] = []
    for row in admin_rows:
        key = admin_key(row)
        if key in grouped:
            out_row = dict(grouped[key])
            out_row["fill_strategy"] = mapping_type[key]
        else:
            out_row = {
                "admin_dong_code": row["admin_dong_code"],
                "city_name": row["city_name"],
                "district_name": row["district_name"],
                "admin_dong_name": row["admin_dong_name"],
                "source_admin_dong_name": "",
                "source_code_7digit": "",
                "fill_strategy": "district_mean_fill",
            }
            for col in numeric_cols:
                out_row[col] = district_means[row["district_name"]][col]
        output.append(out_row)
    return output


def build_commerce_output(admin_rows: List[Dict[str, str]], admin_lookup: Dict[Tuple[str, str], List[Dict[str, str]]]) -> List[Dict[str, object]]:
    rows = read_csv_rows(COMMERCE_FILE)
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    mapping_type: Dict[Tuple[str, str], str] = {}

    for row in rows:
        district = clean_text(row["구"])
        dong = clean_text(row["행정동"])
        targets = match_targets(district, dong, admin_lookup)
        for target_row in targets:
            key = admin_key(target_row)
            target = grouped.setdefault(
                key,
                {
                    "admin_dong_code": target_row["admin_dong_code"],
                    "city_name": target_row["city_name"],
                    "district_name": target_row["district_name"],
                    "admin_dong_name": target_row["admin_dong_name"],
                    "source_admin_dong_name": dong,
                    "source_commerce_code": clean_text(row["행정동_코드"]),
                    "기준_년분기_코드": clean_text(row["기준_년분기_코드"]),
                },
            )
            mapping_type[key] = "source_split_alias" if len(targets) > 1 or dong != target_row["admin_dong_name"] else "source_exact_or_normalized"
            service = clean_text(row["서비스_업종_코드_명"])
            for source_col in ["점포_수", "유사_업종_점포_수", "프랜차이즈_점포_수"]:
                value = parse_number(row[source_col])
                target[f"overall__{source_col}"] = target.get(f"overall__{source_col}", 0.0) + value
                target[f"{service}__{source_col}"] = value

    output: List[Dict[str, object]] = []
    for row in admin_rows:
        key = admin_key(row)
        src = grouped[key]
        out_row = dict(src)
        out_row["fill_strategy"] = mapping_type[key]
        output.append(out_row)
    return output


def build_integrated_output(
    admin_rows: List[Dict[str, str]],
    population_rows: List[Dict[str, object]],
    interest_rows: List[Dict[str, object]],
    telecom_rows: List[Dict[str, object]],
    subway_rows: List[Dict[str, object]],
    commerce_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    pop_map = {row["admin_dong_code"]: row for row in population_rows}
    interest_map = {row["admin_dong_code"]: row for row in interest_rows}
    telecom_map = {row["admin_dong_code"]: row for row in telecom_rows}
    subway_map = {row["admin_dong_code"]: row for row in subway_rows}
    commerce_map = {row["admin_dong_code"]: row for row in commerce_rows}

    output: List[Dict[str, object]] = []
    for admin_row in admin_rows:
        code = admin_row["admin_dong_code"]
        row: Dict[str, object] = {
            "admin_dong_code": code,
            "city_name": admin_row["city_name"],
            "district_name": admin_row["district_name"],
            "admin_dong_name": admin_row["admin_dong_name"],
            "address": admin_row["address"],
            "latitude": admin_row["latitude"],
            "longitude": admin_row["longitude"],
            "station_id": admin_row["station_id"],
            "distance_km": admin_row["distance_km"],
        }
        for prefix, source in [
            ("population", pop_map[code]),
            ("interest", interest_map[code]),
            ("telecom", telecom_map[code]),
            ("subway", subway_map[code]),
            ("commerce", commerce_map[code]),
        ]:
            for key, value in source.items():
                if key in {"admin_dong_code", "city_name", "district_name", "admin_dong_name"}:
                    continue
                row[f"{prefix}__{key}"] = value
        output.append(row)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    admin_rows = load_admin_master()
    admin_lookup = build_admin_lookup(admin_rows)

    new_admin_rows: List[Dict[str, object]] = [dict(row) | {"fill_strategy": "source_exact"} for row in admin_rows]
    new_population_rows = build_population_output(admin_rows)
    new_subway_rows = build_subway_output(admin_rows)
    new_interest_rows = build_interest_output(admin_rows, admin_lookup)
    new_telecom_rows = build_telecom_output(admin_rows, admin_lookup)
    new_commerce_rows = build_commerce_output(admin_rows, admin_lookup)
    new_integrated_rows = build_integrated_output(
        admin_rows,
        new_population_rows,
        new_interest_rows,
        new_telecom_rows,
        new_subway_rows,
        new_commerce_rows,
    )

    datasets = [
        (ADMIN_FILE.name, new_admin_rows),
        (POPULATION_FILE.name, new_population_rows),
        (SUBWAY_FILE.name, new_subway_rows),
        (INTEREST_FILE.name, new_interest_rows),
        (TELECOM_FILE.name, new_telecom_rows),
        (COMMERCE_FILE.name, new_commerce_rows),
        ("integrated_admin_dong_data.csv", new_integrated_rows),
    ]

    for source_name, rows in datasets:
        fieldnames = list(rows[0].keys())
        out_name = f"new_{source_name}"
        write_csv(OUTPUT_DIR / out_name, stringify_rows(rows, fieldnames), fieldnames)
        print(f"{out_name}\trows={len(rows)}")


if __name__ == "__main__":
    main()
