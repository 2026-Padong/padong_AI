from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import recommendation_ml_utils as ml_utils
import resident_recommender as rr


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "pair_training_dataset.csv"


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build user-admin_dong pair dataset from behavior logs.")
    parser.add_argument("--logs-csv", required=True, help="Behavior log csv path.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output pair dataset csv path.")
    parser.add_argument("--dwell-cap-sec", type=float, default=120.0, help="Cap for dwell-time normalization.")
    args = parser.parse_args()

    logs_csv = Path(args.logs_csv)
    output_csv = Path(args.output_csv)
    if not logs_csv.is_absolute():
        logs_csv = SCRIPT_DIR / logs_csv
    if not output_csv.is_absolute():
        output_csv = SCRIPT_DIR / output_csv
    profile_lookup = ml_utils.load_profile_lookup()
    log_rows = read_rows(logs_csv)

    output_rows: List[Dict[str, object]] = []
    missing_codes: Dict[str, int] = {}

    for log_row in log_rows:
        admin_dong_code = str(log_row.get("admin_dong_code", "")).strip()
        profile_row = profile_lookup.get(admin_dong_code)
        if not profile_row:
            missing_codes[admin_dong_code] = missing_codes.get(admin_dong_code, 0) + 1
            continue

        answers = ml_utils.parse_answers_from_log_row(log_row)
        type_result = rr.classify_user_type(answers)
        feature_row = ml_utils.build_candidate_features(answers, profile_row, type_result=type_result)

        clicked = parse_float(log_row.get("clicked"))
        liked = parse_float(log_row.get("liked"))
        dwell_time_sec = parse_float(log_row.get("dwell_time_sec"))
        impression = parse_float(log_row.get("impression"), default=1.0)
        rank_position = parse_float(log_row.get("rank_position"), default=0.0)

        pair_row: Dict[str, object] = {
            "user_id": log_row.get("user_id", ""),
            "session_id": log_row.get("session_id", ""),
            "event_at": log_row.get("event_at", "") or log_row.get("timestamp", ""),
            "admin_dong_code": admin_dong_code,
            "district_name": feature_row["district_name"],
            "admin_dong_name": feature_row["admin_dong_name"],
            "predicted_type_key": feature_row["predicted_type_key"],
            "predicted_type_label": feature_row["predicted_type_label"],
            "impression": impression,
            "rank_position": rank_position,
            "clicked": clicked,
            "liked": liked,
            "dwell_time_sec": dwell_time_sec,
            "label": ml_utils.compute_label(clicked, liked, dwell_time_sec, dwell_cap_sec=args.dwell_cap_sec),
        }
        pair_row.update(feature_row)
        output_rows.append(pair_row)

    write_rows(output_csv, output_rows)
    print(f"pair_dataset={output_csv.name}")
    print(f"row_count={len(output_rows)}")
    if missing_codes:
        print(f"missing_admin_dong_codes={len(missing_codes)}")


if __name__ == "__main__":
    main()
