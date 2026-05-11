from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
from app.db.session import get_engine
from app.utils.dongne_paths import DONGNE_PROCESSED_DATA_DIR
from scripts.recommendation import recommendation_ml_utils as ml_utils
from scripts.recommendation import resident_recommender as rr


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_CSV = DONGNE_PROCESSED_DATA_DIR / "pair_training_dataset.csv"
RECOMMENDATION_LOG_TABLE = "user_recommendation_logs"


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def write_rows(path: str | Path, rows: List[Dict[str, object]]) -> None:
    import csv

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_log_rows_from_database() -> List[Dict[str, object]]:
    query = f"""
    SELECT
        user_id,
        created_at,
        admin_dong_code,
        rank_position,
        impression_count,
        clicked_count,
        liked_count,
        dwell_time_sec,
        q1, q2, q3, q4, q5, q6, q7, q8, q9, q10
    FROM {RECOMMENDATION_LOG_TABLE}
    ORDER BY created_at, user_id, rank_position
    """
    frame = pd.read_sql_query(query, con=get_engine(settings.DATABASE_URL))
    return frame.fillna("").to_dict(orient="records")


def build_pair_rows(dwell_cap_sec: float = 120.0) -> tuple[List[Dict[str, object]], Dict[str, int]]:
    profile_lookup = ml_utils.load_profile_lookup()
    log_rows = read_log_rows_from_database()

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

        clicked_count = parse_float(log_row.get("clicked_count"))
        liked_count = parse_float(log_row.get("liked_count"))
        dwell_time_sec = parse_float(log_row.get("dwell_time_sec"))
        impression_count = parse_float(log_row.get("impression_count"), default=1.0)
        rank_position = parse_float(log_row.get("rank_position"), default=0.0)

        pair_row: Dict[str, object] = {
            "user_id": log_row.get("user_id", ""),
            "created_at": log_row.get("created_at", ""),
            "admin_dong_code": admin_dong_code,
            "district_name": feature_row["district_name"],
            "admin_dong_name": feature_row["admin_dong_name"],
            "predicted_type_key": feature_row["predicted_type_key"],
            "predicted_type_label": feature_row["predicted_type_label"],
            "impression_count": impression_count,
            "rank_position": rank_position,
            "clicked_count": clicked_count,
            "liked_count": liked_count,
            "dwell_time_sec": dwell_time_sec,
            "label": ml_utils.compute_label(clicked_count, liked_count, dwell_time_sec, dwell_cap_sec=dwell_cap_sec),
        }
        pair_row.update(feature_row)
        output_rows.append(pair_row)

    return output_rows, missing_codes


def build_pair_dataset_frame(dwell_cap_sec: float = 120.0) -> pd.DataFrame:
    rows, _ = build_pair_rows(dwell_cap_sec=dwell_cap_sec)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build user-admin_dong pair dataset from database behavior logs.")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output pair dataset csv path.")
    parser.add_argument("--dwell-cap-sec", type=float, default=120.0, help="Cap for dwell-time normalization.")
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute():
        output_csv = PROJECT_ROOT / output_csv
    output_rows, missing_codes = build_pair_rows(dwell_cap_sec=args.dwell_cap_sec)

    write_rows(output_csv, output_rows)
    print(f"pair_dataset={output_csv.name}")
    print(f"row_count={len(output_rows)}")
    if missing_codes:
        print(f"missing_admin_dong_codes={len(missing_codes)}")


if __name__ == "__main__":
    main()
