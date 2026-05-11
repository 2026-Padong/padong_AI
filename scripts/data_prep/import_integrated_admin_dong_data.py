from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import create_engine
from sqlalchemy import Text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_PROCESSED_DATA_DIR
from app.utils.s3_csv import read_csv_dataframe
from scripts.recommendation.resident_recommender import SOURCE_COLUMNS
from scripts.recommendation.resident_recommender import SOURCE_TABLE


DEFAULT_INPUT_CSV = DONGNE_PROCESSED_DATA_DIR / "new_integrated_admin_dong_data.csv"


def import_csv_to_database(
    csv_path: Path = DEFAULT_INPUT_CSV,
    *,
    table_name: str = SOURCE_TABLE,
    if_exists: str = "replace",
    database_url: str | None = None,
) -> int:
    frame = read_csv_dataframe(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing_columns}")
    frame = frame.loc[:, SOURCE_COLUMNS].copy()

    if database_url:
        engine = create_engine(database_url, future=True)
    else:
        from app.core.config import settings
        from app.db.session import get_engine

        engine = get_engine(settings.DATABASE_URL)

    text_columns = {column: Text() for column in frame.columns}
    frame.to_sql(table_name, con=engine, if_exists=if_exists, index=False, dtype=text_columns)
    return len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import new_integrated_admin_dong_data.csv into the recommendation database.")
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV), help="CSV path to import.")
    parser.add_argument("--table-name", default=SOURCE_TABLE, help="Destination table name.")
    parser.add_argument("--database-url", help="Database URL. Defaults to app settings / environment variables.")
    parser.add_argument(
        "--if-exists",
        choices=["fail", "replace", "append"],
        default="replace",
        help="Behavior when the destination table already exists.",
    )
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path

    row_count = import_csv_to_database(
        csv_path,
        table_name=args.table_name,
        if_exists=args.if_exists,
        database_url=args.database_url,
    )
    print(f"table={args.table_name}")
    print(f"source_csv={csv_path}")
    print(f"row_count={row_count}")


if __name__ == "__main__":
    main()
