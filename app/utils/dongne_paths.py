import os
import unicodedata
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DONGNE_DATA_DIR = PROJECT_ROOT / "data" / "dongne"
DONGNE_RAW_DATA_DIR = DONGNE_DATA_DIR / "raw"
DONGNE_PROCESSED_DATA_DIR = DONGNE_DATA_DIR / "processed"
DONGNE_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "dongne"

DONGNE_S3_BUCKET_ENV = os.getenv("DONGNE_S3_BUCKET", "padong")
DONGNE_S3_PREFIX = os.getenv("DONGNE_S3_PREFIX", "padongAI").strip("/")


def _normalize_s3_bucket(value: str) -> str:
    if value.startswith("arn:aws:s3:::"):
        return value.removeprefix("arn:aws:s3:::").strip("/")
    if value.startswith("s3://"):
        return urlparse(value).netloc
    return value.strip("/")


DONGNE_S3_BUCKET = _normalize_s3_bucket(DONGNE_S3_BUCKET_ENV)
DONGNE_S3_DATA_DIR = f"s3://{DONGNE_S3_BUCKET}/{DONGNE_S3_PREFIX}" if DONGNE_S3_PREFIX else f"s3://{DONGNE_S3_BUCKET}"


def dongne_s3_csv_path(filename: str) -> str:
    return f"{DONGNE_S3_DATA_DIR}/{unicodedata.normalize('NFD', filename)}"


DONGNE_INTEREST_CSV = dongne_s3_csv_path("2025.12월_10개_관심집단수.csv")
DONGNE_TELECOM_CSV = dongne_s3_csv_path("2025.12월_29개_통신정보.csv")
DONGNE_POPULATION_CSV = dongne_s3_csv_path("행정구역_읍면동_별_5세별_인구.csv")
DONGNE_INTEGRATED_CSV = dongne_s3_csv_path("new_integrated_admin_dong_data.csv")
DONGNE_LEGACY_INTEGRATED_CSV = dongne_s3_csv_path("integrated_admin_dong_data.csv")
DONGNE_LIFESTYLE_PROFILE_CSV = dongne_s3_csv_path("admin_dong_lifestyle_profiles.csv")
