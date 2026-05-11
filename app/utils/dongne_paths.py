import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DONGNE_DATA_DIR = PROJECT_ROOT / "data" / "dongne"
DONGNE_RAW_DATA_DIR = DONGNE_DATA_DIR / "raw"
DONGNE_PROCESSED_DATA_DIR = DONGNE_DATA_DIR / "processed"
DONGNE_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "dongne"

DONGNE_S3_BUCKET = os.getenv("DONGNE_S3_BUCKET", "padong")
DONGNE_S3_PREFIX = os.getenv("DONGNE_S3_PREFIX", "padongAI").strip("/")
DONGNE_S3_DATA_DIR = f"s3://{DONGNE_S3_BUCKET}/{DONGNE_S3_PREFIX}" if DONGNE_S3_PREFIX else f"s3://{DONGNE_S3_BUCKET}"


def dongne_s3_csv_path(filename: str) -> str:
    return f"{DONGNE_S3_DATA_DIR}/{filename}"
