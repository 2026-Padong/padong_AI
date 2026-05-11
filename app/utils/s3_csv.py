from __future__ import annotations

import csv
import os
import unicodedata
from contextlib import contextmanager
from functools import lru_cache
from io import BytesIO
from io import StringIO
from pathlib import Path
from typing import Iterator
from typing import TextIO
from urllib.parse import urlparse

import pandas as pd

from app.utils.dongne_paths import DONGNE_DATA_DIR
from app.utils.dongne_paths import DONGNE_S3_BUCKET
from app.utils.dongne_paths import DONGNE_S3_PREFIX

S3_CSV_BUCKET = DONGNE_S3_BUCKET
S3_CSV_PREFIX = DONGNE_S3_PREFIX
S3_CSV_ENABLED = os.getenv("DONGNE_S3_ENABLED", "true").lower() not in {"0", "false", "no"}


@lru_cache(maxsize=1)
def _get_boto3_client():
    if not S3_CSV_ENABLED:
        return None

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        return None

    return boto3.client(
        "s3",
        config=Config(connect_timeout=1, read_timeout=2, retries={"max_attempts": 1}),
    )


def csv_basename(path: str | Path) -> str:
    parsed = urlparse(str(path))
    if parsed.scheme == "s3":
        return Path(parsed.path).name
    return Path(path).name


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def s3_key_for(path: str | Path) -> str:
    parsed = urlparse(str(path))
    if parsed.scheme == "s3":
        return parsed.path.lstrip("/")

    filename = csv_basename(path)
    return f"{S3_CSV_PREFIX}/{filename}" if S3_CSV_PREFIX else filename


def s3_uri_for(path: str | Path) -> str:
    return f"s3://{S3_CSV_BUCKET}/{s3_key_for(path)}"


def _s3_location_for(path: str | Path) -> tuple[str, str]:
    parsed = urlparse(str(path))
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    return S3_CSV_BUCKET, s3_key_for(path)


def _local_fallback_path(path: str | Path) -> Path:
    raw_path = Path(str(path))
    if raw_path.exists():
        return raw_path

    filename = csv_basename(path)
    for candidate in DONGNE_DATA_DIR.glob(f"**/{filename}"):
        if candidate.is_file():
            return candidate
    return raw_path


def _read_s3_bytes(path: str | Path) -> bytes | None:
    client = _get_boto3_client()
    if client is None:
        return None

    bucket, key = _s3_location_for(path)
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None

    return response["Body"].read()


def csv_source_exists(path: str | Path) -> bool:
    client = _get_boto3_client()
    if client is not None:
        bucket, key = _s3_location_for(path)
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            pass

    return _local_fallback_path(path).exists()


def find_csv_path(base_dir: str | Path, marker: str, *, exclude_new: bool = False) -> str | Path:
    normalized_marker = _normalized_text(marker)
    parsed_base = urlparse(str(base_dir))
    bucket = parsed_base.netloc if parsed_base.scheme == "s3" else S3_CSV_BUCKET
    base_prefix = parsed_base.path.strip("/") if parsed_base.scheme == "s3" else S3_CSV_PREFIX

    client = _get_boto3_client()
    if client is not None:
        prefix = f"{base_prefix}/" if base_prefix else ""
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    name = Path(item["Key"]).name
                    normalized_name = _normalized_text(name)
                    if not name.endswith(".csv"):
                        continue
                    if exclude_new and name.startswith("new_"):
                        continue
                    if normalized_marker in normalized_name:
                        return f"s3://{bucket}/{item['Key']}"
        except Exception:
            pass

    fallback_dirs = [Path(base_dir)] if parsed_base.scheme != "s3" else []
    fallback_dirs.append(DONGNE_DATA_DIR)
    for fallback_dir in fallback_dirs:
        for path in fallback_dir.glob("**/*.csv"):
            normalized_name = _normalized_text(path.name)
            if exclude_new and path.name.startswith("new_"):
                continue
            if normalized_marker in normalized_name:
                return path

    raise FileNotFoundError(marker)


def read_csv_dataframe(path: str | Path, **kwargs) -> pd.DataFrame:
    data = _read_s3_bytes(path)
    if data is not None:
        return pd.read_csv(BytesIO(data), **kwargs)
    return pd.read_csv(_local_fallback_path(path), **kwargs)


def read_csv_dict_rows(path: str | Path, *, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    data = _read_s3_bytes(path)
    if data is not None:
        text = data.decode(encoding)
        return list(csv.DictReader(StringIO(text)))

    with _local_fallback_path(path).open("r", encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


@contextmanager
def open_csv_text(path: str | Path, *, encoding: str = "utf-8-sig") -> Iterator[StringIO | TextIO]:
    data = _read_s3_bytes(path)
    if data is not None:
        yield StringIO(data.decode(encoding))
        return

    with _local_fallback_path(path).open("r", encoding=encoding, newline="") as f:
        yield f
