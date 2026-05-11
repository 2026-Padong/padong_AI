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

import pandas as pd


S3_CSV_BUCKET = os.getenv("DONGNE_S3_BUCKET", "padong")
S3_CSV_PREFIX = os.getenv("DONGNE_S3_PREFIX", "padongAI").strip("/")
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


def _basename(path: str | Path) -> str:
    return Path(path).name


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def s3_key_for(path: str | Path) -> str:
    filename = _basename(path)
    return f"{S3_CSV_PREFIX}/{filename}" if S3_CSV_PREFIX else filename


def s3_uri_for(path: str | Path) -> str:
    return f"s3://{S3_CSV_BUCKET}/{s3_key_for(path)}"


def _read_s3_bytes(path: str | Path) -> bytes | None:
    client = _get_boto3_client()
    if client is None:
        return None

    try:
        response = client.get_object(Bucket=S3_CSV_BUCKET, Key=s3_key_for(path))
    except Exception:
        return None

    return response["Body"].read()


def csv_source_exists(path: str | Path) -> bool:
    client = _get_boto3_client()
    if client is not None:
        try:
            client.head_object(Bucket=S3_CSV_BUCKET, Key=s3_key_for(path))
            return True
        except Exception:
            pass

    return Path(path).exists()


def find_csv_path(base_dir: str | Path, marker: str, *, exclude_new: bool = False) -> Path:
    normalized_marker = _normalized_text(marker)
    client = _get_boto3_client()
    if client is not None:
        prefix = f"{S3_CSV_PREFIX}/" if S3_CSV_PREFIX else ""
        try:
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=S3_CSV_BUCKET, Prefix=prefix):
                for item in page.get("Contents", []):
                    name = Path(item["Key"]).name
                    normalized_name = _normalized_text(name)
                    if not name.endswith(".csv"):
                        continue
                    if exclude_new and name.startswith("new_"):
                        continue
                    if normalized_marker in normalized_name:
                        return Path(base_dir) / name
        except Exception:
            pass

    for path in Path(base_dir).glob("*.csv"):
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
    return pd.read_csv(path, **kwargs)


def read_csv_dict_rows(path: str | Path, *, encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    data = _read_s3_bytes(path)
    if data is not None:
        text = data.decode(encoding)
        return list(csv.DictReader(StringIO(text)))

    with Path(path).open("r", encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


@contextmanager
def open_csv_text(path: str | Path, *, encoding: str = "utf-8-sig") -> Iterator[StringIO | TextIO]:
    data = _read_s3_bytes(path)
    if data is not None:
        yield StringIO(data.decode(encoding))
        return

    with Path(path).open("r", encoding=encoding, newline="") as f:
        yield f
