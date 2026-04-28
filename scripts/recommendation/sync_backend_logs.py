from __future__ import annotations

from datetime import datetime
from datetime import time
from datetime import timedelta
from zoneinfo import ZoneInfo

import httpx

from app.core.config import settings
from app.schemas.dongne import DongneInteractionBatchRequest
from app.services.dongne_service import save_interactions


KST = ZoneInfo("Asia/Seoul")


def build_sync_window(reference_time: datetime | None = None) -> tuple[datetime, datetime]:
    current = reference_time.astimezone(KST) if reference_time else datetime.now(KST)
    target_date = current.date() - timedelta(days=1)
    window_start = datetime.combine(target_date, time.min, tzinfo=KST)
    window_end = window_start + timedelta(days=1)
    return window_start, window_end


def _extract_interactions(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        if isinstance(payload.get("interactions"), list):
            return [item for item in payload["interactions"] if isinstance(item, dict)]
        if isinstance(payload.get("logs"), list):
            return [item for item in payload["logs"] if isinstance(item, dict)]
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("interactions"), list):
            return [item for item in data["interactions"] if isinstance(item, dict)]

    raise ValueError("백엔드 로그 응답에서 interactions 배열을 찾지 못했습니다.")


def sync_backend_interactions(reference_time: datetime | None = None) -> dict[str, object]:
    if not settings.BACKEND_LOG_SYNC_URL:
        return {
            "requested": False,
            "reason": "BACKEND_LOG_SYNC_URL not configured",
            "fetched_count": 0,
            "updated_count": 0,
        }

    window_start, window_end = build_sync_window(reference_time=reference_time)
    headers: dict[str, str] = {}
    if settings.BACKEND_LOG_SYNC_TOKEN:
        headers["Authorization"] = f"Bearer {settings.BACKEND_LOG_SYNC_TOKEN}"

    with httpx.Client(timeout=settings.BACKEND_LOG_SYNC_TIMEOUT_SEC) as client:
        response = client.get(
            settings.BACKEND_LOG_SYNC_URL,
            params={
                "date": window_start.date().isoformat(),
                "from": window_start.isoformat(),
                "to": window_end.isoformat(),
            },
            headers=headers,
        )
        response.raise_for_status()
        interactions = _extract_interactions(response.json())

    batch = DongneInteractionBatchRequest.model_validate({"interactions": interactions})
    result = save_interactions(batch)
    return {
        "requested": True,
        "sync_date": window_start.date().isoformat(),
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "fetched_count": len(interactions),
        "updated_count": result.updated_count,
    }


def main() -> None:
    print(sync_backend_interactions())


if __name__ == "__main__":
    main()
