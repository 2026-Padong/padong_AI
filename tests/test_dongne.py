import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from app.db.session import get_engine
from app.schemas.dongne import DongRecommendationResponse
from app.services import dongne_service
from scripts.recommendation import build_pair_dataset
from scripts.recommendation import sync_backend_logs


client = TestClient(app)


def test_get_dong_recommendations(monkeypatch, tmp_path: Path) -> None:
    expected = DongRecommendationResponse(user_type="핫플 탐험가형", recommendations=[1144066000, 1144058500, 1168058000])
    db_path = tmp_path / "test_recommendations.db"

    def fake_recommend_dongs(payload):
        assert payload.user_id is None
        assert payload.q1 == 5
        assert payload.q10 == 1
        return expected

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(dongne_service, "recommend_dongs", fake_recommend_dongs)

    response = client.get(
        "/api/v1/dongne/recommendations",
        params={
            "q1": 5,
            "q2": 4,
            "q3": 3,
            "q4": 2,
            "q5": 1,
            "q6": 5,
            "q7": 4,
            "q8": 3,
            "q9": 2,
            "q10": 1,
        },
    )

    assert response.status_code == 200
    assert response.json() == expected.model_dump(mode="json")


def test_get_dong_recommendations_validates_query_params() -> None:
    response = client.get(
        "/api/v1/dongne/recommendations",
        params={
            "q1": 0,
            "q2": 4,
            "q3": 3,
            "q4": 2,
            "q5": 1,
            "q6": 5,
            "q7": 4,
            "q8": 3,
            "q9": 2,
            "q10": 1,
        },
    )

    assert response.status_code == 422


def test_recommend_dongs_returns_recommendations_without_writing_logs(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "padong_ai.db"
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{db_path}")
    dongne_service.load_region_profiles.cache_clear()

    dong_df = pd.DataFrame(
        [
            {"행정동코드": 1111111111, "자치구": "강남구", "행정동명": "역삼동", "총인구": 1200, "대표유형": "고립위험형", "single_household_profile": 0.5, "settled_profile": 0.1, "social_profile": 0.2, "service_profile": 0.3, "residential_stability_profile": 0.1, "youth_mobile_profile": 0.4, "cost_risk_profile": 0.0, "isolation_risk_profile": 0.0},
            {"행정동코드": 2222222222, "자치구": "강남구", "행정동명": "삼성동", "총인구": 1400, "대표유형": "정주고착형", "single_household_profile": 0.4, "settled_profile": 0.2, "social_profile": 0.2, "service_profile": 0.1, "residential_stability_profile": 0.0, "youth_mobile_profile": 0.2, "cost_risk_profile": 0.0, "isolation_risk_profile": 0.0},
            {"행정동코드": 3333333333, "자치구": "마포구", "행정동명": "서교동", "총인구": 1500, "대표유형": "1인가구집중형", "single_household_profile": 0.6, "settled_profile": 0.2, "social_profile": 0.3, "service_profile": 0.5, "residential_stability_profile": 0.0, "youth_mobile_profile": 0.5, "cost_risk_profile": 0.0, "isolation_risk_profile": 0.0},
            {"행정동코드": 4444444444, "자치구": "송파구", "행정동명": "잠실동", "총인구": 1600, "대표유형": "재정부담형", "single_household_profile": 0.3, "settled_profile": 0.4, "social_profile": 0.2, "service_profile": 0.2, "residential_stability_profile": 0.2, "youth_mobile_profile": 0.1, "cost_risk_profile": 0.0, "isolation_risk_profile": 0.0},
        ]
    )
    monkeypatch.setattr(
        dongne_service,
        "load_region_profiles",
        lambda _base_dir, _database_url: {"dong": dong_df},
    )

    payload = dongne_service.DongneRecommendationRequest(
        q1=5,
        q2=4,
        q3=3,
        q4=2,
        q5=1,
        q6=5,
        q7=4,
        q8=3,
        q9=2,
        q10=1,
    )

    response = dongne_service.recommend_dongs(payload, config=dongne_service.RecommenderConfig(top_dong=3))

    assert response.user_type
    assert len(response.recommendations) == 3

    with sqlite3.connect(db_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'user_recommendation_logs'"
        ).fetchall()

    assert tables == []


def test_load_region_profiles_bootstraps_once_and_then_reads_db(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "bootstrap.db"
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{db_path}")
    dongne_service.load_region_profiles.cache_clear()

    dong_df = pd.DataFrame(
        [
            {"행정동코드": 1111111111, "자치구": "강남구", "행정동명": "역삼동", "총인구": 1200, "single_household_profile": 0.5},
        ]
    )
    call_count = {"count": 0}

    def fake_build(_base_dir: str) -> dict[str, pd.DataFrame]:
        call_count["count"] += 1
        return {"dong": dong_df}

    monkeypatch.setattr(dongne_service, "_build_region_profiles_from_csv", fake_build)

    first = dongne_service.load_region_profiles(str(tmp_path), settings.DATABASE_URL)
    second = dongne_service.load_region_profiles(str(tmp_path), settings.DATABASE_URL)

    assert call_count["count"] == 1
    assert first["dong"].iloc[0]["행정동코드"] == 1111111111
    assert second["dong"].iloc[0]["자치구"] == "강남구"


def test_save_interactions_endpoint_removed() -> None:
    response = client.post(
        "/api/v1/dongne/interactions",
        json={
            "interactions": [
                {
                    "user_id": 1,
                    "admin_dong_code": 1111111111,
                    "clicked_count": 1,
                    "liked_count": 1,
                    "dwell_time_sec": 42.5,
                    "impression_count": 1,
                }
            ]
        },
    )

    assert response.status_code == 404


def test_build_pair_dataset_reads_logs_from_database(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "dataset.db"
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{db_path}")

    engine = get_engine(settings.DATABASE_URL)
    log_frame = pd.DataFrame(
        [
            {
                "user_id": 1,
                "created_at": "2026-04-27",
                "admin_dong_code": "1111111111",
                "rank_position": 1,
                "impression_count": 2,
                "clicked_count": 1,
                "liked_count": 0,
                "dwell_time_sec": 30.0,
                "q1": 5,
                "q2": 4,
                "q3": 3,
                "q4": 2,
                "q5": 1,
                "q6": 5,
                "q7": 4,
                "q8": 3,
                "q9": 2,
                "q10": 1,
            }
        ]
    )
    log_frame.to_sql("user_recommendation_logs", con=engine, if_exists="replace", index=False)

    monkeypatch.setattr(
        build_pair_dataset.ml_utils,
        "load_profile_lookup",
        lambda: {
            "1111111111": {
                "district_name": "강남구",
                "admin_dong_name": "역삼동",
            }
        },
    )
    monkeypatch.setattr(
        build_pair_dataset.ml_utils,
        "parse_answers_from_log_row",
        lambda _row: {
            "weekend_activity": 5.0,
            "contact_style": 4.0,
            "call_vs_text": 3.0,
            "finance_interest": 2.0,
            "commute_crowd_tolerance": 1.0,
            "shopping_style": 5.0,
            "cooking_vs_delivery": 4.0,
            "video_consumption": 3.0,
            "mobility_radius": 2.0,
            "hotplace_preference": 1.0,
        },
    )
    monkeypatch.setattr(
        build_pair_dataset.rr,
        "classify_user_type",
        lambda _answers: {"type_key": "balanced_allrounder", "type_label": "균형", "dimensions": {}, "type_fit_score": 0.5},
    )
    monkeypatch.setattr(
        build_pair_dataset.ml_utils,
        "build_candidate_features",
        lambda _answers, _profile_row, type_result=None: {
            "district_name": "강남구",
            "admin_dong_name": "역삼동",
            "predicted_type_key": "balanced_allrounder",
            "predicted_type_label": "균형",
            "feature_a": 0.1,
        },
    )

    rows = build_pair_dataset.read_log_rows_from_database()

    assert len(rows) == 1
    assert str(rows[0]["created_at"]) == "2026-04-27"
    assert rows[0]["clicked_count"] == 1
    assert str(rows[0]["admin_dong_code"]) == "1111111111"


def test_sync_backend_logs_fetches_and_saves_interactions(monkeypatch, tmp_path: Path) -> None:
    db_path = tmp_path / "sync_logs.db"
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setattr(settings, "BACKEND_LOG_SYNC_URL", "https://backend.example.com/api/v1/recommendation/logs")
    monkeypatch.setattr(settings, "BACKEND_LOG_SYNC_TOKEN", "secret-token")
    monkeypatch.setattr(settings, "BACKEND_LOG_SYNC_TIMEOUT_SEC", 5.0)

    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "interactions": [
                    {
                        "user_id": 9,
                        "admin_dong_code": 1111111111,
                        "clicked_count": 2,
                        "liked_count": 1,
                        "dwell_time_sec": 19.5,
                        "impression_count": 3,
                        "rank_position": 1,
                        "q1": 5,
                        "q2": 4,
                        "q3": 3,
                        "q4": 2,
                        "q5": 1,
                        "q6": 5,
                        "q7": 4,
                        "q8": 3,
                        "q9": 2,
                        "q10": 1,
                    }
                ]
            }

    class DummyClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self) -> "DummyClient":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url: str, params: dict[str, str], headers: dict[str, str]) -> DummyResponse:
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return DummyResponse()

    monkeypatch.setattr(sync_backend_logs.httpx, "Client", DummyClient)

    result = sync_backend_logs.sync_backend_interactions(
        reference_time=datetime(2026, 4, 28, 4, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    )

    assert result["requested"] is True
    assert result["sync_date"] == "2026-04-27"
    assert result["fetched_count"] == 1
    assert result["updated_count"] == 1
    assert captured["url"] == "https://backend.example.com/api/v1/recommendation/logs"
    assert captured["headers"] == {"Authorization": "Bearer secret-token"}
    assert captured["params"] == {
        "date": "2026-04-27",
        "from": "2026-04-27T00:00:00+09:00",
        "to": "2026-04-28T00:00:00+09:00",
    }

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT user_id, created_at, admin_dong_code, clicked_count, liked_count, impression_count, dwell_time_sec, q1, q10
            FROM user_recommendation_logs
            """
        ).fetchone()

    assert row == (9, "2026-04-27", 1111111111, 2, 1, 3, 19.5, 5, 1)
