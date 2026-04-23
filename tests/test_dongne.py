from fastapi.testclient import TestClient

from app.main import app
from app.schemas.dongne import DongRecommendationResponse
from app.services import dongne_service


client = TestClient(app)


def test_get_dong_recommendations(monkeypatch) -> None:
    expected = [
        DongRecommendationResponse(
            gu="마포구",
            dong="서교동",
            match_score=0.81,
            population=12000,
            type="1인가구집중형",
            description="추천 설명",
            top_traits=["1인가구 친화성이 높고"],
            caution="전체적으로는 사용자 응답과 큰 충돌 없이 맞는 편입니다",
        )
    ]

    def fake_recommend_dongs(payload):
        assert payload.q1 == 5
        assert payload.q10 == 1
        return expected

    monkeypatch.setattr(dongne_service, "recommend_dongs", fake_recommend_dongs)

    response = client.get(
        "/api/v1/dongne/recommendations",
        params={"q1": 5, "q2": 4, "q3": 3, "q4": 2, "q5": 1, "q6": 5, "q7": 4, "q8": 3, "q9": 2, "q10": 1},
    )

    assert response.status_code == 200
    assert response.json() == [item.model_dump(mode="json") for item in expected]


def test_get_dong_recommendations_validates_query_params() -> None:
    response = client.get(
        "/api/v1/dongne/recommendations",
        params={"q1": 0, "q2": 4, "q3": 3, "q4": 2, "q5": 1, "q6": 5, "q7": 4, "q8": 3, "q9": 2, "q10": 1},
    )

    assert response.status_code == 422
