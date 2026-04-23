from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query

from app.schemas.dongne import DongRecommendationResponse
from app.schemas.dongne import DongneRecommendationRequest
from app.services import dongne_service


router = APIRouter()


def get_recommendation_request(
    q1: Annotated[int, Query(ge=1, le=5)],
    q2: Annotated[int, Query(ge=1, le=5)],
    q3: Annotated[int, Query(ge=1, le=5)],
    q4: Annotated[int, Query(ge=1, le=5)],
    q5: Annotated[int, Query(ge=1, le=5)],
    q6: Annotated[int, Query(ge=1, le=5)],
    q7: Annotated[int, Query(ge=1, le=5)],
    q8: Annotated[int, Query(ge=1, le=5)],
    q9: Annotated[int, Query(ge=1, le=5)],
    q10: Annotated[int, Query(ge=1, le=5)],
) -> DongneRecommendationRequest:
    return DongneRecommendationRequest(
        q1=q1,
        q2=q2,
        q3=q3,
        q4=q4,
        q5=q5,
        q6=q6,
        q7=q7,
        q8=q8,
        q9=q9,
        q10=q10,
    )


@router.get("/recommendations", response_model=list[DongRecommendationResponse])
def get_dong_recommendations(
    payload: Annotated[DongneRecommendationRequest, Depends(get_recommendation_request)],
) -> list[DongRecommendationResponse]:
    try:
        return dongne_service.recommend_dongs(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
