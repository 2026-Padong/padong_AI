from datetime import datetime
from typing import Annotated

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import status

from app.schemas.dongne import DongneInteractionBatchRequest
from app.schemas.dongne import DongneInteractionBatchResponse
from app.schemas.dongne import DongRecommendationResponse
from app.schemas.dongne import DongneRecommendationRequest
from app.services import dongne_service


router = APIRouter()


def get_recommendation_request(
    user_id: Annotated[str, Query(min_length=1)],
    session_id: Annotated[str, Query(min_length=1)],
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
    event_at: Annotated[datetime | None, Query()] = None,
) -> DongneRecommendationRequest:
    return DongneRecommendationRequest(
        user_id=user_id,
        session_id=session_id,
        event_at=event_at,
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


@router.get("/recommendations", response_model=DongRecommendationResponse)
def get_dong_recommendations(
    payload: Annotated[DongneRecommendationRequest, Depends(get_recommendation_request)],
) -> DongRecommendationResponse:
    try:
        return dongne_service.recommend_dongs(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/interactions", response_model=DongneInteractionBatchResponse, status_code=status.HTTP_200_OK)
def save_dongne_interactions(payload: DongneInteractionBatchRequest) -> DongneInteractionBatchResponse:
    try:
        return dongne_service.save_interactions(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
