from datetime import datetime

from pydantic import BaseModel, Field


class DongneRecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    event_at: datetime | None = None
    q1: int = Field(ge=1, le=5)
    q2: int = Field(ge=1, le=5)
    q3: int = Field(ge=1, le=5)
    q4: int = Field(ge=1, le=5)
    q5: int = Field(ge=1, le=5)
    q6: int = Field(ge=1, le=5)
    q7: int = Field(ge=1, le=5)
    q8: int = Field(ge=1, le=5)
    q9: int = Field(ge=1, le=5)
    q10: int = Field(ge=1, le=5)


class DongRecommendationResponse(BaseModel):
    user_type: str
    recommendations: list[int]


class DongneInteractionItem(BaseModel):
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    admin_dong_code: int
    clicked: int = Field(default=0, ge=0, le=1)
    liked: int = Field(default=0, ge=0, le=1)
    dwell_time_sec: float = Field(default=0.0, ge=0.0)
    impression: int = Field(default=1, ge=0, le=1)
    rank_position: int | None = Field(default=None, ge=1)
    event_at: datetime | None = None


class DongneInteractionBatchRequest(BaseModel):
    interactions: list[DongneInteractionItem] = Field(min_length=1)


class DongneInteractionBatchResponse(BaseModel):
    updated_count: int
