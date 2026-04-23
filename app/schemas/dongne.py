from pydantic import BaseModel, Field


class DongneRecommendationRequest(BaseModel):
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
    gu: str
    dong: str
    match_score: float
    population: int
    type: str
    description: str
    top_traits: list[str]
    caution: str
