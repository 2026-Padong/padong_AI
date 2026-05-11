from fastapi import APIRouter

from app.api.v1.endpoints.dongne import router as dongne_router
from app.api.v1.endpoints.health import router as health_router


api_router = APIRouter()
api_router.include_router(dongne_router, prefix="/dongne", tags=["dongne"])
api_router.include_router(health_router, tags=["health"])
