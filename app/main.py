from fastapi import FastAPI

from app.api.v1.api import api_router
from app.core.config import settings


app = FastAPI(
    title="Padong AI API",
    version="1.0.0",
    debug=settings.DEBUG,
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["root"])
def read_root() -> dict[str, str]:
    return {"message": f"Welcome to {settings.APP_NAME}"}
