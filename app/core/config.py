from functools import lru_cache

from pydantic import field_validator
from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Padong AI API"
    DEBUG: bool = True
    DATABASE_URL: str | None = None
    DB_DRIVER: str = "mysql+pymysql"
    DB_HOST: str | None = None
    DB_PORT: int = 3306
    DB_NAME: str | None = None
    DB_USER: str | None = None
    DB_PASSWORD: str | None = None
    BACKEND_LOG_SYNC_URL: str | None = None
    BACKEND_LOG_SYNC_TOKEN: str | None = None
    BACKEND_LOG_SYNC_TIMEOUT_SEC: float = 30.0

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production"}:
                return False
        return value

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def parse_database_url(cls, value: object) -> object:
        if isinstance(value, str) and value.startswith("DATABASE_URL="):
            return value.split("=", 1)[1]
        return value

    @model_validator(mode="after")
    def build_database_url(self) -> "Settings":
        if self.DATABASE_URL:
            return self

        required = {
            "DB_HOST": self.DB_HOST,
            "DB_NAME": self.DB_NAME,
            "DB_USER": self.DB_USER,
            "DB_PASSWORD": self.DB_PASSWORD,
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise ValueError(
                "DATABASE_URL이 없으면 "
                + ", ".join(missing)
                + " 환경변수를 설정해야 합니다."
            )

        self.DATABASE_URL = (
            f"{self.DB_DRIVER}://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
