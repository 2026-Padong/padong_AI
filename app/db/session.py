from collections.abc import Generator

from app.core.config import settings


def get_db() -> Generator[None, None, None]:
    """Placeholder dependency for future database sessions."""
    _ = settings.DATABASE_URL
    yield
