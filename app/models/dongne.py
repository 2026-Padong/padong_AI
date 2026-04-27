from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import Integer
from sqlalchemy import JSON
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column

from app.db.base import Base


class DongRegionProfile(Base):
    __tablename__ = "dong_region_profiles"

    admin_dong_code: Mapped[int] = mapped_column("admin_dong_code", Integer, primary_key=True)
    district_name: Mapped[str] = mapped_column("district_name", String(100), nullable=False, index=True)
    admin_dong_name: Mapped[str] = mapped_column("admin_dong_name", String(100), nullable=False)
    total_population: Mapped[int] = mapped_column("total_population", Integer, nullable=False)
    representative_type: Mapped[str] = mapped_column("representative_type", String(100), nullable=False)
    profile_data: Mapped[dict[str, Any]] = mapped_column("profile_data", JSON, nullable=False)


class UserRecommendationLog(Base):
    __tablename__ = "user_recommendation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    event_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    admin_dong_code: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    impression: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    clicked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    liked: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dwell_time_sec: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    q1: Mapped[int] = mapped_column(Integer, nullable=False)
    q2: Mapped[int] = mapped_column(Integer, nullable=False)
    q3: Mapped[int] = mapped_column(Integer, nullable=False)
    q4: Mapped[int] = mapped_column(Integer, nullable=False)
    q5: Mapped[int] = mapped_column(Integer, nullable=False)
    q6: Mapped[int] = mapped_column(Integer, nullable=False)
    q7: Mapped[int] = mapped_column(Integer, nullable=False)
    q8: Mapped[int] = mapped_column(Integer, nullable=False)
    q9: Mapped[int] = mapped_column(Integer, nullable=False)
    q10: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
