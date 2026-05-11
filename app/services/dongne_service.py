from __future__ import annotations
from datetime import date
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np
import pandas as pd
from sqlalchemy import inspect
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.core.config import settings
from app.db.session import get_engine
from app.schemas.dongne import DongneInteractionBatchRequest
from app.schemas.dongne import DongneInteractionBatchResponse
from app.schemas.dongne import DongRecommendationResponse
from app.schemas.dongne import DongneRecommendationRequest
from app.utils.dongne_paths import DONGNE_INTEREST_CSV
from app.utils.dongne_paths import DONGNE_S3_DATA_DIR
from app.utils.dongne_paths import DONGNE_TELECOM_CSV
from app.utils.s3_csv import find_csv_path
from app.utils.s3_csv import read_csv_dataframe
from scripts.recommendation import recommendation_ml_utils as ml_utils
from scripts.recommendation import resident_recommender as rr


QUESTION_TEXT = {
    "q1": "혼자 사는 사람 비중이 높은 동네가 더 편한가요?",
    "q2": "집에서 보내는 시간이 많은 생활 패턴인가요?",
    "q3": "동네 안에서 사람들과 자연스럽게 연결되는 분위기를 원하나요?",
    "q4": "출퇴근이나 평일 이동이 적은 생활권을 선호하나요?",
    "q5": "주말에도 멀리 나가기보다 집 근처 생활을 선호하나요?",
    "q6": "배달, 쇼핑, 생활서비스를 자주 이용하나요?",
    "q7": "활발한 상권형 동네보다 안정적인 주거형 동네를 원하나요?",
    "q8": "생활비나 고정비 부담이 상대적으로 덜한 동네가 중요하나요?",
    "q9": "청년 1인가구가 많은 동네 분위기가 더 잘 맞나요?",
    "q10": "혼자 지내기 편한 동네와 사람들과 섞이기 쉬운 동네 중 어느 쪽이 더 중요한가요?",
}

QUESTION_ORDER = list(QUESTION_TEXT.keys())
DEFAULT_DATA_DIR = DONGNE_S3_DATA_DIR
DONG_PROFILE_TABLE = "dong_region_profiles"
RECOMMENDATION_LOG_TABLE = "user_recommendation_logs"
INTEGRATED_ADMIN_DONG_TABLE = rr.SOURCE_TABLE


@dataclass(frozen=True)
class RecommenderConfig:
    top_dong: int = 427
    min_population: int = 1000


def _get_engine() -> Engine:
    return get_engine(settings.DATABASE_URL)


def _table_has_rows(engine: Engine, table_name: str) -> bool:
    inspector = inspect(engine)
    if not inspector.has_table(table_name):
        return False

    with engine.begin() as connection:
        row_count = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()
    return row_count > 0


def _normalize_loaded_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    integer_columns = {"행정동코드", "총인구"}
    for column in normalized.columns:
        if column in integer_columns:
            normalized[column] = _coerce_integer_series(normalized[column])
        elif normalized[column].dtype == object:
            numeric_series = pd.to_numeric(normalized[column], errors="coerce")
            if numeric_series.notna().all():
                normalized[column] = numeric_series
    return normalized


def _store_region_profiles(engine: Engine, profiles: dict[str, pd.DataFrame]) -> None:
    profiles["dong"].to_sql(DONG_PROFILE_TABLE, con=engine, if_exists="replace", index=False)


def _ensure_recommendation_log_table(engine: Engine) -> None:
    if inspect(engine).has_table(RECOMMENDATION_LOG_TABLE):
        return

    with engine.begin() as connection:
        connection.execute(
            text(
                f"""
                CREATE TABLE {RECOMMENDATION_LOG_TABLE} (
                    user_id BIGINT NOT NULL,
                    created_at DATE NOT NULL,
                    admin_dong_code BIGINT NOT NULL,
                    rank_position INTEGER NOT NULL,
                    impression_count INTEGER NOT NULL,
                    clicked_count INTEGER NOT NULL,
                    liked_count INTEGER NOT NULL,
                    dwell_time_sec FLOAT NOT NULL,
                    q1 INTEGER NOT NULL,
                    q2 INTEGER NOT NULL,
                    q3 INTEGER NOT NULL,
                    q4 INTEGER NOT NULL,
                    q5 INTEGER NOT NULL,
                    q6 INTEGER NOT NULL,
                    q7 INTEGER NOT NULL,
                    q8 INTEGER NOT NULL,
                    q9 INTEGER NOT NULL,
                    q10 INTEGER NOT NULL
                )
                """
            )
        )


def _load_profiles_from_database(engine: Engine) -> dict[str, pd.DataFrame]:
    dong = pd.read_sql_table(DONG_PROFILE_TABLE, con=engine)
    return {"dong": _normalize_loaded_profiles(dong)}


def save_interactions(payload: DongneInteractionBatchRequest) -> DongneInteractionBatchResponse:
    engine = _get_engine()
    _ensure_recommendation_log_table(engine)
    updated_count = 0

    with engine.begin() as connection:
        for item in payload.interactions:
            created_at = (item.created_at or date.today()).isoformat()
            row = connection.execute(
                text(
                    f"""
                    SELECT 1
                    FROM {RECOMMENDATION_LOG_TABLE}
                    WHERE user_id = :user_id
                      AND created_at = :created_at
                      AND admin_dong_code = :admin_dong_code
                    LIMIT 1
                    """
                ),
                {
                    "user_id": item.user_id,
                    "created_at": created_at,
                    "admin_dong_code": item.admin_dong_code,
                },
            ).fetchone()

            if row is None:
                connection.execute(
                    text(
                        f"""
                        INSERT INTO {RECOMMENDATION_LOG_TABLE} (
                            user_id,
                            created_at,
                            admin_dong_code,
                            rank_position,
                            impression_count,
                            clicked_count,
                            liked_count,
                            dwell_time_sec,
                            q1, q2, q3, q4, q5, q6, q7, q8, q9, q10
                        ) VALUES (
                            :user_id,
                            :created_at,
                            :admin_dong_code,
                            :rank_position,
                            :impression_count,
                            :clicked_count,
                            :liked_count,
                            :dwell_time_sec,
                            :q1, :q2, :q3, :q4, :q5, :q6, :q7, :q8, :q9, :q10
                        )
                        """
                    ),
                    {
                        "user_id": item.user_id,
                        "created_at": created_at,
                        "admin_dong_code": item.admin_dong_code,
                        "rank_position": item.rank_position or 1,
                        "impression_count": item.impression_count,
                        "clicked_count": item.clicked_count,
                        "liked_count": item.liked_count,
                        "dwell_time_sec": item.dwell_time_sec,
                        "q1": item.q1,
                        "q2": item.q2,
                        "q3": item.q3,
                        "q4": item.q4,
                        "q5": item.q5,
                        "q6": item.q6,
                        "q7": item.q7,
                        "q8": item.q8,
                        "q9": item.q9,
                        "q10": item.q10,
                    },
                )
            else:
                connection.execute(
                    text(
                        f"""
                        UPDATE {RECOMMENDATION_LOG_TABLE}
                        SET clicked_count = :clicked_count,
                            liked_count = :liked_count,
                            dwell_time_sec = :dwell_time_sec,
                            impression_count = :impression_count,
                            rank_position = COALESCE(:rank_position, rank_position),
                            q1 = :q1,
                            q2 = :q2,
                            q3 = :q3,
                            q4 = :q4,
                            q5 = :q5,
                            q6 = :q6,
                            q7 = :q7,
                            q8 = :q8,
                            q9 = :q9,
                            q10 = :q10
                        WHERE user_id = :user_id
                          AND created_at = :created_at
                          AND admin_dong_code = :admin_dong_code
                        """
                    ),
                    {
                        "clicked_count": item.clicked_count,
                        "liked_count": item.liked_count,
                        "dwell_time_sec": item.dwell_time_sec,
                        "impression_count": item.impression_count,
                        "rank_position": item.rank_position,
                        "q1": item.q1,
                        "q2": item.q2,
                        "q3": item.q3,
                        "q4": item.q4,
                        "q5": item.q5,
                        "q6": item.q6,
                        "q7": item.q7,
                        "q8": item.q8,
                        "q9": item.q9,
                        "q10": item.q10,
                        "user_id": item.user_id,
                        "created_at": created_at,
                        "admin_dong_code": item.admin_dong_code,
                    },
                )
            updated_count += 1

    return DongneInteractionBatchResponse(updated_count=updated_count)


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "nan": np.nan}),
        errors="coerce",
    )


def _coerce_integer_series(series: pd.Series) -> pd.Series:
    numeric = _to_numeric_series(series)
    if numeric.dropna().empty:
        return numeric.astype("Int64")

    # CSV/DB round-trips sometimes turn integer-like columns into float64
    # values such as 1111051500.0. Treat semantically-integer columns as
    # integers by rounding before converting to pandas' nullable Int64 type.
    rounded = numeric.round()
    return rounded.astype("Int64")


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _normalize_response(value: float) -> float:
    if value < 1 or value > 5:
        raise ValueError(f"응답값은 1~5 범위여야 합니다. 받은 값: {value}")
    return (float(value) - 3.0) / 2.0


def validate_responses(responses: Mapping[str, float]) -> dict[str, float]:
    missing = [key for key in QUESTION_ORDER if key not in responses]
    if missing:
        raise ValueError(f"응답이 누락된 질문이 있습니다: {missing}")
    return {key: _normalize_response(float(responses[key])) for key in QUESTION_ORDER}


def _build_type_answers(responses: Mapping[str, float]) -> dict[str, float]:
    return {
        ml_utils.QKEY_TO_QUESTION_ID[qkey]: float(responses[qkey])
        for qkey in QUESTION_ORDER
    }


def build_user_vector(responses: Mapping[str, float]) -> dict[str, float | dict[str, float]]:
    normalized = validate_responses(responses)
    q1 = normalized["q1"]
    q2 = normalized["q2"]
    q3 = normalized["q3"]
    q4 = normalized["q4"]
    q5 = normalized["q5"]
    q6 = normalized["q6"]
    q7 = normalized["q7"]
    q8 = normalized["q8"]
    q9 = normalized["q9"]
    q10 = normalized["q10"]

    return {
        "responses_normalized": normalized,
        "single_household_affinity": 0.6 * q1 + 0.4 * q9,
        "settled_home_life": 0.5 * q2 + 0.3 * q5 + 0.2 * q7,
        "social_connection_preference": 1.0 * q3 - 0.7 * q10,
        "low_mobility_preference": 0.7 * q4 + 0.3 * q5,
        "lifestyle_service_dependence": 1.0 * q6,
        "residential_stability_preference": 0.8 * q7 + 0.2 * q2,
        "cost_sensitivity": 1.0 * q8,
        "youth_mobile_preference": 0.5 * q1 + 0.5 * q9 - 0.4 * q7,
    }


def _build_region_profiles_from_frames(interest: pd.DataFrame, telecom: pd.DataFrame) -> dict[str, pd.DataFrame]:
    interest = interest.loc[:, [col for col in interest.columns if str(col).strip() and not str(col).startswith("Unnamed:")]].copy()
    telecom = telecom.loc[:, [col for col in telecom.columns if str(col).strip() and not str(col).startswith("Unnamed:")]].copy()
    interest = interest.rename(
        columns={
            "admin_dong_code": "행정동코드",
            "district_name": "자치구",
            "admin_dong_name": "행정동명",
            "total_population": "총인구",
        }
    )
    telecom = telecom.rename(columns={"행정동": "행정동명", "총인구수": "총인구_통신정보", "1인가구수": "1인가구수_통신정보"})
    telecom = telecom.rename(
        columns={
            "admin_dong_code": "행정동코드",
            "district_name": "자치구",
            "admin_dong_name": "행정동명",
        }
    )

    interest_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]
    telecom_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]

    for col in interest.columns:
        if col not in interest_id_cols:
            interest[col] = _to_numeric_series(interest[col])

    for col in telecom.columns:
        if col not in telecom_id_cols:
            telecom[col] = _to_numeric_series(telecom[col])

    aggregated_input = not {"성별", "연령대"}.issubset(interest.columns) and not {"성별", "연령대"}.issubset(telecom.columns)
    if aggregated_input:
        merge_keys = [key for key in ["행정동코드", "자치구", "행정동명"] if key in interest.columns and key in telecom.columns]
        telecom_drop_cols = [col for col in [] if col in telecom.columns]
        merged = interest.merge(
            telecom.drop(columns=telecom_drop_cols),
            on=merge_keys,
            how="left",
        )
    else:
        merged = interest.merge(
            telecom.drop(columns=["자치구"]),
            on=["행정동코드", "행정동명", "성별", "연령대"],
            how="left",
        )

    interest_count_cols = [
        "1인가구수",
        "커뮤니케이션이 적은 집단",
        "평일 외출이 적은 집단",
        "휴일 외출이 적은 집단",
        "출근소요시간 및 근무시간이 많은 집단",
        "외출이 매우 적은 집단(전체)",
        "외출이 매우 많은 집단",
        "동영상서비스 이용이 많은 집단",
        "생활서비스 이용이 많은 집단",
        "재정상태에 대한 관심집단",
        "외출-커뮤니케이션이 모두 적은 집단(전체)",
    ]

    telecom_feature_cols = [
        "최근 3개월 내 요금 연체 비율",
        "평일 총 이동 횟수",
        "휴일 총 이동 횟수 평균",
        "집 추정 위치 휴일 총 체류시간",
        "평균 통화대상자 수",
        "평균 문자대상자 수",
        "데이터 사용량",
        "동영상/방송 서비스 사용일수",
        "금융 서비스 사용일수",
        "쇼핑 서비스 사용일수",
        "배달 서비스 사용일수",
    ]
    telecom_feature_cols = [col for col in telecom_feature_cols if col in merged.columns]

    if aggregated_input:
        dong = merged.copy()
        for col in interest_count_cols:
            if col in dong.columns:
                dong[f"{col}비율"] = dong[col] / dong["총인구"] if "총인구" in dong.columns else np.nan
    else:
        dong_rows: list[dict[str, Any]] = []
        for keys, group in merged.groupby(["행정동코드", "자치구", "행정동명"], dropna=False):
            row: dict[str, Any] = {"행정동코드": keys[0], "자치구": keys[1], "행정동명": keys[2], "총인구": group["총인구"].sum()}

            for col in interest_count_cols:
                row[col] = group[col].sum()
                row[f"{col}비율"] = row[col] / row["총인구"] if row["총인구"] else np.nan

            weights = group["총인구"].fillna(0)
            for col in telecom_feature_cols:
                valid_mask = group[col].notna() & weights.notna()
                if valid_mask.any() and weights[valid_mask].sum() > 0:
                    row[col] = float(np.average(group.loc[valid_mask, col], weights=weights[valid_mask]))
                else:
                    row[col] = np.nan

            dong_rows.append(row)

        dong = pd.DataFrame(dong_rows)

    core_cols = [
        "1인가구수비율",
        "커뮤니케이션이 적은 집단비율",
        "외출이 매우 적은 집단(전체)비율",
        "재정상태에 대한 관심집단비율",
        "외출-커뮤니케이션이 모두 적은 집단(전체)비율",
        "최근 3개월 내 요금 연체 비율",
        "평일 총 이동 횟수",
        "휴일 총 이동 횟수 평균",
        "집 추정 위치 휴일 총 체류시간",
        "평균 통화대상자 수",
        "평균 문자대상자 수",
        "동영상/방송 서비스 사용일수",
        "쇼핑 서비스 사용일수",
        "배달 서비스 사용일수",
        "금융 서비스 사용일수",
    ]

    for col in core_cols:
        if col in dong.columns:
            dong[f"{col}_z"] = _zscore(dong[col])

    dong["single_household_profile"] = 0.7 * dong["1인가구수비율_z"] + 0.3 * dong["재정상태에 대한 관심집단비율_z"]
    dong["settled_profile"] = (
        0.35 * dong["외출이 매우 적은 집단(전체)비율_z"]
        + 0.35 * dong["집 추정 위치 휴일 총 체류시간_z"]
        - 0.15 * dong["평일 총 이동 횟수_z"]
        - 0.15 * dong["휴일 총 이동 횟수 평균_z"]
    )
    dong["social_profile"] = (
        0.4 * dong["평균 통화대상자 수_z"]
        + 0.4 * dong["평균 문자대상자 수_z"]
        - 0.2 * dong["커뮤니케이션이 적은 집단비율_z"]
    )
    dong["service_profile"] = (
        0.4 * dong["배달 서비스 사용일수_z"]
        + 0.3 * dong["쇼핑 서비스 사용일수_z"]
        + 0.3 * dong["동영상/방송 서비스 사용일수_z"]
    )
    dong["cost_risk_profile"] = 0.5 * dong["재정상태에 대한 관심집단비율_z"] + 0.5 * dong["최근 3개월 내 요금 연체 비율_z"]
    dong["isolation_risk_profile"] = (
        0.4 * dong["커뮤니케이션이 적은 집단비율_z"]
        + 0.4 * dong["외출-커뮤니케이션이 모두 적은 집단(전체)비율_z"]
        + 0.2 * dong["외출이 매우 적은 집단(전체)비율_z"]
    )
    dong["youth_mobile_profile"] = (
        0.4 * dong["1인가구수비율_z"] + 0.3 * dong["평일 총 이동 횟수_z"] + 0.3 * dong["쇼핑 서비스 사용일수_z"]
    )
    dong["residential_stability_profile"] = (
        0.5 * dong["settled_profile"] - 0.3 * dong["youth_mobile_profile"] - 0.2 * dong["service_profile"]
    )

    dong["고립위험형점수"] = (
        dong["커뮤니케이션이 적은 집단비율_z"]
        + dong["외출-커뮤니케이션이 모두 적은 집단(전체)비율_z"]
        - dong["평균 통화대상자 수_z"]
        - dong["평균 문자대상자 수_z"]
    ) / 4
    dong["정주고착형점수"] = (
        dong["외출이 매우 적은 집단(전체)비율_z"]
        + dong["집 추정 위치 휴일 총 체류시간_z"]
        - dong["평일 총 이동 횟수_z"]
        - dong["휴일 총 이동 횟수 평균_z"]
    ) / 4
    dong["재정부담형점수"] = (dong["재정상태에 대한 관심집단비율_z"] + dong["최근 3개월 내 요금 연체 비율_z"]) / 2
    dong["1인가구집중형점수"] = dong["1인가구수비율_z"]

    type_cols = ["고립위험형점수", "정주고착형점수", "재정부담형점수", "1인가구집중형점수"]
    type_name_map = {
        "고립위험형점수": "고립위험형",
        "정주고착형점수": "정주고착형",
        "재정부담형점수": "재정부담형",
        "1인가구집중형점수": "1인가구집중형",
    }
    dong["대표유형"] = dong[type_cols].idxmax(axis=1).map(type_name_map)

    return {"dong": dong}


def _build_region_profiles_from_csv(base_dir: str) -> dict[str, pd.DataFrame]:
    if base_dir == DONGNE_S3_DATA_DIR:
        interest_path = DONGNE_INTEREST_CSV
        telecom_path = DONGNE_TELECOM_CSV
    else:
        interest_path = find_csv_path(base_dir, "관심집단수")
        telecom_path = find_csv_path(base_dir, "통신정보")

    interest = read_csv_dataframe(interest_path, encoding="utf-8-sig")
    telecom = read_csv_dataframe(telecom_path, encoding="utf-8-sig")
    return _build_region_profiles_from_frames(interest, telecom)


def _source_frame_from_integrated_data(source: pd.DataFrame, prefix: str) -> pd.DataFrame:
    id_columns = {
        "admin_dong_code": "행정동코드",
        "district_name": "자치구",
        "admin_dong_name": "행정동명",
    }
    frame = source.loc[:, list(id_columns)].rename(columns=id_columns).copy()
    prefixed_columns = [column for column in source.columns if column.startswith(prefix)]
    for column in prefixed_columns:
        frame[column.removeprefix(prefix)] = source[column]
    return frame


def _build_region_profiles_from_integrated_database(engine: Engine) -> dict[str, pd.DataFrame] | None:
    if not _table_has_rows(engine, INTEGRATED_ADMIN_DONG_TABLE):
        return None

    source = pd.read_sql_table(INTEGRATED_ADMIN_DONG_TABLE, con=engine)
    interest = _source_frame_from_integrated_data(source, "interest__")
    telecom = _source_frame_from_integrated_data(source, "telecom__")
    return _build_region_profiles_from_frames(interest, telecom)


@lru_cache(maxsize=8)
def load_region_profiles(base_dir: str, database_url: str) -> dict[str, pd.DataFrame]:
    engine = get_engine(database_url)
    if not _table_has_rows(engine, DONG_PROFILE_TABLE):
        profiles = _build_region_profiles_from_integrated_database(engine)
        if profiles is not None:
            _store_region_profiles(engine, profiles)
            return profiles
        profiles = _build_region_profiles_from_csv(base_dir)
        _store_region_profiles(engine, profiles)
        return profiles
    return _load_profiles_from_database(engine)


def _match_score(user_vector: Mapping[str, float | dict[str, float]], row: pd.Series) -> float:
    return (
        0.18 * (float(user_vector["single_household_affinity"]) * row["single_household_profile"])
        + 0.16 * (float(user_vector["settled_home_life"]) * row["settled_profile"])
        + 0.14 * (float(user_vector["social_connection_preference"]) * row["social_profile"])
        + 0.14 * (float(user_vector["low_mobility_preference"]) * row["settled_profile"])
        + 0.12 * (float(user_vector["lifestyle_service_dependence"]) * row["service_profile"])
        + 0.10 * (float(user_vector["residential_stability_preference"]) * row["residential_stability_profile"])
        + 0.08 * (float(user_vector["youth_mobile_preference"]) * row["youth_mobile_profile"])
        - 0.08 * (float(user_vector["cost_sensitivity"]) * row["cost_risk_profile"])
        - 0.06 * (float(user_vector["social_connection_preference"]) * row["isolation_risk_profile"])
    )


def _pick_traits(row: pd.Series) -> list[str]:
    candidates = [
        ("single_household_profile", "1인가구 친화성이 높고"),
        ("settled_profile", "집 중심·정주형 생활과 잘 맞고"),
        ("social_profile", "사람들과의 연결감이 비교적 살아 있고"),
        ("service_profile", "배달·쇼핑·생활서비스 활용성이 높고"),
        ("youth_mobile_profile", "청년·유동형 생활 패턴과 잘 맞고"),
        ("residential_stability_profile", "주거 안정감이 비교적 강하고"),
    ]
    picked = [text for key, text in sorted(candidates, key=lambda item: row[item[0]], reverse=True)[:3] if pd.notna(row[key])]
    return picked or ["생활 패턴 특성이 비교적 뚜렷하고"]


def _pick_cautions(row: pd.Series) -> str:
    caution_parts: list[str] = []
    if row.get("cost_risk_profile", 0) > 0.8:
        caution_parts.append("비용 부담 신호가 상대적으로 높은 편입니다")
    if row.get("isolation_risk_profile", 0) > 0.8:
        caution_parts.append("활발한 교류를 기대하면 다소 조용하게 느껴질 수 있습니다")
    if row.get("service_profile", 0) < -0.5:
        caution_parts.append("생활서비스 밀도는 기대보다 낮을 수 있습니다")
    return " ".join(caution_parts) if caution_parts else "전체적으로는 사용자 응답과 큰 충돌 없이 맞는 편입니다"


def _user_summary(user_vector: Mapping[str, float | dict[str, float]]) -> str:
    pairs = [
        ("single_household_affinity", "1인가구 친화성이 높은 편이고"),
        ("settled_home_life", "집 중심·정주형 성향이 있고"),
        ("social_connection_preference", "사람들과의 연결을 중시하는 편이고"),
        ("low_mobility_preference", "이동을 줄인 생활권을 선호하는 편이고"),
        ("lifestyle_service_dependence", "생활서비스 활용도가 높은 편이고"),
        ("residential_stability_preference", "주거 안정성을 중시하는 편이고"),
        ("cost_sensitivity", "비용 민감도가 높은 편이고"),
        ("youth_mobile_preference", "청년·유동형 분위기를 선호하는 편이고"),
    ]
    top = sorted(pairs, key=lambda item: float(user_vector[item[0]]), reverse=True)[:3]
    summary = " ".join([text for _, text in top]).strip()
    if summary.endswith("이고"):
        summary = summary[:-2] + "인"
    return summary


def recommend_dongs(
    payload: DongneRecommendationRequest,
    *,
    base_dir: str | Path = DEFAULT_DATA_DIR,
    config: RecommenderConfig | None = None,
) -> DongRecommendationResponse:
    recommender_config = config or RecommenderConfig()
    responses = payload.model_dump()
    user_vector = build_user_vector(responses)
    type_result = rr.classify_user_type(_build_type_answers(responses))
    profiles = load_region_profiles(str(Path(base_dir).resolve()), settings.DATABASE_URL)

    dong = profiles["dong"].copy()
    dong = dong[dong["총인구"] >= recommender_config.min_population].copy()

    dong["match_score"] = dong.apply(lambda row: _match_score(user_vector, row), axis=1)

    top_dongs = dong.sort_values("match_score", ascending=False).head(recommender_config.top_dong)

    recommendation_pks: list[int] = []
    for _, dong_row in top_dongs.iterrows():
        recommendation_pks.append(int(dong_row["행정동코드"]))

    return DongRecommendationResponse(
        user_type=str(type_result["type_label"]),
        recommendations=recommendation_pks,
    )
