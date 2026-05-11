# 추천 로직 수정
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_S3_DATA_DIR
from app.utils.s3_csv import find_csv_path
from app.utils.s3_csv import read_csv_dataframe


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


@dataclass(frozen=True)
class RecommenderConfig:
    top_gu: int = 3
    top_dong_per_gu: int = 3
    min_population: int = 1000


def _to_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "nan": np.nan}),
        errors="coerce",
    )


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _normalize_response(value: float) -> float:
    if value < 1 or value > 5:
        raise ValueError(f"응답값은 1~5 범위여야 합니다. 받은 값: {value}")
    return (float(value) - 3.0) / 2.0


def validate_responses(responses: Mapping[str, float]) -> Dict[str, float]:
    missing = [key for key in QUESTION_ORDER if key not in responses]
    if missing:
        raise ValueError(f"응답이 누락된 질문이 있습니다: {missing}")
    return {key: _normalize_response(float(responses[key])) for key in QUESTION_ORDER}


def build_user_vector(responses: Mapping[str, float]) -> Dict[str, float]:
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


@lru_cache(maxsize=2)
def load_region_profiles(base_dir: str) -> Dict[str, pd.DataFrame]:
    interest_path = find_csv_path(base_dir, "관심집단수")
    telecom_path = find_csv_path(base_dir, "통신정보")

    interest = read_csv_dataframe(interest_path, encoding="utf-8-sig")
    telecom = read_csv_dataframe(telecom_path, encoding="utf-8-sig")

    interest = interest.loc[:, [c for c in interest.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()
    telecom = telecom.loc[:, [c for c in telecom.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()
    telecom = telecom.rename(columns={"행정동": "행정동명", "총인구수": "총인구_통신정보", "1인가구수": "1인가구수_통신정보"})

    interest_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]
    telecom_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]

    for col in interest.columns:
        if col not in interest_id_cols:
            interest[col] = _to_numeric_series(interest[col])

    for col in telecom.columns:
        if col not in telecom_id_cols:
            telecom[col] = _to_numeric_series(telecom[col])

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
    telecom_feature_cols = [c for c in telecom_feature_cols if c in merged.columns]

    dong_rows: List[Dict[str, Any]] = []
    for keys, group in merged.groupby(["행정동코드", "자치구", "행정동명"], dropna=False):
        row: Dict[str, Any] = {"행정동코드": keys[0], "자치구": keys[1], "행정동명": keys[2], "총인구": group["총인구"].sum()}

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
    dong["cost_risk_profile"] = (
        0.5 * dong["재정상태에 대한 관심집단비율_z"] + 0.5 * dong["최근 3개월 내 요금 연체 비율_z"]
    )
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
    dong["재정부담형점수"] = (
        dong["재정상태에 대한 관심집단비율_z"] + dong["최근 3개월 내 요금 연체 비율_z"]
    ) / 2
    dong["1인가구집중형점수"] = dong["1인가구수비율_z"]

    type_cols = ["고립위험형점수", "정주고착형점수", "재정부담형점수", "1인가구집중형점수"]
    type_name_map = {
        "고립위험형점수": "고립위험형",
        "정주고착형점수": "정주고착형",
        "재정부담형점수": "재정부담형",
        "1인가구집중형점수": "1인가구집중형",
    }
    dong["대표유형"] = dong[type_cols].idxmax(axis=1).map(type_name_map)

    gu_sum_cols = ["총인구"] + interest_count_cols
    gu = dong.groupby("자치구", as_index=False)[gu_sum_cols].sum()
    for col in interest_count_cols:
        gu[f"{col}비율"] = gu[col] / gu["총인구"]

    for col in telecom_feature_cols:
        weighted = (
            dong.assign(_weighted=dong[col] * dong["총인구"])
            .groupby("자치구", as_index=False)[["_weighted", "총인구"]]
            .sum()
        )
        gu = gu.merge(weighted.rename(columns={"_weighted": f"{col}_weighted", "총인구": "총인구_tmp"}), on="자치구", how="left")
        gu[col] = gu[f"{col}_weighted"] / gu["총인구_tmp"]
        gu = gu.drop(columns=[f"{col}_weighted", "총인구_tmp"])

    for col in core_cols:
        if col in gu.columns:
            gu[f"{col}_z"] = _zscore(gu[col])

    gu["single_household_profile"] = 0.7 * gu["1인가구수비율_z"] + 0.3 * gu["재정상태에 대한 관심집단비율_z"]
    gu["settled_profile"] = (
        0.35 * gu["외출이 매우 적은 집단(전체)비율_z"]
        + 0.35 * gu["집 추정 위치 휴일 총 체류시간_z"]
        - 0.15 * gu["평일 총 이동 횟수_z"]
        - 0.15 * gu["휴일 총 이동 횟수 평균_z"]
    )
    gu["social_profile"] = (
        0.4 * gu["평균 통화대상자 수_z"]
        + 0.4 * gu["평균 문자대상자 수_z"]
        - 0.2 * gu["커뮤니케이션이 적은 집단비율_z"]
    )
    gu["service_profile"] = (
        0.4 * gu["배달 서비스 사용일수_z"]
        + 0.3 * gu["쇼핑 서비스 사용일수_z"]
        + 0.3 * gu["동영상/방송 서비스 사용일수_z"]
    )
    gu["cost_risk_profile"] = (
        0.5 * gu["재정상태에 대한 관심집단비율_z"] + 0.5 * gu["최근 3개월 내 요금 연체 비율_z"]
    )
    gu["isolation_risk_profile"] = (
        0.4 * gu["커뮤니케이션이 적은 집단비율_z"]
        + 0.4 * gu["외출-커뮤니케이션이 모두 적은 집단(전체)비율_z"]
        + 0.2 * gu["외출이 매우 적은 집단(전체)비율_z"]
    )
    gu["youth_mobile_profile"] = (
        0.4 * gu["1인가구수비율_z"] + 0.3 * gu["평일 총 이동 횟수_z"] + 0.3 * gu["쇼핑 서비스 사용일수_z"]
    )
    gu["residential_stability_profile"] = (
        0.5 * gu["settled_profile"] - 0.3 * gu["youth_mobile_profile"] - 0.2 * gu["service_profile"]
    )

    return {"dong": dong, "gu": gu}


def _match_score(user_vector: Mapping[str, float], row: pd.Series) -> float:
    return (
        0.18 * (user_vector["single_household_affinity"] * row["single_household_profile"])
        + 0.16 * (user_vector["settled_home_life"] * row["settled_profile"])
        + 0.14 * (user_vector["social_connection_preference"] * row["social_profile"])
        + 0.14 * (user_vector["low_mobility_preference"] * row["settled_profile"])
        + 0.12 * (user_vector["lifestyle_service_dependence"] * row["service_profile"])
        + 0.10 * (user_vector["residential_stability_preference"] * row["residential_stability_profile"])
        + 0.08 * (user_vector["youth_mobile_preference"] * row["youth_mobile_profile"])
        - 0.08 * (user_vector["cost_sensitivity"] * row["cost_risk_profile"])
        - 0.06 * (user_vector["social_connection_preference"] * row["isolation_risk_profile"])
    )


def _pick_traits(row: pd.Series) -> List[str]:
    candidates = [
        ("single_household_profile", "1인가구 친화성이 높고"),
        ("settled_profile", "집 중심·정주형 생활과 잘 맞고"),
        ("social_profile", "사람들과의 연결감이 비교적 살아 있고"),
        ("service_profile", "배달·쇼핑·생활서비스 활용성이 높고"),
        ("youth_mobile_profile", "청년·유동형 생활 패턴과 잘 맞고"),
        ("residential_stability_profile", "주거 안정감이 비교적 강하고"),
    ]
    picked = [text for key, text in sorted(candidates, key=lambda x: row[x[0]], reverse=True)[:3] if pd.notna(row[key])]
    return picked or ["생활 패턴 특성이 비교적 뚜렷하고"]


def _pick_cautions(row: pd.Series) -> str:
    caution_parts: List[str] = []
    if row.get("cost_risk_profile", 0) > 0.8:
        caution_parts.append("비용 부담 신호가 상대적으로 높은 편입니다")
    if row.get("isolation_risk_profile", 0) > 0.8:
        caution_parts.append("활발한 교류를 기대하면 다소 조용하게 느껴질 수 있습니다")
    if row.get("service_profile", 0) < -0.5:
        caution_parts.append("생활서비스 밀도는 기대보다 낮을 수 있습니다")
    return " ".join(caution_parts) if caution_parts else "전체적으로는 사용자 응답과 큰 충돌 없이 맞는 편입니다"


def _user_summary(user_vector: Mapping[str, float]) -> str:
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
    top = sorted(pairs, key=lambda x: user_vector[x[0]], reverse=True)[:3]
    summary = " ".join([text for _, text in top]).strip()
    if summary.endswith("이고"):
        summary = summary[:-2] + "인"
    return summary


def build_gu_description(user_vector: Mapping[str, float], row: pd.Series) -> str:
    traits = " ".join(_pick_traits(row))
    caution = _pick_cautions(row)
    user_pref = _user_summary(user_vector)
    return (
        f"{row['자치구']}는 서울 안에서도 {traits.rstrip('고')} 성격이 상대적으로 두드러지는 지역입니다. "
        f"당신은 {user_pref} 성향이 있어 이 구의 생활 패턴과 잘 맞을 가능성이 높습니다. "
        f"특히 서비스 이용, 이동성, 주거 안정감의 조합이 사용자 응답과 잘 맞는 편입니다. "
        f"다만 {caution}"
    )


def build_dong_description(user_vector: Mapping[str, float], row: pd.Series) -> str:
    traits = " ".join(_pick_traits(row))
    caution = _pick_cautions(row)
    user_pref = _user_summary(user_vector)
    return (
        f"{row['자치구']} {row['행정동명']}은 {row['대표유형']} 성격이 강한 동네입니다. "
        f"이곳은 {traits.rstrip('고')} 특징이 함께 나타나 {user_pref} 사용자에게 특히 잘 맞을 수 있습니다. "
        f"응답 기준으로는 이동성, 생활서비스, 사회적 연결감 축에서 높은 적합도를 보였습니다. "
        f"다만 {caution}"
    )


def recommend_seoul_neighborhoods(
    responses: Mapping[str, float],
    base_dir: str | Path = DONGNE_S3_DATA_DIR,
    config: RecommenderConfig | None = None,
) -> Dict[str, Any]:
    config = config or RecommenderConfig()
    user_vector = build_user_vector(responses)
    profiles = load_region_profiles(str(Path(base_dir).resolve()))

    gu = profiles["gu"].copy()
    dong = profiles["dong"].copy()
    dong = dong[dong["총인구"] >= config.min_population].copy()

    gu["match_score"] = gu.apply(lambda row: _match_score(user_vector, row), axis=1)
    dong["match_score"] = dong.apply(lambda row: _match_score(user_vector, row), axis=1)

    gu = gu.sort_values("match_score", ascending=False).reset_index(drop=True)
    top_gus = gu.head(config.top_gu).copy()

    gu_results: List[Dict[str, Any]] = []
    dong_results: List[Dict[str, Any]] = []

    for _, gu_row in top_gus.iterrows():
        gu_result = {
            "gu": gu_row["자치구"],
            "match_score": float(gu_row["match_score"]),
            "description": build_gu_description(user_vector, gu_row),
            "top_traits": _pick_traits(gu_row),
            "caution": _pick_cautions(gu_row),
        }
        gu_results.append(gu_result)

        gu_dongs = (
            dong[dong["자치구"] == gu_row["자치구"]]
            .sort_values("match_score", ascending=False)
            .head(config.top_dong_per_gu)
            .copy()
        )

        for _, dong_row in gu_dongs.iterrows():
            dong_results.append(
                {
                    "gu": dong_row["자치구"],
                    "dong": dong_row["행정동명"],
                    "match_score": float(dong_row["match_score"]),
                    "population": int(dong_row["총인구"]),
                    "type": dong_row["대표유형"],
                    "description": build_dong_description(user_vector, dong_row),
                    "top_traits": _pick_traits(dong_row),
                    "caution": _pick_cautions(dong_row),
                }
            )

    top_gu_names = ", ".join([item["gu"] for item in gu_results])
    top_dong_names = ", ".join([f"{item['gu']} {item['dong']}" for item in dong_results[:5]])
    overall_summary = (
        f"당신은 {_user_summary(user_vector)} 성향이 두드러집니다. "
        f"그래서 서울 안에서는 {top_gu_names} 쪽이 먼저 추천되며, "
        f"세부적으로는 {top_dong_names} 같은 동네가 잘 맞을 가능성이 높습니다."
    )

    return {
        "question_text": QUESTION_TEXT,
        "normalized_responses": user_vector["responses_normalized"],
        "user_vector": {k: v for k, v in user_vector.items() if k != "responses_normalized"},
        "gu_recommendations": gu_results,
        "dong_recommendations": dong_results,
        "summary": overall_summary,
    }


if __name__ == "__main__":
    sample_responses = {
        "q1": 5,
        "q2": 4,
        "q3": 2,
        "q4": 4,
        "q5": 4,
        "q6": 5,
        "q7": 3,
        "q8": 4,
        "q9": 5,
        "q10": 4,
    }
    result = recommend_seoul_neighborhoods(sample_responses, base_dir=DONGNE_S3_DATA_DIR)
    print(result["summary"])
    print("\n[구 추천]")
    for item in result["gu_recommendations"]:
        print(item["gu"], round(item["match_score"], 4))
    print("\n[동 추천]")
    for item in result["dong_recommendations"][:5]:
        print(item["gu"], item["dong"], round(item["match_score"], 4), item["type"])
