from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_ARTIFACT_DIR
from app.utils.dongne_paths import DONGNE_PROCESSED_DATA_DIR


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = DONGNE_PROCESSED_DATA_DIR / "new_integrated_admin_dong_data.csv"
LEGACY_SOURCE_CSV = DONGNE_PROCESSED_DATA_DIR / "integrated_admin_dong_data.csv"
SOURCE_CSV = DEFAULT_SOURCE_CSV if DEFAULT_SOURCE_CSV.exists() else LEGACY_SOURCE_CSV
SOURCE_TABLE = "new_integrated_admin_dong_data"
SOURCE_COLUMNS = [
    "admin_dong_code",
    "district_name",
    "admin_dong_name",
    "population__total_population",
    "interest__총인구",
    "interest__1인가구수",
    "interest__커뮤니케이션이 적은 집단",
    "interest__평일 외출이 적은 집단",
    "interest__휴일 외출이 적은 집단",
    "interest__외출이 매우 적은 집단(전체)",
    "interest__동영상서비스 이용이 많은 집단",
    "interest__재정상태에 대한 관심집단",
    "interest__외출-커뮤니케이션이 모두 적은 집단(전체)",
    "telecom__총인구수",
    "telecom__평일 총 이동 횟수",
    "telecom__휴일 총 이동 횟수 평균",
    "telecom__집 추정 위치 휴일 총 체류시간",
    "telecom__평균 통화대상자 수",
    "telecom__평균 문자대상자 수",
    "telecom__평균 통화량",
    "telecom__평균 문자량",
    "telecom__금융 서비스 사용일수",
    "telecom__지하철이동일수 합계",
    "telecom__쇼핑 서비스 사용일수",
    "telecom__배달 서비스 사용일수",
    "telecom__동영상/방송 서비스 사용일수",
    "telecom__주간상주지 변경횟수 평균",
    "telecom__야간상주지 변경횟수 평균",
    "telecom__평일 총 이동 거리 합계",
    "telecom__휴일 총 이동 거리 합계",
    "subway__subway_commute_congestion_avg",
    "commerce__overall__점포_수",
    "commerce__외식/카페__점포_수",
    "commerce__소매/유통__점포_수",
    "commerce__여가/오락/숙박__점포_수",
]
PROFILE_CSV = DONGNE_PROCESSED_DATA_DIR / "admin_dong_lifestyle_profiles.csv"
SAMPLE_RESULT_JSON = DONGNE_ARTIFACT_DIR / "sample_recommendations.json"


TYPE_LABELS = {
    "hotplace_explorer": "핫플 탐험가형",
    "emotional_social": "감성 사교형",
    "alley_explorer": "골목 탐험가형",
    "healing_emotional": "힐링 감성형",
    "realistic_life": "현실 라이프형",
    "efficient_life": "효율 생활형",
    "networker": "네트워킹형",
    "balanced_allrounder": "균형 잡힌 올라운더형",
}

TYPE_DESCRIPTIONS = {
    "hotplace_explorer": "새로운 공간과 트렌드를 빠르게 흡수하는 타입입니다. 번화한 상권, 높은 활동성, 다양한 선택지가 있는 동네에서 만족도가 높습니다.",
    "emotional_social": "사람과 분위기를 함께 즐기는 타입입니다. 감성적인 상권과 적당한 활기가 공존하는 동네에서 생활 만족도가 높습니다.",
    "alley_explorer": "혼자서도 자기 취향의 공간을 찾아다니는 타입입니다. 과하게 붐비지 않지만 탐색할 요소가 있는 동네와 잘 맞습니다.",
    "healing_emotional": "집과 가까운 생활권에서 조용한 안정감을 중요하게 생각하는 타입입니다. 번잡함보다 휴식감이 큰 동네에서 편안함을 느낍니다.",
    "realistic_life": "생활 효율과 안정적인 루틴을 중시하는 타입입니다. 실용적인 소비와 무난한 이동 환경을 갖춘 동네와 궁합이 좋습니다.",
    "efficient_life": "바쁘게 움직이지만 불필요한 낭비는 싫어하는 타입입니다. 이동 효율, 생활 편의, 실용 인프라가 좋은 동네에 잘 맞습니다.",
    "networker": "사람과 연결되고 활동 반경도 넓은 타입입니다. 접근성과 만남의 기회가 많은 동네에서 강점을 발휘합니다.",
    "balanced_allrounder": "특정 성향으로 치우치기보다 상황에 따라 유연하게 적응하는 타입입니다. 여러 생활 요소가 고르게 갖춰진 동네가 잘 맞습니다.",
}


@dataclass(frozen=True)
class QuestionDefinition:
    question_id: str
    short_label: str
    dimension_weights: Mapping[str, float]
    feature_weights: Mapping[str, float]


QUESTIONS: List[QuestionDefinition] = [
    QuestionDefinition(
        question_id="weekend_activity",
        short_label="주말 활동성",
        dimension_weights={"activity": 1.1},
        feature_weights={
            "dong_activity_level": 0.45,
            "dong_exploration_level": 0.20,
            "dong_homebody_level": -0.35,
        },
    ),
    QuestionDefinition(
        question_id="contact_style",
        short_label="연락 스타일",
        dimension_weights={"sociability": 1.0},
        feature_weights={
            "dong_social_energy": 0.55,
            "dong_low_communication": -0.45,
        },
    ),
    QuestionDefinition(
        question_id="call_vs_text",
        short_label="통화 선호",
        dimension_weights={"sociability": 0.55, "practicality": 0.15},
        feature_weights={
            "dong_call_preference": 0.75,
            "dong_social_energy": 0.25,
        },
    ),
    QuestionDefinition(
        question_id="finance_interest",
        short_label="재테크 관심도",
        dimension_weights={"practicality": 1.15},
        feature_weights={
            "dong_finance_orientation": 1.0,
        },
    ),
    QuestionDefinition(
        question_id="commute_crowd_tolerance",
        short_label="혼잡 수용도",
        dimension_weights={"activity": 0.45, "practicality": 0.45},
        feature_weights={
            "dong_commute_intensity": 0.8,
            "dong_subway_dependency": 0.2,
        },
    ),
    QuestionDefinition(
        question_id="shopping_style",
        short_label="쇼핑 스타일",
        dimension_weights={"trendiness": 0.75},
        feature_weights={
            "dong_shopping_frequency": 0.75,
            "dong_hotplace_level": 0.25,
        },
    ),
    QuestionDefinition(
        question_id="cooking_vs_delivery",
        short_label="요리 vs 배달",
        dimension_weights={"trendiness": 0.35, "practicality": 0.35},
        feature_weights={
            "dong_delivery_affinity": 0.7,
            "dong_food_commerce_density": 0.3,
        },
    ),
    QuestionDefinition(
        question_id="video_consumption",
        short_label="영상 소비",
        dimension_weights={"trendiness": 0.35, "activity": -0.35},
        feature_weights={
            "dong_video_affinity": 0.75,
            "dong_homebody_level": 0.25,
        },
    ),
    QuestionDefinition(
        question_id="mobility_radius",
        short_label="생활 반경",
        dimension_weights={"activity": 0.7, "trendiness": 0.35},
        feature_weights={
            "dong_exploration_level": 0.65,
            "dong_activity_level": 0.35,
        },
    ),
    QuestionDefinition(
        question_id="hotplace_preference",
        short_label="핫플 선호",
        dimension_weights={"trendiness": 0.9, "sociability": 0.25},
        feature_weights={
            "dong_hotplace_level": 0.55,
            "dong_food_commerce_density": 0.20,
            "dong_social_energy": 0.10,
            "dong_commute_intensity": 0.15,
        },
    ),
]


TYPE_CENTROIDS: Mapping[str, Mapping[str, float]] = {
    "hotplace_explorer": {"activity": 0.95, "sociability": 0.85, "trendiness": 0.95, "practicality": 0.40},
    "emotional_social": {"activity": 0.35, "sociability": 0.88, "trendiness": 0.82, "practicality": 0.30},
    "alley_explorer": {"activity": 0.82, "sociability": 0.25, "trendiness": 0.58, "practicality": 0.35},
    "healing_emotional": {"activity": 0.15, "sociability": 0.18, "trendiness": 0.60, "practicality": 0.25},
    "realistic_life": {"activity": 0.22, "sociability": 0.20, "trendiness": 0.18, "practicality": 0.88},
    "efficient_life": {"activity": 0.82, "sociability": 0.25, "trendiness": 0.18, "practicality": 0.92},
    "networker": {"activity": 0.86, "sociability": 0.94, "trendiness": 0.35, "practicality": 0.78},
    "balanced_allrounder": {"activity": 0.50, "sociability": 0.50, "trendiness": 0.50, "practicality": 0.50},
}


DIMENSIONS = ("activity", "sociability", "trendiness", "practicality")


def parse_number(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    return float(text)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: Sequence[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int((len(sorted_values) - 1) * ratio)
    return sorted_values[max(0, min(idx, len(sorted_values) - 1))]


def robust_minmax(values: Sequence[float]) -> tuple[float, float]:
    sorted_values = sorted(values)
    return percentile(sorted_values, 0.05), percentile(sorted_values, 0.95)


def scale_value(value: float, low: float, high: float) -> float:
    if math.isclose(low, high):
        return 0.5
    return clamp((value - low) / (high - low))


def ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _default_database_url() -> str | None:
    try:
        from app.core.config import settings
    except Exception:
        return None
    return settings.DATABASE_URL


def load_rows_from_database(
    table_name: str = SOURCE_TABLE,
    database_url: str | None = None,
) -> List[Dict[str, str]]:
    database_url = database_url or _default_database_url()
    if not database_url:
        return []

    from sqlalchemy import create_engine
    from sqlalchemy import inspect
    from sqlalchemy import text

    engine = create_engine(database_url, future=True)
    if not inspect(engine).has_table(table_name):
        return []

    with engine.begin() as connection:
        rows = connection.execute(text(f"SELECT * FROM {table_name}")).mappings().all()

    return [
        {str(key): "" if value is None else str(value) for key, value in row.items()}
        for row in rows
    ]


def load_source_rows(database_url: str | None = None) -> List[Dict[str, str]]:
    db_rows = load_rows_from_database(database_url=database_url)
    if db_rows:
        return db_rows
    return load_rows(SOURCE_CSV)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalized_answer(score: float) -> float:
    return clamp((score - 1.0) / 4.0)


def denormalized_score(score_01: float) -> float:
    return 1.0 + clamp(score_01) * 4.0


def blend_explicit_and_behavior_answers(
    explicit_answers: Mapping[str, float],
    behavior_answers: Mapping[str, float] | None = None,
    explicit_weight: float = 0.75,
) -> Dict[str, float]:
    result = dict(explicit_answers)
    if not behavior_answers:
        return result

    for question in QUESTIONS:
        qid = question.question_id
        explicit = explicit_answers.get(qid)
        behavior = behavior_answers.get(qid)
        if explicit is None and behavior is None:
            continue
        if explicit is None:
            result[qid] = behavior
        elif behavior is None:
            result[qid] = explicit
        else:
            result[qid] = explicit * explicit_weight + behavior * (1.0 - explicit_weight)
    return result


def build_profile_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    profile_rows: List[Dict[str, object]] = []
    for row in rows:
        population = parse_number(row.get("population__total_population", row.get("population__total")))
        interest_population = parse_number(row.get("interest__총인구")) or population
        telecom_population = parse_number(row.get("telecom__총인구수")) or population
        call_targets = parse_number(row.get("telecom__평균 통화대상자 수"))
        text_targets = parse_number(row.get("telecom__평균 문자대상자 수"))
        call_volume = parse_number(row.get("telecom__평균 통화량"))
        text_volume = parse_number(row.get("telecom__평균 문자량"))

        profile = {
            "district_name": row["district_name"],
            "admin_dong_name": row["admin_dong_name"],
            "admin_dong_code": row.get("admin_dong_code", row.get("admin_dong_code_10digit", "")),
            "admin_dong_code_10digit": row.get("admin_dong_code_10digit", ""),
            "population_total": population,
            "single_household_ratio": ratio(parse_number(row.get("interest__1인가구수")), interest_population),
            "low_communication_ratio": ratio(parse_number(row.get("interest__커뮤니케이션이 적은 집단")), interest_population),
            "finance_interest_ratio": ratio(parse_number(row.get("interest__재정상태에 대한 관심집단")), interest_population),
            "very_low_outdoor_ratio": ratio(parse_number(row.get("interest__외출이 매우 적은 집단(전체)")), interest_population),
            "high_video_interest_ratio": ratio(parse_number(row.get("interest__동영상서비스 이용이 많은 집단")), interest_population),
            "weekday_move_count": parse_number(row.get("telecom__평일 총 이동 횟수")),
            "weekend_move_count": parse_number(row.get("telecom__휴일 총 이동 횟수 평균")),
            "weekend_home_stay_time": parse_number(row.get("telecom__집 추정 위치 휴일 총 체류시간")),
            "weekday_move_distance": parse_number(row.get("telecom__평일 총 이동 거리 합계")),
            "weekend_move_distance": parse_number(row.get("telecom__휴일 총 이동 거리 합계")),
            "day_location_change_count": parse_number(row.get("telecom__주간상주지 변경횟수 평균")),
            "night_location_change_count": parse_number(row.get("telecom__야간상주지 변경횟수 평균")),
            "finance_service_days": parse_number(row.get("telecom__금융 서비스 사용일수")),
            "call_targets": call_targets,
            "text_targets": text_targets,
            "call_volume": call_volume,
            "text_volume": text_volume,
            "video_service_days": parse_number(row.get("telecom__동영상/방송 서비스 사용일수")),
            "shopping_service_days": parse_number(row.get("telecom__쇼핑 서비스 사용일수")),
            "delivery_service_days": parse_number(row.get("telecom__배달 서비스 사용일수")),
            "commute_congestion_avg": parse_number(row.get("subway__subway_commute_congestion_avg", row.get("subway__commute_congestion_avg"))),
            "subway_use_days": parse_number(row.get("telecom__지하철이동일수 합계")),
            "store_count_total": parse_number(row.get("commerce__overall__점포_수", row.get("commerce__total__store_count"))),
            "store_count_food": parse_number(row.get("commerce__외식/카페__점포_수", row.get("commerce__외식/카페__store_count"))),
            "store_count_leisure": parse_number(row.get("commerce__여가/오락/숙박__점포_수", row.get("commerce__여가/오락/숙박__store_count"))),
            "store_count_life": parse_number(row.get("commerce__생활서비스__점포_수", row.get("commerce__생활서비스__store_count"))),
            "store_count_retail": parse_number(row.get("commerce__소매/유통__점포_수", row.get("commerce__소매/유통__store_count"))),
            "telecom_population_total": telecom_population,
        }

        profile["call_target_share"] = ratio(call_targets, call_targets + text_targets)
        profile["call_volume_share"] = ratio(call_volume, call_volume + text_volume)
        profile["food_commerce_density_raw"] = ratio(
            profile["store_count_food"] + profile["store_count_leisure"],
            population,
        )
        profile["hotplace_density_raw"] = ratio(
            profile["store_count_total"] + profile["store_count_food"] + profile["store_count_leisure"],
            population,
        )
        profile["shopping_density_raw"] = ratio(
            profile["store_count_retail"],
            population,
        )
        profile_rows.append(profile)
    return profile_rows


def normalize_profiles(profile_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    feature_names = [
        "single_household_ratio",
        "low_communication_ratio",
        "finance_interest_ratio",
        "very_low_outdoor_ratio",
        "high_video_interest_ratio",
        "weekday_move_count",
        "weekend_move_count",
        "weekend_home_stay_time",
        "weekday_move_distance",
        "weekend_move_distance",
        "day_location_change_count",
        "night_location_change_count",
        "finance_service_days",
        "call_targets",
        "text_targets",
        "call_target_share",
        "call_volume_share",
        "video_service_days",
        "shopping_service_days",
        "delivery_service_days",
        "commute_congestion_avg",
        "subway_use_days",
        "store_count_total",
        "store_count_food",
        "store_count_leisure",
        "store_count_life",
        "store_count_retail",
        "food_commerce_density_raw",
        "hotplace_density_raw",
        "shopping_density_raw",
    ]
    scalers: Dict[str, tuple[float, float]] = {}
    for feature_name in feature_names:
        values = [parse_number(row.get(feature_name)) for row in profile_rows]
        scalers[feature_name] = robust_minmax(values)

    for row in profile_rows:
        for feature_name in feature_names:
            low, high = scalers[feature_name]
            row[f"norm__{feature_name}"] = scale_value(parse_number(row.get(feature_name)), low, high)

        row["dong_homebody_level"] = mean(
            [
                row["norm__very_low_outdoor_ratio"],
                row["norm__weekend_home_stay_time"],
                1.0 - row["norm__weekend_move_count"],
            ]
        )
        row["dong_activity_level"] = mean(
            [
                row["norm__weekday_move_count"],
                row["norm__weekend_move_count"],
                row["norm__weekday_move_distance"],
                row["norm__weekend_move_distance"],
                1.0 - row["dong_homebody_level"],
            ]
        )
        row["dong_social_energy"] = mean(
            [
                1.0 - row["norm__low_communication_ratio"],
                row["norm__call_targets"],
                row["norm__text_targets"],
            ]
        )
        row["dong_low_communication"] = row["norm__low_communication_ratio"]
        row["dong_call_preference"] = mean(
            [
                row["norm__call_target_share"],
                row["norm__call_volume_share"],
            ]
        )
        row["dong_finance_orientation"] = mean(
            [
                row["norm__finance_interest_ratio"],
                row["norm__finance_service_days"],
            ]
        )
        row["dong_commute_intensity"] = row["norm__commute_congestion_avg"]
        row["dong_subway_dependency"] = row["norm__subway_use_days"]
        row["dong_shopping_frequency"] = mean(
            [
                row["norm__shopping_service_days"],
                row["norm__shopping_density_raw"],
            ]
        )
        row["dong_delivery_affinity"] = mean(
            [
                row["norm__delivery_service_days"],
                row["norm__food_commerce_density_raw"],
            ]
        )
        row["dong_video_affinity"] = mean(
            [
                row["norm__video_service_days"],
                row["norm__high_video_interest_ratio"],
            ]
        )
        row["dong_exploration_level"] = mean(
            [
                row["norm__day_location_change_count"],
                row["norm__night_location_change_count"],
                row["norm__weekday_move_distance"],
                row["norm__weekend_move_distance"],
                row["norm__weekend_move_count"],
            ]
        )
        row["dong_food_commerce_density"] = row["norm__food_commerce_density_raw"]
        row["dong_hotplace_level"] = mean(
            [
                row["norm__hotplace_density_raw"],
                row["norm__store_count_total"],
                row["norm__store_count_food"],
                row["norm__store_count_leisure"],
                row["norm__commute_congestion_avg"],
            ]
        )
    return profile_rows


def score_question(question: QuestionDefinition, row: Mapping[str, object]) -> float:
    weighted_total = 0.0
    weight_sum = 0.0
    for feature_name, weight in question.feature_weights.items():
        feature_value = parse_number(row.get(feature_name))
        aligned_value = feature_value if weight >= 0 else 1.0 - feature_value
        weighted_total += aligned_value * abs(weight)
        weight_sum += abs(weight)
    return weighted_total / weight_sum if weight_sum else 0.5


def classify_user_type(answers: Mapping[str, float]) -> Dict[str, object]:
    dimensions = {dimension: 0.0 for dimension in DIMENSIONS}
    weights = {dimension: 0.0 for dimension in DIMENSIONS}

    for question in QUESTIONS:
        user_score = normalized_answer(answers[question.question_id])
        centered = user_score - 0.5
        for dimension, weight in question.dimension_weights.items():
            direction = 1.0 if weight >= 0 else -1.0
            magnitude = abs(weight)
            dimensions[dimension] += ((centered * direction) + 0.5) * magnitude
            weights[dimension] += magnitude

    for dimension in DIMENSIONS:
        if weights[dimension]:
            dimensions[dimension] = clamp(dimensions[dimension] / weights[dimension])
        else:
            dimensions[dimension] = 0.5

    type_scores: List[tuple[str, float]] = []
    for type_key, centroid in TYPE_CENTROIDS.items():
        distance = math.sqrt(
            sum((dimensions[dimension] - centroid[dimension]) ** 2 for dimension in DIMENSIONS) / len(DIMENSIONS)
        )
        fit_score = 1.0 - clamp(distance / math.sqrt(1.0))
        type_scores.append((type_key, fit_score))
    type_scores.sort(key=lambda item: item[1], reverse=True)

    top_type_key, top_fit = type_scores[0]
    return {
        "type_key": top_type_key,
        "type_label": TYPE_LABELS[top_type_key],
        "type_fit_score": round(top_fit * 100, 2),
        "dimensions": {k: round(v, 4) for k, v in dimensions.items()},
        "type_rankings": [
            {"type_key": key, "type_label": TYPE_LABELS[key], "score": round(score * 100, 2)}
            for key, score in type_scores
        ],
    }


def recommend_admin_dongs(
    answers: Mapping[str, float],
    profile_rows: List[Dict[str, object]],
    type_result: Mapping[str, object] | None = None,
    top_k: int = 10,
) -> List[Dict[str, object]]:
    if type_result is None:
        type_result = classify_user_type(answers)
    centroid = TYPE_CENTROIDS[str(type_result["type_key"])]

    scored_rows: List[Dict[str, object]] = []
    for row in profile_rows:
        question_scores: Dict[str, float] = {}
        similarities: List[float] = []
        for question in QUESTIONS:
            target_score = score_question(question, row)
            user_score = normalized_answer(answers[question.question_id])
            similarity = 1.0 - abs(user_score - target_score)
            question_scores[question.question_id] = similarity
            similarities.append(similarity)

        lifestyle_fit = mean(similarities)
        row_dimensions = estimate_row_dimensions(row)
        dimension_fit = 1.0 - math.sqrt(
            sum((row_dimensions[d] - centroid[d]) ** 2 for d in DIMENSIONS) / len(DIMENSIONS)
        )
        recommendation_score = 0.75 * lifestyle_fit + 0.25 * dimension_fit
        scored_rows.append(
            {
                "district_name": row["district_name"],
                "admin_dong_name": row["admin_dong_name"],
                "admin_dong_code": row["admin_dong_code"],
                "admin_dong_code_10digit": row["admin_dong_code_10digit"],
                "recommendation_score": round(recommendation_score * 100, 2),
                "lifestyle_fit_score": round(lifestyle_fit * 100, 2),
                "dimension_fit_score": round(dimension_fit * 100, 2),
                "question_similarity": {qid: round(score * 100, 2) for qid, score in question_scores.items()},
                "signals": build_signal_summary(row),
            }
        )

    scored_rows.sort(key=lambda item: item["recommendation_score"], reverse=True)
    return scored_rows[:top_k]


def group_recommendations_by_district(
    recommendations: List[Dict[str, object]],
    district_count: int = 3,
    dongs_per_district: int = 2,
) -> List[Dict[str, object]]:
    grouped: List[Dict[str, object]] = []
    used_districts = set()

    for recommendation in recommendations:
        district = str(recommendation["district_name"])
        if district in used_districts:
            continue
        district_rows = [row for row in recommendations if row["district_name"] == district][:dongs_per_district]
        if len(district_rows) < dongs_per_district:
            continue
        grouped.append(
            {
                "district_name": district,
                "admin_dongs": district_rows,
                "district_recommendation_score": round(
                    mean(row["recommendation_score"] for row in district_rows), 2
                ),
            }
        )
        used_districts.add(district)
        if len(grouped) >= district_count:
            break
    return grouped


def estimate_row_dimensions(row: Mapping[str, object]) -> Dict[str, float]:
    return {
        "activity": mean([parse_number(row.get("dong_activity_level")), parse_number(row.get("dong_exploration_level"))]),
        "sociability": mean([parse_number(row.get("dong_social_energy")), 1.0 - parse_number(row.get("dong_low_communication"))]),
        "trendiness": mean(
            [
                parse_number(row.get("dong_hotplace_level")),
                parse_number(row.get("dong_shopping_frequency")),
                parse_number(row.get("dong_video_affinity")),
            ]
        ),
        "practicality": mean(
            [
                parse_number(row.get("dong_finance_orientation")),
                parse_number(row.get("dong_subway_dependency")),
                parse_number(row.get("dong_delivery_affinity")),
            ]
        ),
    }


def build_signal_summary(row: Mapping[str, object]) -> Dict[str, float]:
    return {
        "activity": round(parse_number(row.get("dong_activity_level")) * 100, 2),
        "homebody": round(parse_number(row.get("dong_homebody_level")) * 100, 2),
        "social": round(parse_number(row.get("dong_social_energy")) * 100, 2),
        "finance": round(parse_number(row.get("dong_finance_orientation")) * 100, 2),
        "commute": round(parse_number(row.get("dong_commute_intensity")) * 100, 2),
        "shopping": round(parse_number(row.get("dong_shopping_frequency")) * 100, 2),
        "delivery": round(parse_number(row.get("dong_delivery_affinity")) * 100, 2),
        "video": round(parse_number(row.get("dong_video_affinity")) * 100, 2),
        "exploration": round(parse_number(row.get("dong_exploration_level")) * 100, 2),
        "hotplace": round(parse_number(row.get("dong_hotplace_level")) * 100, 2),
    }


def describe_type(type_key: str) -> str:
    return TYPE_DESCRIPTIONS[type_key]


def build_reason_lines(recommendation: Mapping[str, object]) -> List[str]:
    signals = recommendation["signals"]
    question_similarity = recommendation["question_similarity"]

    signal_labels = {
        "activity": "활동 반경이 넓고 이동성이 높은 편",
        "homebody": "집 중심으로 쉬기 좋은 성향이 강한 편",
        "social": "사교성과 연락 빈도가 비교적 높은 편",
        "finance": "재테크·실용 소비 성향이 강한 편",
        "commute": "출퇴근 이동 강도와 교통 활용도가 높은 편",
        "shopping": "쇼핑/소비 활동성이 높은 편",
        "delivery": "배달·외식 접근성이 생활 패턴과 잘 맞는 편",
        "video": "영상 소비 친화적인 생활 패턴이 강한 편",
        "exploration": "새로운 생활권을 오가며 탐색하는 성향이 있는 편",
        "hotplace": "상권 밀도와 핫플 감도가 높은 편",
    }
    top_signals = sorted(signals.items(), key=lambda item: item[1], reverse=True)[:3]
    top_questions = sorted(question_similarity.items(), key=lambda item: item[1], reverse=True)[:3]

    question_labels = {
        question.question_id: question.short_label for question in QUESTIONS
    }

    line1 = f"{recommendation['district_name']} {recommendation['admin_dong_name']}은(는) 전체 추천 점수 {recommendation['recommendation_score']}점으로 상위권에 들어온 동네입니다."
    signal_text = ", ".join(signal_labels[key] for key, _ in top_signals)
    line2 = f"특히 이 동네는 {signal_text} 쪽 신호가 강해서, 당신의 생활 패턴과 잘 맞는 편으로 계산됐습니다."
    question_text = ", ".join(question_labels[key] for key, _ in top_questions)
    line3 = f"질문 기준으로는 `{question_text}` 문항과의 적합도가 높게 나와 추천 우선순위가 올라갔습니다."
    return [line1, line2, line3]


def build_profile_csv(rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "district_name",
        "admin_dong_name",
        "admin_dong_code",
        "admin_dong_code_10digit",
        "population_total",
        "single_household_ratio",
        "low_communication_ratio",
        "finance_interest_ratio",
        "very_low_outdoor_ratio",
        "high_video_interest_ratio",
        "weekday_move_count",
        "weekend_move_count",
        "weekend_home_stay_time",
        "day_location_change_count",
        "night_location_change_count",
        "finance_service_days",
        "call_target_share",
        "video_service_days",
        "shopping_service_days",
        "delivery_service_days",
        "commute_congestion_avg",
        "store_count_total",
        "store_count_food",
        "store_count_leisure",
        "dong_homebody_level",
        "dong_activity_level",
        "dong_social_energy",
        "dong_low_communication",
        "dong_call_preference",
        "dong_finance_orientation",
        "dong_commute_intensity",
        "dong_subway_dependency",
        "dong_shopping_frequency",
        "dong_delivery_affinity",
        "dong_video_affinity",
        "dong_exploration_level",
        "dong_food_commerce_density",
        "dong_hotplace_level",
    ]

    out_rows: List[Dict[str, object]] = []
    for row in rows:
        out_row = {}
        for field in fieldnames:
            value = row.get(field, "")
            if isinstance(value, float):
                out_row[field] = f"{value:.6f}".rstrip("0").rstrip(".")
            else:
                out_row[field] = value
        out_rows.append(out_row)
    write_csv(PROFILE_CSV, out_rows, fieldnames)


def parse_answers_argument(raw: str) -> Dict[str, float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != len(QUESTIONS):
        raise ValueError(f"Expected {len(QUESTIONS)} answers, got {len(parts)}")
    answers: Dict[str, float] = {}
    for question, part in zip(QUESTIONS, parts):
        score = float(part)
        if score < 1 or score > 5:
            raise ValueError(f"Answer for {question.question_id} must be between 1 and 5")
        answers[question.question_id] = score
    return answers


def build_sample_personas() -> Dict[str, Dict[str, float]]:
    return {
        "hotplace_explorer_sample": parse_answers_argument("5,5,4,3,5,5,5,4,5,5"),
        "healing_emotional_sample": parse_answers_argument("1,1,1,2,1,1,1,4,1,1"),
        "efficient_life_sample": parse_answers_argument("4,2,4,5,4,2,4,2,4,2"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify resident type and recommend Seoul admin dongs.")
    parser.add_argument("--answers", help="Comma-separated 10 answers in the same order as the survey questions.")
    parser.add_argument("--behavior-answers", help="Optional comma-separated inferred answers from behavior logs.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to print.")
    parser.add_argument("--write-samples", action="store_true", help="Write sample recommendation outputs to JSON.")
    args = parser.parse_args()

    raw_rows = load_source_rows()
    profile_rows = normalize_profiles(build_profile_rows(raw_rows))
    build_profile_csv(profile_rows)

    if args.write_samples:
        payload = {}
        for sample_name, answers in build_sample_personas().items():
            type_result = classify_user_type(answers)
            recommendations = recommend_admin_dongs(answers, profile_rows, type_result=type_result, top_k=30)
            grouped_recommendations = group_recommendations_by_district(recommendations)
            primary_recommendation = grouped_recommendations[0]["admin_dongs"][0] if grouped_recommendations else None
            payload[sample_name] = {
                "resident_type": type_result,
                "type_description": describe_type(str(type_result["type_key"])),
                "grouped_recommendations": grouped_recommendations,
                "primary_recommendation": primary_recommendation,
                "primary_recommendation_reasons": build_reason_lines(primary_recommendation) if primary_recommendation else [],
            }
        SAMPLE_RESULT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.answers:
        print(f"profile_csv={PROFILE_CSV.name}")
        if args.write_samples:
            print(f"sample_recommendations={SAMPLE_RESULT_JSON.name}")
        return

    answers = parse_answers_argument(args.answers)
    behavior_answers = parse_answers_argument(args.behavior_answers) if args.behavior_answers else None
    blended_answers = blend_explicit_and_behavior_answers(answers, behavior_answers)
    type_result = classify_user_type(blended_answers)
    recommendations = recommend_admin_dongs(blended_answers, profile_rows, type_result=type_result, top_k=max(args.top_k, 30))
    grouped_recommendations = group_recommendations_by_district(recommendations)
    primary_recommendation = grouped_recommendations[0]["admin_dongs"][0] if grouped_recommendations else None

    result = {
        "resident_type": {
            **type_result,
            "description": describe_type(str(type_result["type_key"])),
        },
        "recommended_district_groups": grouped_recommendations,
        "primary_recommendation": primary_recommendation,
        "primary_recommendation_reasons": build_reason_lines(primary_recommendation) if primary_recommendation else [],
        "profile_csv": PROFILE_CSV.name,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
