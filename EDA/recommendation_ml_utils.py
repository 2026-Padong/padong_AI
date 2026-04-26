from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import resident_recommender as rr


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "artifacts"
QUESTION_ID_TO_QKEY = {question.question_id: f"q{index}" for index, question in enumerate(rr.QUESTIONS, start=1)}
QKEY_TO_QUESTION_ID = {value: key for key, value in QUESTION_ID_TO_QKEY.items()}
IDENTIFIER_COLUMNS = {
    "district_name",
    "admin_dong_name",
    "admin_dong_code",
    "admin_dong_code_10digit",
}


def load_profile_rows() -> List[Dict[str, object]]:
    raw_rows = rr.load_rows(rr.SOURCE_CSV)
    return rr.normalize_profiles(rr.build_profile_rows(raw_rows))


def load_profile_lookup() -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for row in load_profile_rows():
        code = str(row.get("admin_dong_code", "")).strip()
        if code:
            lookup[code] = row
    return lookup


def parse_answers_from_log_row(row: Mapping[str, str]) -> Dict[str, float]:
    answers: Dict[str, float] = {}
    for index, question in enumerate(rr.QUESTIONS, start=1):
        raw_value = row.get(f"q{index}") or row.get(question.question_id)
        if raw_value is None or str(raw_value).strip() == "":
            raise ValueError(f"Missing answer column for q{index} / {question.question_id}")
        score = float(raw_value)
        if score < 1 or score > 5:
            raise ValueError(f"Invalid score for q{index}: {score}")
        answers[question.question_id] = score
    return answers


def compute_label(clicked: float, liked: float, dwell_time_sec: float, dwell_cap_sec: float = 120.0) -> float:
    dwell_norm = max(0.0, min(float(dwell_time_sec) / dwell_cap_sec, 1.0))
    return round((0.2 * float(clicked)) + (0.6 * float(liked)) + (0.2 * dwell_norm), 6)


def build_candidate_features(
    answers: Mapping[str, float],
    profile_row: Mapping[str, object],
    type_result: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    if type_result is None:
        type_result = rr.classify_user_type(answers)

    user_dimensions = {key: float(value) for key, value in type_result["dimensions"].items()}
    row_dimensions = rr.estimate_row_dimensions(profile_row)
    centroid = rr.TYPE_CENTROIDS[str(type_result["type_key"])]

    features: Dict[str, object] = {
        "district_name": str(profile_row["district_name"]),
        "admin_dong_name": str(profile_row["admin_dong_name"]),
        "admin_dong_code": str(profile_row.get("admin_dong_code", "")),
        "admin_dong_code_10digit": str(profile_row.get("admin_dong_code_10digit", "")),
        "predicted_type_key": str(type_result["type_key"]),
        "predicted_type_label": str(type_result["type_label"]),
        "predicted_type_fit_score": float(type_result["type_fit_score"]),
    }

    for field, value in profile_row.items():
        if field in IDENTIFIER_COLUMNS:
            continue
        if isinstance(value, (int, float)):
            features[f"dong__{field}"] = float(value)

    similarities: List[float] = []
    for index, question in enumerate(rr.QUESTIONS, start=1):
        qkey = f"q{index}"
        user_score = float(answers[question.question_id])
        user_score_norm = rr.normalized_answer(user_score)
        target_score = rr.score_question(question, profile_row)
        similarity = 1.0 - abs(user_score_norm - target_score)

        features[qkey] = user_score
        features[f"{qkey}_norm"] = user_score_norm
        features[f"{question.question_id}__target"] = target_score
        features[f"{question.question_id}__similarity"] = similarity
        features[f"{question.question_id}__interaction"] = user_score_norm * target_score
        similarities.append(similarity)

    lifestyle_fit = rr.mean(similarities)
    dimension_fit = 1.0 - rr.math.sqrt(
        sum((row_dimensions[dimension] - centroid[dimension]) ** 2 for dimension in rr.DIMENSIONS) / len(rr.DIMENSIONS)
    )
    rule_recommendation_score = 0.75 * lifestyle_fit + 0.25 * dimension_fit

    for dimension in rr.DIMENSIONS:
        features[f"user_dimension__{dimension}"] = user_dimensions[dimension]
        features[f"dong_dimension__{dimension}"] = row_dimensions[dimension]
        features[f"dimension_match__{dimension}"] = 1.0 - abs(user_dimensions[dimension] - row_dimensions[dimension])

    for signal_name, signal_value in rr.build_signal_summary(profile_row).items():
        features[f"signal__{signal_name}"] = float(signal_value) / 100.0

    features["rule_lifestyle_fit_score"] = lifestyle_fit
    features["rule_dimension_fit_score"] = dimension_fit
    features["rule_recommendation_score"] = rule_recommendation_score
    return features


def build_candidate_table(
    answers: Mapping[str, float],
    profile_rows: Sequence[Mapping[str, object]],
    type_result: Mapping[str, object] | None = None,
) -> List[Dict[str, object]]:
    if type_result is None:
        type_result = rr.classify_user_type(answers)
    return [build_candidate_features(answers, row, type_result=type_result) for row in profile_rows]


def feature_columns(rows: Sequence[Mapping[str, object]], exclude: Iterable[str] | None = None) -> List[str]:
    excluded = set(exclude or set())
    if not rows:
        return []
    columns: List[str] = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float)):
            columns.append(key)
    return columns

