from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_ARTIFACT_DIR
from scripts.recommendation import recommendation_ml_utils as ml_utils
from scripts.recommendation import resident_recommender as rr


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = DONGNE_ARTIFACT_DIR / "lgbm_regressor.joblib"
DEFAULT_META_PATH = DONGNE_ARTIFACT_DIR / "lgbm_regressor_meta.json"


def rank_candidates_rule_based(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(candidates, key=lambda row: float(row["rule_recommendation_score"]), reverse=True)


def score_candidates_model_based(
    candidates: list[dict[str, object]],
    model_path: Path,
    meta_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_names = metadata["feature_columns"]
    frame = pd.DataFrame(candidates)
    for feature_name in feature_names:
        if feature_name not in frame.columns:
            frame[feature_name] = 0.0
    scores = model.predict(frame[feature_names])
    ranked_rows = []
    for candidate, score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["model_score"] = float(score)
        ranked_rows.append(enriched)
    return ranked_rows, metadata


def resolve_blend_weights(metadata: dict[str, object]) -> tuple[float, float]:
    metrics = metadata.get("metrics", {})
    total_rows = int(metrics.get("total_rows", metrics.get("train_rows", 0) + metrics.get("test_rows", 0)))
    unique_users = int(metrics.get("unique_users", 0))
    r2 = float(metrics.get("r2", 0.0))

    if total_rows < 500 or unique_users < 30:
        rule_weight, model_weight = 1.0, 0.0
    elif total_rows < 2_000 or unique_users < 100:
        rule_weight, model_weight = 0.9, 0.1
    elif total_rows < 5_000 or unique_users < 250:
        rule_weight, model_weight = 0.8, 0.2
    elif total_rows < 10_000 or unique_users < 500:
        rule_weight, model_weight = 0.6, 0.4
    else:
        rule_weight, model_weight = 0.4, 0.6

    if r2 < 0 and total_rows < 5_000:
        model_weight = min(model_weight, 0.1)
        rule_weight = 1.0 - model_weight
    return rule_weight, model_weight


def rank_candidates_blended(
    candidates: list[dict[str, object]],
    model_path: Path,
    meta_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, float]]:
    scored_rows, metadata = score_candidates_model_based(candidates, model_path, meta_path)
    rule_weight, model_weight = resolve_blend_weights(metadata)

    ranked_rows: list[dict[str, object]] = []
    for row in scored_rows:
        enriched = dict(row)
        enriched["final_score"] = (rule_weight * float(row["rule_recommendation_score"])) + (
            model_weight * float(row["model_score"]) * 100.0
        )
        ranked_rows.append(enriched)

    ranked_rows.sort(key=lambda row: float(row["final_score"]), reverse=True)
    return ranked_rows, metadata, {"rule_weight": rule_weight, "model_weight": model_weight}


def main() -> None:
    parser = argparse.ArgumentParser(description="Return top-k admin dong keys for backend use.")
    parser.add_argument("--answers", required=True, help="Comma-separated q1~q10 answers.")
    parser.add_argument("--behavior-answers", help="Optional comma-separated inferred behavior answers.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of admin dong keys to return.")
    parser.add_argument("--use-trained-model", action="store_true", help="Use trained LightGBM artifacts for reranking.")
    parser.add_argument("--force-rule-based", action="store_true", help="Ignore trained model artifacts.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Path to LightGBM model artifact.")
    parser.add_argument("--meta-path", default=str(DEFAULT_META_PATH), help="Path to model metadata json.")
    args = parser.parse_args()

    answers = rr.parse_answers_argument(args.answers)
    behavior_answers = rr.parse_answers_argument(args.behavior_answers) if args.behavior_answers else None
    blended_answers = rr.blend_explicit_and_behavior_answers(answers, behavior_answers)
    type_result = rr.classify_user_type(blended_answers)

    profile_rows = ml_utils.load_profile_rows()
    candidates = ml_utils.build_candidate_table(blended_answers, profile_rows, type_result=type_result)

    model_path = Path(args.model_path)
    meta_path = Path(args.meta_path)
    use_model = args.use_trained_model and (not args.force_rule_based) and model_path.exists() and meta_path.exists()
    metadata: dict[str, object] | None = None
    blend_weights = {"rule_weight": 1.0, "model_weight": 0.0}
    if use_model:
        ranked, metadata, blend_weights = rank_candidates_blended(candidates, model_path, meta_path)
    else:
        ranked = rank_candidates_rule_based(candidates)

    recommendation_keys = [
        str(row["admin_dong_code"])
        for row in ranked
        if str(row.get("admin_dong_code", "")).strip()
    ][: args.top_k]

    result = {
        "resident_type_key": type_result["type_key"],
        "resident_type_label": type_result["type_label"],
        "scoring_mode": "blended" if use_model and blend_weights["model_weight"] > 0 else "rule_based",
        "blend_weights": blend_weights,
        "top_k": args.top_k,
        "recommendation_keys": recommendation_keys,
    }
    if metadata:
        result["model_training_summary"] = metadata.get("metrics", {})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
