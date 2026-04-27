from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_ARTIFACT_DIR
from app.utils.dongne_paths import DONGNE_PROCESSED_DATA_DIR
from scripts.recommendation import build_pair_dataset
from scripts.recommendation import recommendation_ml_utils as ml_utils


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_CSV = DONGNE_PROCESSED_DATA_DIR / "pair_training_dataset.csv"
DEFAULT_ARTIFACT_DIR = DONGNE_ARTIFACT_DIR

NON_FEATURE_COLUMNS = {
    "user_id",
    "session_id",
    "event_at",
    "admin_dong_code",
    "admin_dong_code_10digit",
    "district_name",
    "admin_dong_name",
    "predicted_type_key",
    "predicted_type_label",
    "impression",
    "rank_position",
    "clicked",
    "liked",
    "dwell_time_sec",
    "label",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first-pass LightGBM regressor for admin-dong preference.")
    parser.add_argument("--dataset-csv", help="Optional pair dataset csv path. If omitted, build from DB logs directly.")
    parser.add_argument("--export-dataset-csv", default=str(DEFAULT_DATASET_CSV), help="Optional snapshot csv path to export the DB-built dataset before training.")
    parser.add_argument("--dwell-cap-sec", type=float, default=120.0, help="Cap for dwell-time normalization when building dataset from DB.")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Output artifact directory.")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv) if args.dataset_csv else None
    export_dataset_csv = Path(args.export_dataset_csv) if args.export_dataset_csv else None
    artifact_dir = Path(args.artifact_dir)
    if dataset_csv is not None and not dataset_csv.is_absolute():
        dataset_csv = PROJECT_ROOT / dataset_csv
    if export_dataset_csv is not None and not export_dataset_csv.is_absolute():
        export_dataset_csv = PROJECT_ROOT / export_dataset_csv
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if dataset_csv is not None:
        frame = pd.read_csv(dataset_csv, encoding="utf-8-sig")
        dataset_source = str(dataset_csv)
    else:
        frame = build_pair_dataset.build_pair_dataset_frame(dwell_cap_sec=args.dwell_cap_sec)
        dataset_source = "database:user_recommendation_logs"
        if export_dataset_csv is not None:
            export_dataset_csv.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(export_dataset_csv, index=False, encoding="utf-8-sig")

    if frame.empty:
        raise ValueError("Pair dataset is empty.")

    candidate_rows = frame.to_dict(orient="records")
    feature_columns = ml_utils.feature_columns(candidate_rows, exclude=NON_FEATURE_COLUMNS)
    if not feature_columns:
        raise ValueError("No numeric feature columns found for training.")

    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(frame["label"], errors="coerce").fillna(0.0)

    if "user_id" in frame.columns and frame["user_id"].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_index, test_index = next(splitter.split(X, y, groups=frame["user_id"]))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rmse": round(mean_squared_error(y_test, predictions) ** 0.5, 6),
        "mae": round(mean_absolute_error(y_test, predictions), 6),
        "r2": round(r2_score(y_test, predictions), 6),
        "total_rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "unique_users": int(frame["user_id"].nunique()) if "user_id" in frame.columns else 0,
        "unique_sessions": int(frame["session_id"].nunique()) if "session_id" in frame.columns else 0,
        "label_mean": round(float(y.mean()), 6),
        "feature_count": int(len(feature_columns)),
    }

    model_path = artifact_dir / "lgbm_regressor.joblib"
    meta_path = artifact_dir / "lgbm_regressor_meta.json"
    importance_path = artifact_dir / "lgbm_feature_importance.csv"

    joblib.dump(model, model_path)
    meta_path.write_text(
        json.dumps(
            {
                "dataset_source": dataset_source,
                "feature_columns": feature_columns,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    importance_frame = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(importance_path, index=False, encoding="utf-8-sig")

    print(json.dumps({"model_path": str(model_path), "meta_path": str(meta_path), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
