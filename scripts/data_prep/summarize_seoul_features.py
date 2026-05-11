# feature 수정
from pathlib import Path
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_S3_DATA_DIR
from app.utils.dongne_paths import DONGNE_INTEREST_CSV
from app.utils.dongne_paths import DONGNE_TELECOM_CSV
from app.utils.s3_csv import find_csv_path
from app.utils.s3_csv import read_csv_dataframe


def to_numeric_series(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "nan": np.nan}),
        errors="coerce",
    )


base = DONGNE_S3_DATA_DIR
if base == DONGNE_S3_DATA_DIR:
    interest_path = DONGNE_INTEREST_CSV
    telecom_path = DONGNE_TELECOM_CSV
else:
    interest_path = find_csv_path(base, "관심집단수")
    telecom_path = find_csv_path(base, "통신정보")

interest = read_csv_dataframe(interest_path, encoding="utf-8-sig")
telecom = read_csv_dataframe(telecom_path, encoding="utf-8-sig")

interest = interest.loc[:, [c for c in interest.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()
telecom = telecom.loc[:, [c for c in telecom.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()
telecom = telecom.rename(columns={"행정동": "행정동명", "총인구수": "총인구_통신정보", "1인가구수": "1인가구수_통신정보"})

interest_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]
telecom_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]

for col in interest.columns:
    if col not in interest_id_cols:
        interest[col] = to_numeric_series(interest[col])

for col in telecom.columns:
    if col not in telecom_id_cols:
        telecom[col] = to_numeric_series(telecom[col])

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

rows = []
for keys, group in merged.groupby(["행정동코드", "자치구", "행정동명"], dropna=False):
    row = {"행정동코드": keys[0], "자치구": keys[1], "행정동명": keys[2], "총인구": group["총인구"].sum()}
    for col in interest_count_cols:
        row[col] = group[col].sum()
        row[f"{col}비율"] = row[col] / row["총인구"] if row["총인구"] else np.nan

    weights = group["총인구"].fillna(0)
    for col in telecom_feature_cols:
        valid_mask = group[col].notna() & weights.notna()
        if valid_mask.any() and weights[valid_mask].sum() > 0:
            row[col] = np.average(group.loc[valid_mask, col], weights=weights[valid_mask])
        else:
            row[col] = np.nan
    rows.append(row)

dong = pd.DataFrame(rows)

for col in [
    "1인가구수비율",
    "커뮤니케이션이 적은 집단비율",
    "외출이 매우 적은 집단(전체)비율",
    "재정상태에 대한 관심집단비율",
    "외출-커뮤니케이션이 모두 적은 집단(전체)비율",
    "최근 3개월 내 요금 연체 비율",
    "집 추정 위치 휴일 총 체류시간",
    "평일 총 이동 횟수",
    "휴일 총 이동 횟수 평균",
    "평균 통화대상자 수",
    "평균 문자대상자 수",
]:
    std = dong[col].std(ddof=0)
    dong[f"{col}_z"] = (dong[col] - dong[col].mean()) / std if pd.notna(std) and std != 0 else 0

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

risk_cols = [
    "1인가구수비율_z",
    "커뮤니케이션이 적은 집단비율_z",
    "외출이 매우 적은 집단(전체)비율_z",
    "재정상태에 대한 관심집단비율_z",
    "외출-커뮤니케이션이 모두 적은 집단(전체)비율_z",
    "최근 3개월 내 요금 연체 비율_z",
    "집 추정 위치 휴일 총 체류시간_z",
    "평일 총 이동 횟수_z",
    "휴일 총 이동 횟수 평균_z",
    "평균 통화대상자 수_z",
    "평균 문자대상자 수_z",
]
dong["행정동판단지수"] = (
    dong[
        [
            "1인가구수비율_z",
            "커뮤니케이션이 적은 집단비율_z",
            "외출이 매우 적은 집단(전체)비율_z",
            "재정상태에 대한 관심집단비율_z",
            "외출-커뮤니케이션이 모두 적은 집단(전체)비율_z",
            "최근 3개월 내 요금 연체 비율_z",
            "집 추정 위치 휴일 총 체류시간_z",
        ]
    ].sum(axis=1)
    - dong[["평일 총 이동 횟수_z", "휴일 총 이동 횟수 평균_z", "평균 통화대상자 수_z", "평균 문자대상자 수_z"]].sum(axis=1)
) / 11

gu_sum_cols = ["총인구"] + interest_count_cols
gu = dong.groupby("자치구", as_index=False)[gu_sum_cols].sum()
for col in interest_count_cols:
    gu[f"{col}비율"] = gu[col] / gu["총인구"]
for col in telecom_feature_cols:
    gu[col] = dong.groupby("자치구").apply(
        lambda x: (x[col] * x["총인구"]).sum() / x["총인구"].sum() if x[col].notna().any() else np.nan
    ).values

print("[구 기준 상위 특징]")
for col in [
    "1인가구수비율",
    "외출-커뮤니케이션이 모두 적은 집단(전체)비율",
    "재정상태에 대한 관심집단비율",
    "최근 3개월 내 요금 연체 비율",
]:
    print(f"\nTOP {col}")
    print(gu.nlargest(5, col)[["자치구", col]].to_string(index=False))

print("\nLOW 평일 총 이동 횟수")
print(gu.nsmallest(5, "평일 총 이동 횟수")[["자치구", "평일 총 이동 횟수"]].to_string(index=False))

print("\n[동 기준 상위 판단지수]")
print(dong.nlargest(15, "행정동판단지수")[["자치구", "행정동명", "대표유형", "행정동판단지수"]].to_string(index=False))

print("\n[동 유형 분포]")
print(dong["대표유형"].value_counts().to_string())
