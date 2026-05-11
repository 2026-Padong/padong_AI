#행정동 기준

from pathlib import Path
import nbformat as nbf
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_RAW_DATA_DIR

NOTEBOOK_OUTPUT_PATH = PROJECT_ROOT / "EDA" / "2025_12_3csv_dong_integrated_eda.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


cells = [
    md(
        """# 2025년 12월 행정동 통합 EDA

`2025.12월_10개 관심집단수.csv`, `2025.12월_29개 통신정보.csv`, `행정구역_읍면동_별_5세별_인구.csv`를 결합해 행정동 기준 인사이트를 도출합니다.

- 분석 단위: 행정동 중심
- 결합 레벨: `행정동코드 + 성별 + 연령대` 후 행정동 집계
- 목표: 관심집단 특성과 통신/이동 행동 특성을 함께 반영한 행정동 인사이트 강화
"""
    ),
    code(
        """from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

PROJECT_ROOT = Path(r'""" + str(PROJECT_ROOT) + """')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.dongne_paths import DONGNE_RAW_DATA_DIR
from app.utils.s3_csv import find_csv_path
from app.utils.s3_csv import read_csv_dataframe

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="talk")

font_candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic"]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
selected_font = next((font for font in font_candidates if font in available_fonts), None)
if selected_font:
    rcParams["font.family"] = selected_font
else:
    print("경고: 한글 폰트를 찾지 못했습니다. 일부 그래프 레이블이 깨질 수 있습니다.")
rcParams["axes.unicode_minus"] = False
"""
    ),
    code(
        """base_path = DONGNE_RAW_DATA_DIR
pop_path = find_csv_path(base_path, "5세별_인구")
interest_path = find_csv_path(base_path, "관심집단수")
telecom_path = find_csv_path(base_path, "통신정보")

print("공공 인구 파일:", pop_path.name)
print("관심집단 파일:", interest_path.name)
print("통신정보 파일:", telecom_path.name)
"""
    ),
    code(
        """interest_raw = read_csv_dataframe(interest_path, encoding="utf-8-sig")
telecom_raw = read_csv_dataframe(telecom_path, encoding="utf-8-sig")
pop_raw = read_csv_dataframe(pop_path, encoding="utf-8-sig", header=None)
pop_raw.columns = ["지역", "항목", "값"]

interest_raw = interest_raw.loc[:, [c for c in interest_raw.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()
telecom_raw = telecom_raw.loc[:, [c for c in telecom_raw.columns if str(c).strip() and not str(c).startswith("Unnamed:")]].copy()
pop_raw = pop_raw.dropna(subset=["지역"]).copy()

print("관심집단 shape:", interest_raw.shape)
print("통신정보 shape:", telecom_raw.shape)
print("공공 인구 shape:", pop_raw.shape)

display(interest_raw.head(2))
display(telecom_raw.head(2))
display(pop_raw.head(5))
"""
    ),
    md(
        """## 1. 전처리 및 병합 준비

세 파일을 같은 행정동 기준으로 보기 위해 이름 표기와 숫자형 컬럼을 정리합니다.  
공공 인구 파일은 계층형 텍스트 구조라서 `자치구-행정동-총인구` 형태로 재구성합니다.
"""
    ),
    code(
        """SPACE_PATTERN = re.compile(r"^[\\s\\u3000]+")

def strip_leading_space(text):
    return SPACE_PATTERN.sub("", str(text)).strip()

def leading_len(text):
    matched = SPACE_PATTERN.match(str(text))
    return len(matched.group(0)) if matched else 0

def normalize_dong_name(text):
    s = str(text).strip()
    s = s.replace("·", ".")
    s = re.sub(r"제(?=\\d)", "", s)
    s = re.sub(r"\\s+", "", s)
    return s

def to_numeric_series(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "nan": np.nan}),
        errors="coerce",
    )

interest = interest_raw.copy()
telecom = telecom_raw.copy()

interest.columns = [str(c).strip() for c in interest.columns]
telecom.columns = [str(c).strip() for c in telecom.columns]
telecom = telecom.rename(columns={"행정동": "행정동명", "총인구수": "총인구_통신정보", "1인가구수": "1인가구수_통신정보"})

interest_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]
telecom_id_cols = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]

for col in interest.columns:
    if col not in interest_id_cols:
        interest[col] = to_numeric_series(interest[col])

for col in telecom.columns:
    if col not in telecom_id_cols:
        telecom[col] = to_numeric_series(telecom[col])

for col in ["행정동코드", "성별", "연령대"]:
    interest[col] = pd.to_numeric(interest[col], errors="coerce").astype("Int64")
    telecom[col] = pd.to_numeric(telecom[col], errors="coerce").astype("Int64")

interest["성별라벨"] = interest["성별"].map({1: "남성", 2: "여성"}).fillna("기타")

pop_work = pop_raw.copy()
pop_work["들여쓰기"] = pop_work["지역"].map(leading_len)
pop_work["지역정리"] = pop_work["지역"].map(strip_leading_space)

current_gu = None
pop_rows = []
for _, row in pop_work.iterrows():
    if row["들여쓰기"] == 3:
        current_gu = row["지역정리"]
    elif row["들여쓰기"] == 6 and current_gu is not None:
        pop_rows.append(
            {
                "자치구": current_gu,
                "행정동명_공공": row["지역정리"],
                "행정동총인구_공공": pd.to_numeric(str(row["값"]).replace(",", ""), errors="coerce"),
            }
        )

pop_df = pd.DataFrame(pop_rows)
pop_df["행정동명_norm"] = pop_df["행정동명_공공"].map(normalize_dong_name)
interest["행정동명_norm"] = interest["행정동명"].map(normalize_dong_name)
telecom["행정동명_norm"] = telecom["행정동명"].map(normalize_dong_name)

print("공공 인구 행정동 수:", len(pop_df))
display(pop_df.head())
"""
    ),
    code(
        """merge_keys = ["행정동코드", "자치구", "행정동명", "성별", "연령대"]
merge_status = interest[merge_keys].merge(
    telecom[merge_keys],
    on=merge_keys,
    how="outer",
    indicator=True,
)

print("관심집단-통신정보 키 병합 상태")
print(merge_status["_merge"].value_counts())

merged_cell = interest.merge(
    telecom.drop(columns=["자치구", "행정동명_norm"]),
    on=["행정동코드", "행정동명", "성별", "연령대"],
    how="left",
    suffixes=("", "_통신"),
)

print("병합 후 셀 단위 shape:", merged_cell.shape)
display(merged_cell.head(2))
"""
    ),
    md(
        """## 2. 행정동 단위 통합 데이터셋 생성

관심집단 수치는 합계로, 통신/행동 지표는 `총인구` 가중평균으로 집계합니다.  
이후 공공 인구와 결합하고 행정동 기준 파생 지표를 만듭니다.
"""
    ),
    code(
        """interest_count_cols = [
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
    "야간상주지 변경횟수 평균",
    "주간상주지 변경횟수 평균",
    "평균 출근 소요시간 평균",
    "평균 근무시간 평균",
    "최근 3개월 내 요금 연체 비율",
    "SNS 사용횟수",
    "평균 통화량",
    "평균 문자량",
    "평균 통화대상자 수",
    "평균 문자대상자 수",
    "데이터 사용량",
    "평일 총 이동 횟수",
    "휴일 총 이동 횟수 평균",
    "집 추정 위치 평일 총 체류시간",
    "집 추정 위치 휴일 총 체류시간",
    "평일 총 이동 거리 합계",
    "휴일 총 이동 거리 합계",
    "지하철이동일수 합계",
    "게임 서비스 사용일수",
    "금융 서비스 사용일수",
    "쇼핑 서비스 사용일수",
    "동영상/방송 서비스 사용일수",
    "유튜브 사용일수",
    "넷플릭스 사용일수",
    "배달 서비스 사용일수",
    "배달_브랜드 서비스 사용일수",
    "배달_식재료 서비스 사용일수",
]
telecom_feature_cols = [c for c in telecom_feature_cols if c in merged_cell.columns]

dong_rows = []
group_cols = ["행정동코드", "자치구", "행정동명", "행정동명_norm"]

for keys, group in merged_cell.groupby(group_cols, dropna=False):
    row = dict(zip(group_cols, keys))
    total_pop = group["총인구"].sum()
    row["총인구"] = total_pop
    row["성별연령셀수"] = len(group)

    for col in interest_count_cols:
        row[col] = group[col].sum()

    if "총인구_통신정보" in group.columns:
        row["총인구_통신정보합"] = group["총인구_통신정보"].sum()
        row["총인구_차이"] = row["총인구_통신정보합"] - total_pop

    weights = group["총인구"].fillna(0)
    valid_weight_sum = weights.sum()

    for col in telecom_feature_cols:
        valid_mask = group[col].notna() & weights.notna()
        if valid_mask.any() and weights[valid_mask].sum() > 0:
            row[col] = np.average(group.loc[valid_mask, col], weights=weights[valid_mask])
        else:
            row[col] = np.nan

    dong_rows.append(row)

dong_df = pd.DataFrame(dong_rows)
dong_df = dong_df.merge(
    pop_df[["자치구", "행정동명_norm", "행정동총인구_공공"]],
    on=["자치구", "행정동명_norm"],
    how="left",
)

for col in interest_count_cols:
    dong_df[f"{col}비율"] = np.where(dong_df["총인구"] > 0, dong_df[col] / dong_df["총인구"], np.nan)

dong_df["공공인구매칭여부"] = dong_df["행정동총인구_공공"].notna()
dong_df["데이터총인구_공공인구차이"] = dong_df["총인구"] - dong_df["행정동총인구_공공"]
dong_df["데이터총인구_공공인구비"] = np.where(dong_df["행정동총인구_공공"] > 0, dong_df["총인구"] / dong_df["행정동총인구_공공"], np.nan)

print("행정동 통합 데이터 shape:", dong_df.shape)
display(dong_df.head(3))
"""
    ),
    code(
        """overview = pd.DataFrame({
    "dtype": dong_df.dtypes.astype(str),
    "결측치수": dong_df.isna().sum(),
    "결측치비율": dong_df.isna().mean(),
}).sort_values(["결측치수", "dtype"], ascending=[False, True])

print("행정동 수:", dong_df.shape[0])
print("공공 인구 매칭 수:", int(dong_df["공공인구매칭여부"].sum()))
print("공공 인구 미매칭 수:", int((~dong_df["공공인구매칭여부"]).sum()))

display(overview.head(20))
display(dong_df.loc[~dong_df["공공인구매칭여부"], ["자치구", "행정동명"]].sort_values(["자치구", "행정동명"]))
"""
    ),
    md(
        """## 3. 행정동 취약도/유형 지표 생성

관심집단 비율과 통신/이동 행태를 함께 반영한 복합지수를 만들고,  
행정동의 대표적인 특징이 무엇인지 유형화합니다.
"""
    ),
    code(
        """def zscore(series):
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0, index=series.index)
    return (series - series.mean()) / std

dong_df["1인가구수비율"] = np.where(dong_df["총인구"] > 0, dong_df["1인가구수"] / dong_df["총인구"], np.nan)

interest_risk_cols = [
    "1인가구수비율",
    "커뮤니케이션이 적은 집단비율",
    "외출이 매우 적은 집단(전체)비율",
    "재정상태에 대한 관심집단비율",
    "외출-커뮤니케이션이 모두 적은 집단(전체)비율",
]

telecom_high_risk_cols = [
    "최근 3개월 내 요금 연체 비율",
    "집 추정 위치 평일 총 체류시간",
    "집 추정 위치 휴일 총 체류시간",
]
telecom_low_risk_cols = [
    "평일 총 이동 횟수",
    "휴일 총 이동 횟수 평균",
    "평균 통화대상자 수",
    "평균 문자대상자 수",
]

for col in interest_risk_cols + telecom_high_risk_cols + telecom_low_risk_cols:
    if col in dong_df.columns:
        dong_df[f"{col}_z"] = zscore(dong_df[col])

interest_score_cols = [f"{col}_z" for col in interest_risk_cols if f"{col}_z" in dong_df.columns]
telecom_score_cols = [f"{col}_z" for col in telecom_high_risk_cols if f"{col}_z" in dong_df.columns]
telecom_inverse_score_cols = [f"{col}_z" for col in telecom_low_risk_cols if f"{col}_z" in dong_df.columns]

dong_df["관심집단취약지수"] = dong_df[interest_score_cols].mean(axis=1)
if telecom_score_cols or telecom_inverse_score_cols:
    dong_df["통신행동취약지수"] = pd.concat(
        [
            dong_df[telecom_score_cols],
            -dong_df[telecom_inverse_score_cols],
        ],
        axis=1,
    ).mean(axis=1)
else:
    dong_df["통신행동취약지수"] = np.nan

dong_df["행정동통합취약지수"] = dong_df[["관심집단취약지수", "통신행동취약지수"]].mean(axis=1)

dong_df["고립위험형점수"] = dong_df[[
    "커뮤니케이션이 적은 집단비율_z",
    "외출-커뮤니케이션이 모두 적은 집단(전체)비율_z",
    "평균 통화대상자 수_z",
    "평균 문자대상자 수_z",
]].assign(
    평균통화대상자_역=-dong_df["평균 통화대상자 수_z"],
    평균문자대상자_역=-dong_df["평균 문자대상자 수_z"],
).drop(columns=["평균 통화대상자 수_z", "평균 문자대상자 수_z"], errors="ignore").mean(axis=1)

dong_df["정주고착형점수"] = pd.concat(
    [
        dong_df[[c for c in ["외출이 매우 적은 집단(전체)비율_z", "집 추정 위치 평일 총 체류시간_z", "집 추정 위치 휴일 총 체류시간_z"] if c in dong_df.columns]],
        -dong_df[[c for c in ["평일 총 이동 횟수_z", "휴일 총 이동 횟수 평균_z"] if c in dong_df.columns]],
    ],
    axis=1,
).mean(axis=1)

dong_df["재정부담형점수"] = dong_df[[c for c in ["재정상태에 대한 관심집단비율_z", "최근 3개월 내 요금 연체 비율_z"] if c in dong_df.columns]].mean(axis=1)
dong_df["1인가구집중형점수"] = dong_df[[c for c in ["1인가구수비율_z"] if c in dong_df.columns]].mean(axis=1)

type_cols = ["고립위험형점수", "정주고착형점수", "재정부담형점수", "1인가구집중형점수"]
type_name_map = {
    "고립위험형점수": "고립위험형",
    "정주고착형점수": "정주고착형",
    "재정부담형점수": "재정부담형",
    "1인가구집중형점수": "1인가구집중형",
}
dong_df["대표유형"] = dong_df[type_cols].idxmax(axis=1).map(type_name_map)

display(
    dong_df[["자치구", "행정동명", "총인구", "관심집단취약지수", "통신행동취약지수", "행정동통합취약지수", "대표유형"]]
    .sort_values("행정동통합취약지수", ascending=False)
    .head(15)
)
"""
    ),
    md(
        """## 4. 행정동 중심 시각화와 비교

행정동 상위 위험 지역, 관심집단-통신행태 관계, 상관 구조를 함께 확인합니다.
"""
    ),
    code(
        """top_integrated = dong_df.sort_values("행정동통합취약지수", ascending=False).head(15).copy()

fig, axes = plt.subplots(1, 2, figsize=(26, 10))

sns.barplot(
    data=top_integrated.sort_values("행정동통합취약지수"),
    x="행정동통합취약지수",
    y="행정동명",
    hue="대표유형",
    dodge=False,
    ax=axes[0],
)
axes[0].set_title("행정동 통합취약지수 상위 15개")
axes[0].set_xlabel("통합취약지수")
axes[0].set_ylabel("행정동명")

scatter_df = dong_df.copy()
sns.scatterplot(
    data=scatter_df,
    x="1인가구수비율",
    y="외출-커뮤니케이션이 모두 적은 집단(전체)비율",
    hue="대표유형",
    size="총인구",
    sizes=(30, 280),
    alpha=0.75,
    ax=axes[1],
)
axes[1].set_title("1인가구 비율과 복합 저활성 비율")
axes[1].set_xlabel("1인가구 비율")
axes[1].set_ylabel("외출·커뮤니케이션 동시 저활성 비율")

for _, row in top_integrated.head(8).iterrows():
    axes[1].text(row["1인가구수비율"], row["외출-커뮤니케이션이 모두 적은 집단(전체)비율"], row["행정동명"], fontsize=9)

plt.tight_layout()
plt.show()
"""
    ),
    code(
        """corr_cols = [
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
    "금융 서비스 사용일수",
    "배달 서비스 사용일수",
]
corr_cols = [c for c in corr_cols if c in dong_df.columns]
corr_matrix = dong_df[corr_cols].corr()

plt.figure(figsize=(16, 13))
sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", center=0, fmt=".2f", square=True)
plt.title("행정동 기준 관심집단-통신행태 상관관계")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        """focus_ratio_cols = {
    "1인가구 비율": "1인가구수비율",
    "커뮤니케이션 저활성 비율": "커뮤니케이션이 적은 집단비율",
    "재정 관심 비율": "재정상태에 대한 관심집단비율",
    "복합 저활성 비율": "외출-커뮤니케이션이 모두 적은 집단(전체)비율",
    "요금 연체 비율": "최근 3개월 내 요금 연체 비율",
}

dong_top_tables = []
for label, col in focus_ratio_cols.items():
    if col not in dong_df.columns:
        continue
    top_row = dong_df.nlargest(1, col).iloc[0]
    dong_top_tables.append(
        {
            "지표": label,
            "자치구": top_row["자치구"],
            "행정동명": top_row["행정동명"],
            "총인구": int(top_row["총인구"]),
            "값": top_row[col],
            "대표유형": top_row["대표유형"],
        }
    )

pd.DataFrame(dong_top_tables)
"""
    ),
    md(
        """## 5. 자동 행정동 인사이트

세 CSV를 결합한 결과를 바탕으로, 행정동 중심의 핵심 포인트를 자동으로 정리합니다.
"""
    ),
    code(
        """top_integrated_row = dong_df.nlargest(1, "행정동통합취약지수").iloc[0]
top_single_row = dong_df.nlargest(1, "1인가구수비율").iloc[0]
top_overdue_row = dong_df.nlargest(1, "최근 3개월 내 요금 연체 비율").iloc[0]
top_home_row = dong_df.nlargest(1, "집 추정 위치 휴일 총 체류시간").iloc[0]
low_move_row = dong_df.nsmallest(1, "평일 총 이동 횟수").iloc[0]

high_risk_cut = dong_df["행정동통합취약지수"].quantile(0.90)
high_risk_dongs = dong_df[dong_df["행정동통합취약지수"] >= high_risk_cut].copy()
high_risk_gu = (
    high_risk_dongs.groupby("자치구")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="상위10% 행정동 수")
    .iloc[0]
)

type_dist = dong_df["대표유형"].value_counts().reset_index()
type_dist.columns = ["대표유형", "행정동 수"]

pop_gap_df = dong_df[dong_df["공공인구매칭여부"]].copy()
pop_gap_row = pop_gap_df.iloc[(pop_gap_df["데이터총인구_공공인구비"] - 1).abs().sort_values(ascending=False).index[0]]

insights = [
    f"1. 세 데이터 결합 기준 통합취약지수가 가장 높은 행정동은 {top_integrated_row['자치구']} {top_integrated_row['행정동명']}이며, 대표유형은 {top_integrated_row['대표유형']}입니다.",
    f"2. 1인가구 비율이 가장 높은 행정동은 {top_single_row['자치구']} {top_single_row['행정동명']}이며 비율은 {top_single_row['1인가구수비율']:.2%}입니다.",
    f"3. 최근 3개월 내 요금 연체 비율이 가장 높은 행정동은 {top_overdue_row['자치구']} {top_overdue_row['행정동명']}이며 비율은 {top_overdue_row['최근 3개월 내 요금 연체 비율']:.2%}입니다.",
    f"4. 휴일 기준 집 체류시간이 가장 긴 행정동은 {top_home_row['자치구']} {top_home_row['행정동명']}이며 평균 체류시간은 {top_home_row['집 추정 위치 휴일 총 체류시간']:,.1f}입니다.",
    f"5. 평일 총 이동 횟수가 가장 낮은 행정동은 {low_move_row['자치구']} {low_move_row['행정동명']}이며 평균 이동 횟수는 {low_move_row['평일 총 이동 횟수']:,.2f}입니다.",
    f"6. 통합취약지수 상위 10% 행정동이 가장 많이 분포한 자치구는 {high_risk_gu['자치구']}이며, 총 {int(high_risk_gu['상위10% 행정동 수'])}개 행정동이 포함됩니다.",
    f"7. 공공 인구가 매칭된 행정동 중 데이터 총인구와 공공 인구의 차이가 가장 큰 곳은 {pop_gap_row['자치구']} {pop_gap_row['행정동명']}이며, 비율은 {pop_gap_row['데이터총인구_공공인구비']:.2f}배입니다.",
    f"8. 전체 행정동에서 가장 많이 나타난 대표유형은 {type_dist.iloc[0]['대표유형']}이며 {int(type_dist.iloc[0]['행정동 수'])}개 행정동이 해당합니다.",
]

for line in insights:
    print(line)

display(type_dist)
display(high_risk_dongs[["자치구", "행정동명", "행정동통합취약지수", "대표유형"]].sort_values("행정동통합취약지수", ascending=False).head(20))
"""
    ),
    code(
        """dong_export_cols = [
    "행정동코드", "자치구", "행정동명", "총인구", "행정동총인구_공공", "데이터총인구_공공인구비",
    "1인가구수비율", "커뮤니케이션이 적은 집단비율", "재정상태에 대한 관심집단비율",
    "외출-커뮤니케이션이 모두 적은 집단(전체)비율", "최근 3개월 내 요금 연체 비율",
    "평일 총 이동 횟수", "휴일 총 이동 횟수 평균", "집 추정 위치 휴일 총 체류시간",
    "평균 통화대상자 수", "동영상/방송 서비스 사용일수", "금융 서비스 사용일수",
    "배달 서비스 사용일수", "관심집단취약지수", "통신행동취약지수", "행정동통합취약지수", "대표유형"
]
dong_export_cols = [c for c in dong_export_cols if c in dong_df.columns]
dong_df[dong_export_cols].sort_values("행정동통합취약지수", ascending=False).head(30)
"""
    ),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10.14",
    },
}

output_path = NOTEBOOK_OUTPUT_PATH
nbf.write(nb, output_path)
print(output_path)
