# Seoul Neighborhood Recommender

실행 스크립트와 데이터 전처리 코드는 `scripts/`로 이동했고, 이 디렉터리에는 EDA 노트북과 분석 중심 문서만 남깁니다.

- 추천/학습 스크립트: `scripts/recommendation/`
- 데이터 전처리 스크립트: `scripts/data_prep/`
- 원천/가공 CSV: `data/dongne/raw`, `data/dongne/processed`

서울의 `구`와 `행정동`을 사용자 설문 응답 기반으로 추천하는 Python 추천 로직입니다.

이 프로젝트는 아래 3개의 CSV를 결합해 만든 행정동 단위 특성 데이터를 바탕으로 동작합니다.

- `2025.12월_10개 관심집단수.csv`
- `2025.12월_29개 통신정보.csv`
- `행정구역_읍면동_별_5세별_인구.csv`

핵심 목적은 단순 인기 지역 추천이 아니라, 사용자의 생활 패턴과 선호에 맞는 서울 동네를 찾는 것입니다.

예를 들어 아래 같은 질문에 답하면, 그 응답을 점수화해:

- 어떤 구가 먼저 맞는지
- 그 구 안에서 어떤 행정동이 더 잘 맞는지
- 왜 그 동네가 맞는지

를 설명과 함께 반환합니다.

---

## 1. What This Does

`seoul_neighborhood_recommender.py`는 다음 흐름으로 동작합니다.

1. 사용자 설문 응답 10개를 입력받습니다.
2. 각 응답을 `1~5점`에서 `-1 ~ +1`로 정규화합니다.
3. 응답을 여러 생활 성향 축으로 변환합니다.
4. 서울의 구/동 프로필 데이터와 매칭합니다.
5. 구 추천과 동 추천을 계산합니다.
6. 추천 이유와 주의 포인트를 설명문으로 생성합니다.

---

## 2. Files

주요 파일은 아래와 같습니다.

- [seoul_neighborhood_recommender.py](/C:/Users/82102/Desktop/파동/seoul_neighborhood_recommender.py)
  서울 구/동 추천 함수가 들어있는 핵심 모듈

- [seoul_neighborhood_recommender_logic.md](/C:/Users/82102/Desktop/파동/seoul_neighborhood_recommender_logic.md)
  추천 로직 설계 문서

- [2025_12_3csv_dong_integrated_eda.ipynb](/C:/Users/82102/Desktop/파동/2025_12_3csv_dong_integrated_eda.ipynb)
  세 CSV를 취합한 통합 EDA 노트북

- [summarize_seoul_features.py](/C:/Users/82102/Desktop/파동/summarize_seoul_features.py)
  서울 구/동 특징 요약용 스크립트

- [build_dong_integrated_eda.py](/C:/Users/82102/Desktop/파동/build_dong_integrated_eda.py)
  통합 EDA 노트북 생성 스크립트

---

## 3. Input Format

추천 함수는 `q1 ~ q10` 형식의 딕셔너리를 입력으로 받습니다.

각 값은 `1 ~ 5` 범위의 숫자여야 합니다.

- `1`: 전혀 아니다
- `2`: 아니다
- `3`: 보통이다
- `4`: 그렇다
- `5`: 매우 그렇다

### 고정 질문 10개

- `q1`: 혼자 사는 사람 비중이 높은 동네가 더 편한가요?
- `q2`: 집에서 보내는 시간이 많은 생활 패턴인가요?
- `q3`: 동네 안에서 사람들과 자연스럽게 연결되는 분위기를 원하나요?
- `q4`: 출퇴근이나 평일 이동이 적은 생활권을 선호하나요?
- `q5`: 주말에도 멀리 나가기보다 집 근처 생활을 선호하나요?
- `q6`: 배달, 쇼핑, 생활서비스를 자주 이용하나요?
- `q7`: 활발한 상권형 동네보다 안정적인 주거형 동네를 원하나요?
- `q8`: 생활비나 고정비 부담이 상대적으로 덜한 동네가 중요하나요?
- `q9`: 청년 1인가구가 많은 동네 분위기가 더 잘 맞나요?
- `q10`: 혼자 지내기 편한 동네와 사람들과 섞이기 쉬운 동네 중 어느 쪽이 더 중요한가요?

### Example Input

```python
responses = {
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
```

---

## 4. Normalization

입력 응답은 내부에서 아래 공식으로 정규화됩니다.

```text
normalized = (response - 3) / 2
```

정규화 결과는 다음과 같습니다.

- `1 -> -1.0`
- `2 -> -0.5`
- `3 -> 0.0`
- `4 -> 0.5`
- `5 -> 1.0`

이 방식 덕분에 `강한 비선호`와 `강한 선호`를 대칭 구조로 계산할 수 있습니다.

---

## 5. User Vector

입력된 10개 응답은 아래 사용자 성향 축으로 변환됩니다.

### 1. `single_household_affinity`

1인가구가 많은 동네를 얼마나 편하게 느끼는지

```text
0.6 * q1 + 0.4 * q9
```

### 2. `settled_home_life`

집 중심, 정주형 생활 패턴 선호

```text
0.5 * q2 + 0.3 * q5 + 0.2 * q7
```

### 3. `social_connection_preference`

사람들과 연결되는 분위기 선호

```text
1.0 * q3 - 0.7 * q10
```

### 4. `low_mobility_preference`

이동이 적은 생활권 선호

```text
0.7 * q4 + 0.3 * q5
```

### 5. `lifestyle_service_dependence`

배달/쇼핑/생활서비스 활용 성향

```text
1.0 * q6
```

### 6. `residential_stability_preference`

상권 중심보다 주거 안정형 선호

```text
0.8 * q7 + 0.2 * q2
```

### 7. `cost_sensitivity`

비용 부담에 대한 민감도

```text
1.0 * q8
```

### 8. `youth_mobile_preference`

청년/1인가구/유동형 생활권 선호

```text
0.5 * q1 + 0.5 * q9 - 0.4 * q7
```

---

## 6. Region Profiles

서울 각 `구`와 `행정동`은 데이터 기반으로 여러 프로필 축을 가집니다.

주요 기반 지표는 아래와 같습니다.

- `1인가구수비율`
- `커뮤니케이션이 적은 집단비율`
- `외출이 매우 적은 집단(전체)비율`
- `재정상태에 대한 관심집단비율`
- `외출-커뮤니케이션이 모두 적은 집단(전체)비율`
- `최근 3개월 내 요금 연체 비율`
- `평일 총 이동 횟수`
- `휴일 총 이동 횟수 평균`
- `집 추정 위치 휴일 총 체류시간`
- `평균 통화대상자 수`
- `평균 문자대상자 수`
- `동영상/방송 서비스 사용일수`
- `쇼핑 서비스 사용일수`
- `배달 서비스 사용일수`
- `금융 서비스 사용일수`

이 값들을 z-score로 표준화한 뒤, 다음 프로필을 만듭니다.

### `single_household_profile`

1인가구 친화적인 지역인지

### `settled_profile`

집 중심, 정주형 생활 패턴이 강한 지역인지

### `social_profile`

교류와 연결감이 비교적 살아 있는 지역인지

### `service_profile`

배달/쇼핑/콘텐츠 서비스 활용 밀도가 높은 지역인지

### `cost_risk_profile`

비용 부담/재정 민감 신호가 높은 지역인지

### `isolation_risk_profile`

고립 위험이 상대적으로 높은 지역인지

### `youth_mobile_profile`

청년/유동형 생활권 성격이 강한지

### `residential_stability_profile`

주거 안정성이 강한지

---

## 7. Matching Logic

사용자 벡터와 지역 프로필을 가중합으로 비교합니다.

현재 구현된 적합도 점수는 아래 구조입니다.

```text
0.18 * (single_household_affinity * single_household_profile)
+ 0.16 * (settled_home_life * settled_profile)
+ 0.14 * (social_connection_preference * social_profile)
+ 0.14 * (low_mobility_preference * settled_profile)
+ 0.12 * (lifestyle_service_dependence * service_profile)
+ 0.10 * (residential_stability_preference * residential_stability_profile)
+ 0.08 * (youth_mobile_preference * youth_mobile_profile)
- 0.08 * (cost_sensitivity * cost_risk_profile)
- 0.06 * (social_connection_preference * isolation_risk_profile)
```

해석은 단순합니다.

- 점수가 높을수록 사용자와 잘 맞는 지역
- 비용 민감한 사용자는 비용 부담 신호가 높은 지역에서 점수가 깎임
- 교류를 중시하는 사용자는 고립 위험이 높은 지역에서 점수가 깎임

---

## 8. Recommendation Flow

추천은 아래 순서로 나옵니다.

### 1단계. 구 추천

서울의 각 구에 대해 `match_score`를 계산합니다.

기본적으로 상위 `3개 구`를 뽑습니다.

### 2단계. 동 추천

상위 구 안의 행정동들만 다시 계산합니다.

기본적으로 `구당 3개 동`을 추천합니다.

또한 인구가 너무 적어 왜곡될 수 있는 동은 제외합니다.

기본 설정:

- `min_population = 1000`

---

## 9. Description Generation

추천 결과에는 설명문이 포함됩니다.

설명문은 아래 요소를 조합해서 만듭니다.

- 이 지역의 대표유형
- 상위 프로필 특성
- 사용자 선호와 맞는 이유
- 주의 포인트

### Example Description

```text
관악구 신림동은 1인가구집중형 성격이 강한 동네입니다.
이곳은 1인가구 친화성이 높고 생활서비스 활용성이 높고 청년·유동형 생활 패턴과 잘 맞는 특징이 함께 나타나,
청년층·단독가구·서비스 활용도가 높은 사용자에게 특히 잘 맞을 수 있습니다.
다만 비용 부담 신호가 상대적으로 높게 나타나면 생활비 민감 사용자에게는 주의가 필요합니다.
```

---

## 10. How To Use

### Basic Usage

```python
from pathlib import Path
from seoul_neighborhood_recommender import recommend_seoul_neighborhoods

responses = {
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

result = recommend_seoul_neighborhoods(
    responses,
    base_dir=Path(r"C:\Users\82102\Desktop\파동")
)

print(result["summary"])
print(result["gu_recommendations"])
print(result["dong_recommendations"])
```

### Run As Script

아래처럼 직접 실행하면 샘플 입력으로 결과를 확인할 수 있습니다.

```powershell
python C:\Users\82102\Desktop\파동\seoul_neighborhood_recommender.py
```

---

## 11. Return Structure

`recommend_seoul_neighborhoods(...)`는 딕셔너리를 반환합니다.

### Top-level keys

- `question_text`
- `normalized_responses`
- `user_vector`
- `gu_recommendations`
- `dong_recommendations`
- `summary`

### Example Shape

```python
{
    "question_text": {...},
    "normalized_responses": {
        "q1": 1.0,
        "q2": 0.5,
        ...
    },
    "user_vector": {
        "single_household_affinity": 0.8,
        "settled_home_life": 0.35,
        ...
    },
    "gu_recommendations": [
        {
            "gu": "관악구",
            "match_score": 1.1305,
            "description": "...",
            "top_traits": [...],
            "caution": "..."
        }
    ],
    "dong_recommendations": [
        {
            "gu": "관악구",
            "dong": "신림동",
            "match_score": 1.6691,
            "population": 12345,
            "type": "1인가구집중형",
            "description": "...",
            "top_traits": [...],
            "caution": "..."
        }
    ],
    "summary": "..."
}
```

---

## 12. Main Functions

### `validate_responses(responses)`

응답이 모두 있는지, `q1~q10`이 들어왔는지, 값이 `1~5` 범위인지 검증합니다.

### `build_user_vector(responses)`

응답을 정규화하고 사용자 생활 성향 벡터를 생성합니다.

### `load_region_profiles(base_dir)`

CSV를 읽어 구/동 프로필 데이터프레임을 만듭니다.

이 함수는 `lru_cache`가 적용되어 있어 같은 경로에서 반복 호출 시 재사용됩니다.

### `recommend_seoul_neighborhoods(responses, base_dir, config)`

실제 추천 실행 함수입니다.

---

## 13. Config

`RecommenderConfig`로 추천 개수와 최소 인구 기준을 조정할 수 있습니다.

```python
from seoul_neighborhood_recommender import RecommenderConfig

config = RecommenderConfig(
    top_gu=3,
    top_dong_per_gu=3,
    min_population=1000,
)
```

### Fields

- `top_gu`
  추천할 구 개수

- `top_dong_per_gu`
  각 구에서 추천할 동 개수

- `min_population`
  추천 대상에서 제외할 최소 총인구 기준

---

## 14. Data Assumptions

현재 로직은 아래를 가정합니다.

- 질문은 항상 `q1 ~ q10` 고정
- 응답은 항상 `1~5`
- CSV 파일 3개는 같은 폴더에 존재
- 관심집단수와 통신정보는 `행정동코드-성별-연령대` 기준으로 결합 가능
- 추천은 `행정동` 중심이지만 먼저 `구` 단위로 좁힌 뒤 추천

---

## 15. Notes

### 공공 인구 파일

현재 추천 함수 자체는 공공 인구 파일을 직접 사용하지 않습니다.  
추천의 핵심은 관심집단수와 통신정보 병합 데이터이며, 공공 인구 파일은 통합 EDA 단계에서 행정동 기준 검증용으로 활용되었습니다.

### 설명문 성격

설명문은 데이터 기반 요약문입니다.  
정책적 판단이나 부동산 가치 판단이 아니라, 생활 패턴 적합도를 설명하는 문장으로 해석해야 합니다.

### 점수 해석

`match_score`는 절대 점수라기보다 상대 점수입니다.

- 같은 사용자 안에서 점수가 높은 지역끼리 비교하는 용도
- 서로 다른 사용자 간 점수를 직접 비교하는 용도는 아님

---

## 16. Example Output

샘플 응답 기준 실제 출력 예시는 아래와 비슷합니다.

```text
당신은 1인가구 친화성이 높은 편이고 생활서비스 활용도가 높은 편이고 청년·유동형 분위기를 선호하는 편인 성향이 두드러집니다.
그래서 서울 안에서는 관악구, 광진구, 중구 쪽이 먼저 추천되며,
세부적으로는 관악구 신림동, 관악구 대학동, 관악구 서원동, 광진구 화양동, 광진구 능동 같은 동네가 잘 맞을 가능성이 높습니다.
```

---

## 17. Next Steps

이 README 기준으로 다음 단계 확장이 가능합니다.

- FastAPI/Flask API로 감싸기
- Streamlit 설문 웹앱 만들기
- JSON 응답 포맷으로 프론트엔드 연동
- 추천 결과를 CSV/DB에 저장
- 질문 수를 늘린 고급 버전 만들기

---

## 18. Quick Start

가장 빠르게 확인하려면:

1. CSV 3개와 `seoul_neighborhood_recommender.py`를 같은 폴더에 둡니다.
2. `responses` 딕셔너리를 준비합니다.
3. `recommend_seoul_neighborhoods(...)`를 호출합니다.
4. `summary`, `gu_recommendations`, `dong_recommendations`를 확인합니다.

---

## 19. Contact Between Logic And Data

이 로직은 아래 성격의 서비스에 잘 맞습니다.

- 서울 동네 추천 설문
- 사용자 생활 패턴 기반 지역 추천
- 구/동 매칭형 로컬 추천
- 거주 후보지 탐색 보조 도구

반대로 아래 목적에는 그대로 쓰기 어렵습니다.

- 부동산 가격 예측
- 투자 가치 예측
- 안전도 공식 평가
- 정책 우선순위 판단
