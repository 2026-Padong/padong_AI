# 로그 스키마

학습 로그의 기본 단위는 `사용자 x 추천된 행정동 1개`입니다.  
즉 추천 결과로 20개 행정동을 노출했다면, 같은 `session_id` 아래 20행이 저장되어야 합니다.

## 최소 저장 컬럼

- `user_id`: 사용자 식별자
- `session_id`: 추천 1회 단위 식별자
- `event_at`: 추천 로그 시각
- `admin_dong_code`: 추천된 행정동 key
- `rank_position`: 추천 순위, 1~20
- `impression`: 노출 여부, 기본값 `1`
- `clicked`: 클릭 여부, `0/1`
- `liked`: 좋아요 여부, `0/1`
- `dwell_time_sec`: 체류시간(초)
- `q1` ~ `q10`: 설문 응답값, 각 `1~5`

## 권장 규칙

- `top_k = 20`으로 추천한 결과는 20행 모두 저장
- 반응이 없어도 `clicked=0`, `liked=0`, `dwell_time_sec=0`으로 남김
- `admin_dong_code`는 [new*서울시*행정동.csv](new_서울시_행정동.csv) 기준 key 사용

## 예시 CSV 헤더

```csv
user_id,session_id,event_at,admin_dong_code,rank_position,impression,clicked,liked,dwell_time_sec,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
```

## label 생성 규칙

현재 1차 모델에서는 아래 규칙으로 label을 만듭니다.

```text
label = 0.2 * clicked + 0.6 * liked + 0.2 * min(dwell_time_sec / 120, 1.0)
```

- 좋아요 신호를 가장 강하게 반영
- 클릭은 약한 선호 신호
- 체류시간은 보조 신호

## 처리 순서

1. 백엔드가 추천 결과 `admin_dong_code` 20개를 프론트에 전달
2. 프론트/백엔드가 행정동별 반응 로그 저장
3. [build_pair_dataset.py](build_pair_dataset.py)로 pair dataset 생성
4. [train_lgbm_regressor.py](train_lgbm_regressor.py)로 LightGBM Regressor 학습
