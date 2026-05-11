# 로그 스키마

학습 로그의 기본 단위는 `사용자 x 추천된 행정동 1개`입니다.  
즉 추천 결과로 20개 행정동을 노출했다면, 백엔드가 행정동별 반응 로그 20행을 저장하고 AI 배치가 이를 동기화합니다.

## 최소 저장 컬럼

- `user_id`: 사용자 식별자
- `created_at`: 로그 기준일, `YYYY-MM-DD`
- `admin_dong_code`: 추천된 행정동 key
- `rank_position`: 추천 순위, 1~20
- `impression_count`: 하루 기준 노출 횟수
- `clicked_count`: 하루 기준 클릭 횟수
- `liked_count`: 하루 기준 좋아요 횟수
- `dwell_time_sec`: 체류시간(초)
- `q1` ~ `q10`: 설문 응답값, 각 `1~5`

## 권장 규칙

- `top_k = 20`으로 추천한 결과는 20행 모두 저장
- 반응이 없어도 `clicked_count=0`, `liked_count=0`, `dwell_time_sec=0`으로 남김
- `admin_dong_code`는 [new*서울시*행정동.csv](new_서울시_행정동.csv) 기준 key 사용

## 예시 CSV 헤더

```csv
user_id,created_at,admin_dong_code,rank_position,impression_count,clicked_count,liked_count,dwell_time_sec,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
```

## label 생성 규칙

현재 1차 모델에서는 아래 규칙으로 label을 만듭니다.

```text
label = 0.2 * min(clicked_count / 5, 1.0) + 0.6 * min(liked_count / 3, 1.0) + 0.2 * min(dwell_time_sec / 120, 1.0)
```

- 좋아요 신호를 가장 강하게 반영
- 클릭은 약한 선호 신호
- 체류시간은 보조 신호

## 처리 순서

1. 백엔드가 추천 결과 `admin_dong_code` 20개를 프론트에 전달
2. 프론트/백엔드가 행정동별 반응 로그를 백엔드 DB에 저장
3. 매일 새벽 4시에 AI 배치가 백엔드 로그를 동기화하고 [build_pair_dataset.py](../../scripts/recommendation/build_pair_dataset.py)로 pair dataset 생성
4. 이어서 [train_lgbm_regressor.py](../../scripts/recommendation/train_lgbm_regressor.py) 또는 [run_nightly_training.py](../../scripts/recommendation/run_nightly_training.py)로 학습
5. 운영 등록 방식은 [nightly_training.md](./nightly_training.md)와 [padong_ai_4am.cron](../../ops/cron/padong_ai_4am.cron) 참고
