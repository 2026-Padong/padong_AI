# Nightly Training

매일 새벽 4시에 추천 학습 배치를 돌리는 기준 문서입니다.

## 실행 스크립트

- 학습 엔트리: [run_nightly_training.py](../../scripts/recommendation/run_nightly_training.py)
- cron 예시: [padong_ai_4am.cron](../../ops/cron/padong_ai_4am.cron)

`run_nightly_training.py`는 아래 순서로 실행됩니다.

1. 백엔드 로그 API에 전날 사용자 반응 로그를 요청
2. 응답으로 받은 `interactions`를 `user_recommendation_logs`에 upsert
3. `user_recommendation_logs`를 DB에서 읽어 pair dataset 생성
4. pair dataset으로 LightGBM 재학습
5. 학습 산출물을 `artifacts/dongne/`에 저장

## 백엔드 로그 연동 환경변수

새벽 배치에서 백엔드 로그를 pull 하려면 아래 환경변수를 주입합니다.

```text
BACKEND_LOG_SYNC_URL=https://backend.example.com/api/v1/recommendation/logs
BACKEND_LOG_SYNC_TOKEN=your-secret-token
BACKEND_LOG_SYNC_TIMEOUT_SEC=30
```

- `BACKEND_LOG_SYNC_URL`
  - 새벽 배치가 호출할 백엔드 로그 조회 API 주소
- `BACKEND_LOG_SYNC_TOKEN`
  - 필요할 때 Bearer 토큰으로 전달
- `BACKEND_LOG_SYNC_TIMEOUT_SEC`
  - 요청 타임아웃 초

배치는 아래 쿼리 파라미터를 붙여서 `GET` 요청합니다.

```text
date=2026-04-27
from=2026-04-27T00:00:00+09:00
to=2026-04-28T00:00:00+09:00
```

응답은 아래 둘 중 하나 형태면 됩니다.

```json
{
  "interactions": [
    {
      "user_id": 1,
      "created_at": "2026-04-27",
      "admin_dong_code": 1144066000,
      "clicked_count": 1,
      "liked_count": 0,
      "dwell_time_sec": 42.5,
      "impression_count": 1,
      "rank_position": 1,
      "q1": 5,
      "q2": 4,
      "q3": 3,
      "q4": 2,
      "q5": 1,
      "q6": 5,
      "q7": 4,
      "q8": 3,
      "q9": 2,
      "q10": 1
    }
  ]
}
```

또는

```json
[
  {
    "user_id": 1,
    "created_at": "2026-04-27",
    "admin_dong_code": 1144066000,
    "clicked_count": 1,
    "liked_count": 0,
    "dwell_time_sec": 42.5,
    "impression_count": 1,
    "rank_position": 1,
    "q1": 5,
    "q2": 4,
    "q3": 3,
    "q4": 2,
    "q5": 1,
    "q6": 5,
    "q7": 4,
    "q8": 3,
    "q9": 2,
    "q10": 1
  }
]
```

## Cron 등록 예시

서버에서 아래처럼 등록하면 매일 `04:00`에 실행됩니다.

```bash
crontab ops/cron/padong_ai_4am.cron
```

현재 cron 예시는 프로젝트 경로를 아래처럼 고정해두었습니다.

```text
/Users/yerin/Documents/project/padong_AI
```

다른 서버에 배포할 때는 이 경로를 실제 배포 경로로 바꿔야 합니다.

## Cron 식

```text
0 4 * * *
```

의미:

- 분: `0`
- 시: `4`
- 매일 실행

## 로그 파일

배치 실행 로그는 아래 파일에 append 됩니다.

```text
artifacts/dongne/nightly_training.log
```

## 수동 실행

배치를 바로 시험하려면:

```bash
source .venv/bin/activate
python scripts/recommendation/run_nightly_training.py
```
