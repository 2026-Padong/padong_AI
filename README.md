# Padong_AI

`Padong_AI`는 FastAPI 기반의 백엔드 API 서버 스타터 템플릿입니다. 무거운 풀스택 구성이 아니라, 초기 개발과 확장에 적합한 깔끔한 실무형 구조를 목표로 구성했습니다.

## 프로젝트 구조

```text
Padong_AI/
├── app/
│   ├── main.py
│   ├── core/
│   ├── api/
│   ├── schemas/
│   ├── services/
│   ├── models/
│   ├── db/
│   └── utils/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 실행 방법

### 1. 가상환경 생성

macOS / Linux:

```bash
python3 -m venv .venv
```

Windows:

```powershell
python -m venv .venv
```

### 2. 가상환경 활성화

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 서버 실행

```bash
uvicorn app.main:app --reload
```

기본 실행 주소:

```text
http://127.0.0.1:8000
```

Swagger 문서 주소:

```text
http://127.0.0.1:8000/docs
```

## 테스트 실행

```bash
pytest
```

## 주요 엔드포인트

- `GET /api/v1/health`
