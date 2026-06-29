# DACON / 제 5회 ETRI 휴먼이해 인공지능 논문경진대회

https://dacon.io/competitions/official/236690/overview/description

이 저장소는 DACON/ETRI “라이프로그 데이터를 활용한 수면, 감정, 스트레스 인식 및 추론” 대회의 최종 결과와 재현 절차만 남긴 정리본이다. 공개 범위는 최종 제출 파일을 검증·재생성하는 코드와 결과 해석 문서로 제한한다.

---

## 1. 최종 결과

| 항목 | 값 |
| --- | ---: |
| Public logloss | `0.5726881984` |
| Public rank | `67` |
| Private logloss | `0.61533` |
| Private rank | `324` |

최종 public score는 개선되었지만, private score에서는 성능이 크게 악화되었다. 따라서 이 결과는 “private 일반화가 잘 된 모델”이 아니라, **public leaderboard feedback으로 선택된 재현 가능한 최종 제출 아티팩트**로 해석해야 한다.

점수 분해:

```text
public 44% score        = 0.5726881984
private 100% score      = 0.61533
implied hidden 56% score ≈ 0.64883
```

---

## 2. 빠른 재현

현재 로컬 패키지는 최종 보관 파일 `submissions/final/final_submission.csv`를 기준으로 `submission.csv`를 재생성하고, sample submission과 행/열 구조를 검증한다.

```bash
python3 reproduce_final_submission.py --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```

성공 시 핵심 출력:

```text
artifact_byte_equal: True
```

검증 항목:

- shape: `(250, 10)`
- columns: `data/ch2026_submission_sample.csv`와 동일
- row order: sample submission과 동일
- probability range: `[0, 1]`

---

## 3. 실행 환경

필수 패키지는 `requirements.txt`에 정의되어 있다.

```text
pandas>=2.0
```

권장 환경 구성:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

기본 데이터 경로는 `data`이며, 다른 환경에서는 다음처럼 지정할 수 있다.

```bash
python3 reproduce_final_submission.py --data-dir /data --out submission.csv
DATA_DIR=/data python3 reproduce_final_submission.py --out submission.csv
```

---

## 4. 공개 파일 구조

```text
.
├── README.md
├── requirements.txt
├── reproduce_final_submission.py
└── docs/
    ├── competition_overview.md
    ├── model_and_reproducibility.md
    └── RESULT_ANALYSIS.md
```

주요 파일:

| 경로 | 역할 |
| --- | --- |
| `reproduce_final_submission.py` | 최종 제출 파일을 재생성하고 sample submission 구조를 검증하는 스크립트 |
| `requirements.txt` | 재현 스크립트 실행에 필요한 최소 패키지 |
| `docs/competition_overview.md` | 대회 목표, 평가 방식, 데이터 구조 |
| `docs/model_and_reproducibility.md` | 최종 artifact 재현 방식과 재현 범위 |
| `docs/RESULT_ANALYSIS.md` | private score 악화 및 public-overfit 원인 분석 |

---

## 5. 공개 제외 항목

다음 항목은 로컬 보관용이며 공개 저장소에 올리지 않는다.

- `data/`: 대회 제공 CSV/PDF/parquet sensor data
- `submissions/`: 최종 제출 CSV와 후보 제출물
- `reports/`: 재현 검증 리포트와 public-probe 기록
- `submission.csv`: 재생성된 제출 파일
- 탐색용 학습 파이프라인 코드

정리 기준은 최종 결과 검증과 재현성이다. 최종 제출 CSV를 직접 재생성·검증하는 코드만 남기고, public leaderboard 탐색과 모델 후보 생성을 위한 학습 코드는 공개 범위에서 제거했다.
