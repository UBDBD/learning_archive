# DACON / 제 5회 ETRI 휴먼이해 인공지능 논문경진대회

https://dacon.io/competitions/official/236690/overview/description

이 저장소는 DACON/ETRI “라이프로그 데이터를 활용한 수면, 감정, 스트레스 인식 및 추론” 대회용 정리 패키지다. 스마트폰 및 웨어러블 lifelog sensor data를 사용해 수면, 감정, 스트레스 관련 7개 binary target의 확률을 예측한다.

현재 구조는 다음 목적에 맞게 정리되어 있다.

- 최종 제출 파일 재현 및 형식 검증
- 학습/추론 코드와 feature engineering 구조 보존
- public/private 결과와 public-overfit 사후 분석 기록
- GitHub 공개용 코드/문서와 로컬 보관용 데이터/산출물 분리

---

## 1. 최종 결과

| 항목 | 값 |
| --- | ---: |
| Public logloss | `0.5726881984` |
| Public rank | `67` |
| Private logloss | `0.61533` |
| Private rank | `324` |

최종 public score는 개선되었지만, private score에서는 성능이 크게 악화되었다. 이 파일은 “private 일반화가 잘 된 모델”이라기보다, **public leaderboard feedback으로 선택된 재현 가능한 최종 제출 아티팩트**로 해석해야 한다.

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

과거 S3 mean-lock 조정 과정을 재실행하려면 `base_public_best_before_final.csv`와 `stage2_source.csv`가 필요하다. 현재 정리된 로컬 패키지는 중복 후보 파일을 제거하고 최종 제출 CSV 하나만 보관한다.

---

## 3. 실행 환경

필수 패키지는 `requirements.txt`에 정의되어 있다.

```text
pandas>=2.0
numpy>=1.22
pyarrow>=12.0
scikit-learn>=1.3
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
python3 solution.py --data-dir /data --out submission.csv
```

---

## 4. 프로젝트 구조

```text
.
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── solution.py
├── reproduce_final_submission.py
├── data/                         # 로컬 보관. GitHub 공개 제외
├── submissions/final/            # 로컬 보관. 최종 제출 CSV
├── reports/                      # 로컬 보관. 검증/선택 리포트
└── docs/                         # 세부 설명 문서
```

주요 파일:

| 경로 | 역할 |
| --- | --- |
| `reproduce_final_submission.py` | 최종 제출 파일을 재생성하고 sample submission 구조를 검증하는 스크립트 |
| `solution.py` | 원본 데이터에서 feature/model/submission 후보를 생성하는 전체 학습 코드 |
| `submissions/final/final_submission.csv` | 최종 선택 제출 파일. 로컬 보관용이며 GitHub 공개 제외 |
| `reports/final_reproduction_report.json` | 재현 검증 결과. 로컬 보관용이며 GitHub 공개 제외 |
| `PROJECT_STRUCTURE.md` | 공개/로컬 파일 구조와 보관 기준 |
| `docs/competition_overview.md` | 대회 목표, 평가 방식, 데이터 구조 |
| `docs/model_and_reproducibility.md` | 최종 artifact 재현 방식과 모델 구조 |
| `docs/private_score_postmortem.md` | private score 악화 및 public-overfit 원인 분석 |

---

## 5. 모델 파이프라인 요약

`solution.py`는 train/test를 합쳐 `subject_id`, `lifelog_date` 기준 feature table을 만든 뒤, 7개 target을 각각 binary classifier로 학습한다.

```text
train/sample CSV 로드
-> train/test feature table 결합
-> lifelog_date 기준 sensor parquet 집계
-> sleep episode / sleep block feature 생성
-> subject/time/calendar feature 생성
-> target별 binary classifier 학습
-> OOF logloss 검증
-> final model fit on full train
-> submission variant 저장
```

주요 feature group은 calendar, scalar sensor 집계, time window, activity/status, pedometer, heart rate, light, GPS, Wifi/BLE, ambience, app usage, sleep block proxy다.

Target별 base model family:

- LogisticRegression
- ExtraTreesClassifier
- RandomForestClassifier
- HistGradientBoostingClassifier
- LightGBM optional

---

## 6. 중요 기준

- `sleep_date`는 `lifelog_date`의 다음 날이다.
- 센서 로그는 `sleep_date`가 아니라 `lifelog_date` 기준으로 merge한다.
- 수면 episode proxy는 `lifelog_date 18:00 -> sleep_date 12:00` 구간을 사용한다.
- 제출 파일의 행/열 순서는 `data/ch2026_submission_sample.csv`와 동일해야 한다.
- target 값은 hard label이 아니라 `0~1` 확률이다.
- `data/`, `submissions/`, `reports/`, parquet/PDF/CSV 제출 산출물은 공개 저장소에 올리지 않는다.
