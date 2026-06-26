# DACON / 제 5회 ETRI 휴먼이해 인공지능 논문경진대회
https://dacon.io/competitions/official/236690/overview/description

이 저장소는 DACON/ETRI “라이프로그 데이터를 활용한 수면, 감정, 스트레스 인식 및 추론” 대회용 최종 제출 패키지입니다. 스마트폰 및 웨어러블 lifelog sensor data를 사용해 수면, 감정, 스트레스 관련 7개 binary target의 확률을 예측합니다.

현재 저장소는 대회 종료 후 다음 목적에 맞게 정리되어 있습니다.

- 최종 `submission.csv` 재현
- 모델/피처 엔지니어링 구조 설명
- 최종 public/private 결과 및 private score 사후분석

> GitHub 공개본에서는 대회 제공 원본 데이터(`data/`), 최종 제출 CSV(`submission.csv`, `submissions/`), public probing 기록(`reports/`)을 포함하지 않습니다. 원본 데이터는 DACON 대회 페이지에서 별도로 받아야 하며, 제출/리포트 산출물은 로컬 실행 결과로만 관리합니다.

---

## 1. 최종 결과 요약

| 항목 | 값 |
| --- | ---: |
| Public logloss | `0.5726881984` |
| Public rank | `67` |
| Private logloss | `0.61533` |
| Private rank | `324` |

최종 public score는 개선되었지만, private score에서는 성능이 크게 악화되었습니다. 따라서 이 최종 파일은 “private 일반화가 잘 된 모델”이라기보다, **public leaderboard feedback으로 선택된 재현 가능한 최종 제출 아티팩트**로 해석해야 합니다.

Private 결과 기준 핵심 해석:

```text
public 44% score        = 0.5726881984
private 100% score      = 0.61533
implied hidden 56% score ≈ 0.64883
```

즉, hidden 56%에서 public 대비 약 `+0.076` 나빠졌고, public subset에 과하게 맞춰진 public-overfit이 확인되었습니다.

---

## 2. 대회 목표와 평가 방식

대회 목표는 lifelog sensor data를 이용해 7개 binary target을 예측하는 것입니다.

| Target | 의미 |
| --- | --- |
| `Q1` | 취침 후 수면의 질 |
| `Q2` | 취침 전 피로도 |
| `Q3` | 취침 전 스트레스 |
| `S1` | 총 수면시간 |
| `S2` | 수면효율 |
| `S3` | 수면 지연시간 |
| `S4` | 수면 중 각성 시간 |

제출 파일은 test 250행에 대해 위 target 7개 열에 `0~1` 확률을 채운 CSV입니다. Hard label이 아니라 확률을 제출해야 합니다.

평가 지표는 7개 target별 binary logloss의 평균입니다.

```text
score = mean(
  logloss(Q1), logloss(Q2), logloss(Q3),
  logloss(S1), logloss(S2), logloss(S3), logloss(S4)
)
```

Leaderboard 구조:

| 구간 | 설명 |
| --- | --- |
| Public Score | 전체 test 중 사전 샘플링된 44% |
| Private Score | 전체 test 100% |

---

## 3. 데이터 구조

현재 패키지의 데이터 구조는 다음과 같습니다.

```text
data/
├── ch2025_data_items/
│   ├── ch2025_mACStatus.parquet
│   ├── ch2025_mActivity.parquet
│   ├── ch2025_mAmbience.parquet
│   ├── ch2025_mBle.parquet
│   ├── ch2025_mGps.parquet
│   ├── ch2025_mLight.parquet
│   ├── ch2025_mScreenStatus.parquet
│   ├── ch2025_mUsageStats.parquet
│   ├── ch2025_mWifi.parquet
│   ├── ch2025_wHr.parquet
│   ├── ch2025_wLight.parquet
│   └── ch2025_wPedo.parquet
├── ch2026_metrics_description.pdf
├── ch2026_metrics_train.csv
└── ch2026_submission_sample.csv
```

주요 CSV:

| 파일 | 역할 |
| --- | --- |
| `data/ch2026_metrics_train.csv` | train 450일분 label. `subject_id`, `sleep_date`, `lifelog_date`, 7개 target 포함 |
| `data/ch2026_submission_sample.csv` | test 250행 제출 template. 행/열 순서 유지 필요 |
| `data/ch2026_metrics_description.pdf` | 7개 metric 설명 |

중요한 날짜 기준:

```text
sleep_date = lifelog_date 다음 날
sensor merge 기준 = lifelog_date
```

수면 관련 feature는 일반적인 0시~24시 하루 단위가 아니라 다음 구간을 하나의 sleep episode로 해석합니다.

```text
lifelog_date 18:00 -> sleep_date 12:00
```

---

## 4. Sensor 항목

| Sensor file | 주요 컬럼/내용 |
| --- | --- |
| `mACStatus` | `m_charging` |
| `mActivity` | `m_activity` |
| `mAmbience` | `list[[sound_label, probability]]` |
| `mBle` | `list[{address, device_class, rssi}]` |
| `mGps` | `list[{altitude, latitude, longitude, speed}]` |
| `mLight` | `m_light` |
| `mScreenStatus` | `m_screen_use` |
| `mUsageStats` | `list[{app_name, total_time}]` |
| `mWifi` | `list[{bssid, rssi}]` |
| `wHr` | `list[int]` heart rate |
| `wLight` | `w_light` |
| `wPedo` | `step`, `step_frequency`, `running_step`, `walking_step`, `distance`, `speed`, `burned_calories` |

---

## 5. 빠른 재현

최종 `submission.csv`는 `reproduce_final_submission.py`로 재현합니다.

```bash
python3 reproduce_final_submission.py --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```

성공 시 핵심 출력:

```text
reference_byte_equal: True
```

검증된 제출 형식:

```text
shape: (250, 10)
columns: data/ch2026_submission_sample.csv와 동일
row order: sample submission과 동일
```

`reports/final_reproduction_report.json`에는 최종 재현에 사용한 base/source 경로, S3 mean-lock 수치, target 평균, 확률 범위, reference 비교 결과가 저장됩니다.

---

## 6. 실행 환경

필수 패키지는 `requirements.txt`에 정의되어 있습니다.

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

`reproduce_final_submission.py`와 `solution.py`는 기본 데이터 경로로 `data`를 사용합니다. 다른 환경에서는 다음처럼 경로를 지정할 수 있습니다.

```bash
python3 reproduce_final_submission.py --data-dir /data --out submission.csv
DATA_DIR=/data python3 reproduce_final_submission.py --out submission.csv
python3 solution.py --data-dir /data --out submission.csv
```

---

## 7. 프로젝트 구조

```text
.
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── solution.py
├── reproduce_final_submission.py
├── submission.csv
├── data/
├── docs/
├── reports/
└── submissions/
```

주요 파일 역할:

| 경로 | 역할 |
| --- | --- |
| `submission.csv` | 최종 제출 파일. `submissions/final/final_selected.csv`와 동일 |
| `reproduce_final_submission.py` | 최종 제출 파일을 정확히 재현하는 스크립트 |
| `solution.py` | 원본 데이터에서 feature/model/submission 후보를 생성하는 전체 학습 코드 |
| `requirements.txt` | 실행 패키지 목록 |
| `PROJECT_STRUCTURE.md` | 정리된 프로젝트 구조 설명 |
| `data/` | 대회 CSV, PDF, parquet sensor data |
| `submissions/final/` | 최종 재현에 필요한 submission artifact 3개 |
| `reports/` | 최종 재현 검증 및 최종 선택 요약 |
| `docs/` | 대회 개요, 모델 설명, private 사후 분석 문서 |

---

## 8. 최종 submission artifact

`submissions/`는 최종 재현에 필요한 파일만 남겼습니다.

```text
submissions/final/
├── README.md
├── final_selected.csv
├── base_public_best_before_final.csv
└── stage2_source.csv
```

| 파일 | 설명 |
| --- | --- |
| `submissions/final/final_selected.csv` | 최종 선택 제출 파일. root `submission.csv`와 동일 |
| `submissions/final/base_public_best_before_final.csv` | 최종 S3 조정 전 public-best base |
| `submissions/final/stage2_source.csv` | 최종 S3 방향 계산에 사용한 stage2 prediction source |

최종 제출물은 base에서 `S3`만 stage2 방향으로 이동한 뒤, S3 평균을 base와 같게 mean-lock한 결과입니다.

```text
S3_raw = clip(base_S3 + 0.70 * (stage2_S3 - base_S3))
S3_final = sigmoid(logit(S3_raw) + intercept)
```

주요 수치:

| 항목 | 값 |
| --- | ---: |
| base S3 mean | `0.6729949309496949` |
| stage2 S3 mean | `0.6697581513742239` |
| raw toward-stage2 mean | `0.6707291852468651` |
| locked S3 mean | `0.6729949309496948` |
| logit intercept shift | `0.01273903376585006` |

최종 제출물은 base 대비 `S3`만 변경합니다.

---

## 9. 모델 파이프라인

`solution.py`는 train/test를 합쳐 `subject_id`, `lifelog_date` 기준 feature table을 만든 뒤, 7개 target을 각각 binary classifier로 학습합니다.

전체 흐름:

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

향후 `solution.py`를 다시 실행하면 보조 submission variant는 `submissions/generated/`에 생성됩니다. 최종 보관용 파일은 `submissions/final/`에 유지됩니다.

---

## 10. Feature Engineering 요약

| 그룹 | 내용 |
| --- | --- |
| calendar | dayofweek, month, elapsed day, subject sequence |
| scalar sensors | count, mean, std, min, max, median, q25/q75, sum, availability |
| time windows | dawn, morning, afternoon, evening |
| activity/status | charging, activity, screen-use 값별 count/proportion |
| pedometer | step/distance/calories/speed, active minutes, zero ratio |
| heart rate | list explode 후 mean/std/min/max/quantile, night/evening statistics |
| light | dark/bright ratio, evening light statistics |
| GPS | speed/altitude stats, moving/stationary ratio, location-cell count |
| Wifi/BLE | detection count, unique id count, RSSI stats, strong signal ratio |
| ambience | top sound label/probability, entropy, label별 probability stats |
| usage | app count, total time stats, top app usage |
| sleep block | `lifelog_date 18:00 -> sleep_date 12:00` 구간의 sleep episode proxy |

수면 episode는 5분 bin으로 나눈 뒤 screen-off, no-step, still activity, darkness, low heart rate, no app usage 등을 조합해 sleep block proxy를 만듭니다.

---

## 11. 모델과 검증

Target별 base model family:

| 모델 | 역할 |
| --- | --- |
| LogisticRegression | 선형 baseline, calibration 안정성 |
| ExtraTreesClassifier | 비선형 tree ensemble |
| RandomForestClassifier | bagging 계열 다양성 |
| HistGradientBoostingClassifier | compact boosting |
| LightGBM | 설치된 경우 optional 사용 |

Validation:

- StratifiedKFold OOF logloss
- GroupKFold by `subject_id` diagnostic
- blocked-time split diagnostic
- target별 logloss 및 7-target 평균 logloss 출력

Leakage 방지:

- subject target mean은 OOF / leave-one-out 방식으로 계산
- validation fold의 label은 validation feature에 직접 들어가지 않음
- test placeholder label은 학습에 사용하지 않음

---

## 12. Private score 사후 분석 요약

Private 결과는 최종 후보가 public subset에 과하게 맞춰졌음을 보여줍니다.

점수 분해:

```text
private = 0.44 * public + 0.56 * hidden
hidden  ≈ (0.61533 - 0.44 * 0.5726881984) / 0.56
        ≈ 0.64883
```

주요 원인:

1. 학습 모델 자체의 일반화 개선보다 public feedback 기반 target-wise 후처리가 누적됨
2. mean-lock/prior 보정 기준이 hidden label 분포와 맞지 않았을 가능성
3. 일부 확률이 과신되어 hidden 오답에서 logloss penalty가 커졌을 가능성
4. 내부 CV보다 public score를 후보 선택 기준으로 강하게 사용함
5. 같은 base 위의 후처리 변형이 많아 후보 간 error correlation이 높았음

정확한 해석:

```text
public leaderboard feedback으로 선택된 최종 제출 아티팩트.
재현은 가능하지만, private 결과상 public-overfit이 확인된 파일.
```

---

## 13. 문서 구조

| 문서 | 내용 |
| --- | --- |
| `docs/README.md` | 문서 읽는 순서 |
| `docs/competition_overview.md` | 대회 목표, 평가 방식, 데이터 구조, 제출 규칙 |
| `docs/model_and_reproducibility.md` | 최종 제출물 재현 방법과 모델 구조 |
| `docs/private_score_postmortem.md` | private score 악화 및 public-overfit 원인 분석 |

README에 핵심 내용을 통합했지만, 세부 확인이 필요하면 위 문서를 참고하면 됩니다.

---

## 14. 주의사항

- `sleep_date`는 `lifelog_date`의 다음 날입니다.
- 센서 로그는 `sleep_date`가 아니라 `lifelog_date` 기준으로 merge합니다.
- 제출 파일의 행/열 순서는 `data/ch2026_submission_sample.csv`와 동일해야 합니다.
- target 값은 hard label이 아니라 확률입니다.
- 최종 `submission.csv`는 재현 가능하지만, private 결과상 일반화 성능이 좋은 모델로 해석하면 안 됩니다.
- `solution.py`는 전체 모델 구현 근거 코드이고, 최종 제출 CSV의 정확한 재현은 `reproduce_final_submission.py`가 담당합니다.
