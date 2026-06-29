# 모델 및 재현성 가이드

정리 기준일: 2026-06-29  
최종 제출 파일: `submission.csv`  
최종 보관 파일: `submissions/final/final_submission.csv`  
최종 public score: `0.5726881984`  
최종 private score: `0.61533`  

이 문서는 현재 정리된 패키지에서 실제로 동작하는 재현 방식과 모델 구조를 설명한다.

---

## 1. 최종 결과 재현

프로젝트 루트에서 실행한다.

```bash
python3 reproduce_final_submission.py --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```

현재 기본 모드는 `submissions/final/final_submission.csv`를 읽어 sample submission 구조를 검증한 뒤 `submission.csv`로 복사한다.

성공 조건:

```text
artifact_byte_equal: True
```

현재 검증된 제출 형식:

| 항목 | 값 |
| --- | --- |
| shape | `(250, 10)` |
| columns | sample submission과 동일 |
| row order | sample submission과 동일 |
| probability min | `0.0312928347977511` |
| probability max | `0.999999` |

---

## 2. 최종 submission artifact

정리 후 `submissions/final/`에는 최종 제출 CSV 하나만 남긴다.

| 파일 | 역할 |
| --- | --- |
| `submissions/final/final_submission.csv` | 최종 선택 제출 파일. `submission.csv` 재생성 기준 파일 |

기존 탐색 과정에서 사용한 base/stage2 후보 CSV는 중복 산출물로 보고 보관 대상에서 제외했다. 대신 최종 조정 방식과 수치는 `reports/final_reproduction_report.json`과 이 문서에 남긴다.

과거 S3 조정 공식:

```text
S3_raw = clip(base_S3 + 0.70 * (stage2_S3 - base_S3))
S3_final = sigmoid(logit(S3_raw) + intercept)
```

`intercept`는 `mean(S3_final) == mean(base_S3)`가 되도록 이진 탐색으로 찾았다.

최종 수치:

| 항목 | 값 |
| --- | ---: |
| base S3 mean | `0.6729949309496949` |
| stage2 S3 mean | `0.6697581513742239` |
| raw toward-stage2 mean | `0.6707291852468651` |
| locked S3 mean | `0.6729949309496948` |
| logit intercept shift | `0.01273903376585006` |

base/stage2 파일을 별도로 복원한 경우에는 다음처럼 과거 조정을 다시 실행할 수 있다.

```bash
python3 reproduce_final_submission.py --mode mean-lock --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```

---

## 3. 실행 환경

`requirements.txt`:

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

`reproduce_final_submission.py`와 `solution.py` 모두 기본 데이터 경로는 `data`이며, `--data-dir` 또는 `DATA_DIR` 환경변수로 다른 경로를 받을 수 있다.

---

## 4. `solution.py` 모델 구조

`solution.py`는 train/test를 합쳐 `subject_id`, `lifelog_date` 기준 feature table을 만든 뒤, 7개 target을 각각 binary classifier로 학습한다.

전체 흐름:

```text
load train/sample
-> add calendar/subject sequence features
-> aggregate sensor parquet files by lifelog_date
-> add sleep block / sleep episode features
-> build design matrix with subject one-hot
-> target별 OOF CV
-> final model fit on full train
-> submission variants 저장
```

주요 feature group:

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
| sleep block | lifelog_date 18:00부터 sleep_date 12:00까지의 sleep episode proxy |

수면 episode 해석:

```text
lifelog_date 18:00 -> sleep_date 12:00
```

이 구간을 5분 bin으로 나누고 screen-off, no-step, still activity, darkness, low heart rate, no app usage 등을 조합해 sleep block proxy를 만든다.

---

## 5. 모델 family와 validation

Target별 base model:

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

## 6. 최종 결과 해석상 주의

최종 `submission.csv`는 재생성 가능하지만, private 결과상 일반화가 좋았던 모델은 아니다.

정확한 해석:

```text
public leaderboard feedback으로 선택된 최종 제출 아티팩트.
재현은 가능하지만, private 결과상 public-overfit이 확인된 파일.
```

Private 결과:

| 항목 | 값 |
| --- | ---: |
| public 44% | `0.5726881984` |
| private 100% | `0.61533` |
| implied hidden 56% | `≈ 0.64883` |

원인 분석은 `private_score_postmortem.md`를 참조한다.
