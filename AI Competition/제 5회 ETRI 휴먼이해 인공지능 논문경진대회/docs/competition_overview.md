# Competition and Data Overview

정리 기준일: 2026-06-26  
프로젝트: DACON / ETRI “라이프로그 데이터를 활용한 수면, 감정, 스트레스 인식 및 추론”

이 문서는 최종 제출 패키지에 필요한 대회 개요, 평가 방식, 데이터 구조만 압축해 정리한다.

---

## 1. 대회 목표

스마트폰 및 웨어러블 lifelog sensor data를 사용해 7개 binary target을 예측한다.

| Target | 의미 |
| --- | --- |
| `Q1` | 취침 후 수면의 질 |
| `Q2` | 취침 전 피로도 |
| `Q3` | 취침 전 스트레스 |
| `S1` | 총 수면시간 |
| `S2` | 수면효율 |
| `S3` | 수면 지연시간 |
| `S4` | 수면 중 각성 시간 |

제출 파일은 test 250행에 대해 위 7개 target의 확률을 채운 CSV다. Hard label이 아니라 `0~1` 확률을 제출한다.

---

## 2. 평가 방식

평가 지표는 7개 target별 binary logloss의 평균이다.

```text
score = mean(logloss(Q1), logloss(Q2), logloss(Q3), logloss(S1), logloss(S2), logloss(S3), logloss(S4))
```

대회 leaderboard 구조:

| 구간 | 설명 |
| --- | --- |
| Public Score | 전체 test 중 사전 샘플링된 44% |
| Private Score | 전체 test 100% |

최종 확인 결과:

| 항목 | 값 |
| --- | ---: |
| final public score | `0.5726881984` |
| final private score | `0.61533` |
| final private rank | `324` |

Private 결과상 최종 제출물은 public subset에 과하게 맞춰진 것으로 판단한다. 자세한 원인은 `private_score_postmortem.md`에 정리했다.

---

## 3. 데이터 구조

현재 패키지의 데이터 폴더:

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

CSV files:

| 파일 | 역할 |
| --- | --- |
| `ch2026_metrics_train.csv` | train 450일분 label. `subject_id`, `sleep_date`, `lifelog_date`, 7개 target 포함 |
| `ch2026_submission_sample.csv` | test 250행 제출 template. 행/열 순서를 반드시 유지 |
| `ch2026_metrics_description.pdf` | 7개 metric 설명 |

중요한 날짜 기준:

```text
sleep_date = lifelog_date 다음 날
sensor merge 기준 = lifelog_date
```

수면 관련 feature는 일반적인 하루 단위가 아니라 다음 구간을 하나의 sleep episode로 해석한다.

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

## 5. 코드 및 모델 설명서 제출 규칙 요약

재현성 검증 대상이 되는 코드는 다음 조건을 만족해야 한다.

- `/data` 또는 인자 기반 데이터 경로 지원
- `.py` 코드 UTF-8 인코딩
- 모든 코드가 오류 없이 실행 가능
- OS 및 라이브러리 버전 또는 `requirements.txt` 제공
- Private leaderboard score를 복원할 수 있어야 함
- 모델 설명서 자유 양식 제출

현재 패키지는 다음 방식으로 최종 제출물을 복원한다.

```bash
python3 reproduce_final_submission.py --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```
