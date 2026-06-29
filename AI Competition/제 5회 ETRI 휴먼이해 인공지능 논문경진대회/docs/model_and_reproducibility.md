# 모델 및 재현성 가이드

최종 제출 파일: `submission.csv`  
최종 보관 파일: `submissions/final/final_submission.csv`  
최종 public score: `0.5726881984`  
최종 private score: `0.61533`  

이 문서는 현재 정리본에서 실제로 보장하는 재현 범위를 설명한다.

---

## 1. 재현 범위

현재 공개 코드의 목적은 최종 제출 CSV를 다시 학습해서 만드는 것이 아니라, 로컬에 보관된 최종 제출 artifact가 대회 제출 형식과 일치하는지 검증하고 `submission.csv`로 재생성하는 것이다.

```bash
python3 reproduce_final_submission.py --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```

성공 조건:

```text
artifact_byte_equal: True
```

검증 항목:

| 항목 | 값 |
| --- | --- |
| shape | `(250, 10)` |
| columns | sample submission과 동일 |
| row order | sample submission과 동일 |
| probability min | `0.0312928347977511` |
| probability max | `0.999999` |

---

## 2. 최종 artifact

| 파일 | 역할 |
| --- | --- |
| `submissions/final/final_submission.csv` | 최종 선택 제출 파일. `submission.csv` 재생성 기준 파일 |
| `data/ch2026_submission_sample.csv` | 제출 파일의 행/열 구조 검증 기준 |
| `reports/final_reproduction_report.json` | 재현 검증 결과 |

위 파일들은 로컬 보관용이며 공개 저장소에는 올리지 않는다.

---

## 3. 결과 생성 방식 기록

최종 제출물은 당시 public-best base에서 `S3`만 stage2 방향으로 이동하고, S3 평균을 base와 같게 mean-lock한 결과였다.

```text
S3_raw = clip(base_S3 + 0.70 * (stage2_S3 - base_S3))
S3_final = sigmoid(logit(S3_raw) + intercept)
```

최종 수치:

| 항목 | 값 |
| --- | ---: |
| base S3 mean | `0.6729949309496949` |
| stage2 S3 mean | `0.6697581513742239` |
| raw toward-stage2 mean | `0.6707291852468651` |
| locked S3 mean | `0.6729949309496948` |
| logit intercept shift | `0.01273903376585006` |

base/stage2 후보 CSV는 중복 산출물로 보고 정리 대상에서 제외했다. 따라서 현재 정리본은 위 조정 과정을 다시 계산하지 않고, 최종 artifact의 형식과 byte-level 재생성을 검증한다.

---

## 4. 실행 환경

`requirements.txt`:

```text
pandas>=2.0
```

권장 환경 구성:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. 공개 범위

공개 저장소에는 결과와 재현성에 직접 필요한 코드와 문서만 남긴다.

- 유지: `reproduce_final_submission.py`, `requirements.txt`, 결과/재현성 문서
- 제외: 대회 제공 데이터, 제출 CSV, 리포트 JSON, public-probe 기록, 탐색용 학습 파이프라인 코드

최종 결과 해석은 `RESULT_ANALYSIS.md`를 참조한다.
