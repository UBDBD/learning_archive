# 프로젝트 구조

Private score 사후 분석 이후 정리한 최종 제출 패키지 구조다.

## 루트

- `README.md`: 한국어 프로젝트 개요와 빠른 재현 방법을 정리한 문서다.
- `solution.py`: 학습/추론 스크립트다. 다시 실행하면 향후 보조 산출물은 `submissions/generated/`에 저장된다.
- `reproduce_final_submission.py`: 최종 제출 파일을 정확히 재현하는 스크립트다.
- `requirements.txt`: Python 의존성 목록이다.
- `submission.csv`: 루트 업로드 파일이며, `submissions/final/final_selected.csv`와 byte 단위로 동일하다.
- `PROJECT_STRUCTURE.md`: 현재 문서다.

## 데이터

- `data/ch2026_metrics_train.csv`: train label 파일이다.
- `data/ch2026_submission_sample.csv`: sample/test 행 순서를 정의하는 파일이다.
- `data/ch2026_metrics_description.pdf`: 원본 데이터 설명서다.
- `data/ch2025_data_items/*.parquet`: `solution.py`가 사용하는 sensor log 파일이다.

## 제출 파일

`submissions/`는 하나의 최종 artifact 폴더로 통합되어 있다.

- `submissions/final/final_selected.csv`: 최종 선택된 public-best 제출 파일이다. public `0.5726881984`, private `0.61533`을 기록했다.
- `submissions/final/base_public_best_before_final.csv`: 최종 재현 스크립트가 사용하는 이전 public-best base다.
- `submissions/final/stage2_source.csv`: 최종 S3 이동에만 사용한 stage2 source prediction이다.
- `submissions/final/README.md`: 위 세 파일의 로컬 설명 문서다.

`solution.py`를 다시 실행하면 final이 아닌 generated variant는 `submissions/generated/`에 저장된다. 정리된 패키지에는 이 폴더를 의도적으로 포함하지 않았다.

## 리포트

- `reports/final_reproduction_report.json`: 최종 재현 검증 리포트다.
- `reports/selected_next_submission.json`: 최종 선택 요약 파일이다.
- `reports/public_probe_queue/final_day_queue_2026-06-26.json`: 보존한 최종일 public-probe 기록이다.

## 문서

- `docs/competition_overview.md`: 대회 목표, 평가 방식, 데이터 구조, 제출 규칙을 정리한 문서다.
- `docs/model_and_reproducibility.md`: 최종 artifact 재현 방법과 모델/코드 구조를 설명한 문서다.
- `docs/private_score_postmortem.md`: private score overfit을 분석한 문서다.


## 최종 제출 파일 재현

```bash
python3 reproduce_final_submission.py --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```
