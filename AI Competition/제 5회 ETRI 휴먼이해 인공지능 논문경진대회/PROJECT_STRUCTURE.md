# 프로젝트 구조

Private score 사후 분석 이후 공개용 코드/문서와 로컬 보관용 데이터/산출물을 분리한 구조다.

## 공개용 파일

GitHub에 올리는 파일은 코드와 문서만 남긴다.

- `.gitignore`: 데이터, 제출물, 리포트, 모델 산출물, 환경 파일을 제외한다.
- `README.md`: 프로젝트 개요, 빠른 재현, 핵심 결과를 담은 진입점이다.
- `PROJECT_STRUCTURE.md`: 현재 파일 구조와 보관 기준을 설명한다.
- `requirements.txt`: Python 의존성 목록이다.
- `solution.py`: 원본 데이터에서 feature/model/submission 후보를 생성하는 전체 학습 코드다.
- `reproduce_final_submission.py`: 최종 제출 CSV를 재생성하고 sample submission 구조를 검증하는 스크립트다.
- `docs/competition_overview.md`: 대회 목표, 평가 방식, 데이터 구조를 정리한다.
- `docs/model_and_reproducibility.md`: 최종 artifact 재현 방식과 모델 구조를 설명한다.
- `docs/private_score_postmortem.md`: private score 악화와 public-overfit 원인을 분석한다.

## 로컬 보관 파일

아래 항목은 대회 제공 데이터, 제출 산출물, 리포트이므로 GitHub 공개 대상에서 제외한다.

- `data/ch2026_metrics_train.csv`: train label 파일이다.
- `data/ch2026_submission_sample.csv`: sample/test 행 순서를 정의하는 파일이다.
- `data/ch2026_metrics_description.pdf`: 원본 데이터 설명서다.
- `data/ch2025_data_items/*.parquet`: `solution.py`가 사용하는 sensor log 파일이다.
- `submissions/final/final_submission.csv`: 최종 선택 제출 파일이다.
- `reports/final_reproduction_report.json`: 최종 제출 파일 재현/검증 리포트다.
- `reports/selected_next_submission.json`: 최종 선택 요약 기록이다.
- `reports/public_probe_queue/final_day_queue_2026-06-26.json`: 최종일 public-probe 기록이다.

## 최종 제출 파일 재현

현재 정리된 패키지는 최종 보관 파일 하나를 기준으로 `submission.csv`를 재생성한다.

```bash
python3 reproduce_final_submission.py --data-dir data --out submission.csv --report-out reports/final_reproduction_report.json
```

성공 시 `artifact_byte_equal: True`가 출력된다.

## 재생성 산출물

`solution.py`를 다시 실행하면 final이 아닌 generated variant는 기본적으로 `submissions/generated/`에 저장된다. 이 폴더는 실험 산출물이므로 정리된 패키지에는 포함하지 않는다.
