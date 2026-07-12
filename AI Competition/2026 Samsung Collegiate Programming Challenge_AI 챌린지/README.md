# 2026 Samsung Collegiate Programming Challenge: AI 챌린지
https://dacon.io/competitions/official/236730/overview/description

최종 결과는 **133등**이다. 이 저장소는 대회 제공 task stream을 스키마에 맞는 `submission.csv`로 변환하는 결정적 Python harness다. 모델 학습, 외부 API 호출, 핵심 실행 경로의 네트워크 접근은 사용하지 않는다.

## 최종 결과
| 항목 | 값 |
| --- | ---: |
| Public score | `0.8776` |
| Public rank | `133/649` |

## 빠른 시작

필요 조건은 Python 3.9 이상과 `data/` 아래의 제공 파일이다. 핵심 생성 및 검증 경로는 Python 표준 라이브러리만 사용한다.

선택적 분류기 분석까지 실행하려면 다음을 설치한다.

```bash
python3 -m pip install -r requirements.txt
```

```bash
make dev
make submission
make verify
```

동일한 명령은 다음과 같다.

```bash
python3 run_dev.py
python3 make_submission.py --output submission.csv
python3 scripts/verify_reproducibility.py
```

`make submission`은 루트의 `submission.csv`를 결정적으로 다시 생성한다. `make verify`는 screening 제출본을 두 번 생성하고, `submission.csv` 및 최종 아카이브와 바이트 단위로 비교한 뒤 로컬 dev 회귀 검증을 다시 수행한다.

예상 검증 출력에는 다음이 포함된다.

```text
submission_sha256=6399e26dc6159e1e94045688c952871249f840470892e1667dc5d0099f23d650
screening_tasks=700
deterministic_generation=ok
working_submission_matches_archive=ok
dev_overall=0.9446
dev_regression=ok
```

## 저장소 구조

| 경로 | 역할 |
| --- | --- |
| `harness.py` | 결정적 focal, target, control, scope, policy, plan 판단 로직 |
| `make_submission.py` | screening 추론, 스키마 검증, 단일 셀 CSV 생성 |
| `run_dev.py` | 로컬 dev 평가와 진단 산출물 생성 |
| `scripts/verify_reproducibility.py` | 최종 후보 바이트 비교 및 dev 회귀 검증 |
| `data/` | 주최측 제공 task, 스키마, 용어 가이드, baseline notebook, dev 답안 |
| `reports/` | 다시 생성 가능한 dev 산출물 |
| `doc/SOLUTION.md` | 상세 구현 및 규칙 설계 설명 |
| `doc/result_analysis.md` | 결과, 반복 실험 분석, 한계 및 교훈 |

## 재현 범위

이 프로젝트는 추론 전용이다. 학습된 체크포인트, 학습 스크립트, 외부 서비스 없이 제공된 `data/screening_tasks.jsonl`과 저장소의 결정적 코드만으로 최종 CSV를 재현할 수 있다.

`data/dev_answers.json`은 `run_dev.py`와 회귀 검증에서만 사용한다. 제출 생성기는 screening task만 로드하며 dev 답안을 읽지 않는다. 선택적 사후 분석인 `analyze_control_model.py`는 `requirements.txt`의 `scikit-learn`을 사용한다.

## 최종 산출물

루트의 `submission.csv`

```text
6399e26dc6159e1e94045688c952871249f840470892e1667dc5d0099f23d650
```

## 준수 사항

- harness는 로컬 결정적 facade만 사용하며 `uses_external_api`는 항상 `false`다.
- `harness.py`에는 task ID, session ID, 정답을 하드코딩하지 않는다.
- 규칙은 공개 스키마, 용어 가이드, dev 예시를 바탕으로 만들었으며 전체 task stream에 일반적으로 적용한다.
- 주최측 제공 screening task는 추론 입력으로만 사용하며, 해당 정답 라벨은 이 저장소에 포함되어 있지 않다.
