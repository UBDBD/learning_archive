# DACON / 2026 성균관대학교 멀티모달 AI Bias 챌린지
https://dacon.io/competitions/official/236722/overview/description

이 저장소는 `2026 성균관대학교 멀티모달 AI Bias 챌린지` 최종 제출 패키지를 정리한 작업물이다. 이미지와 텍스트가 함께 주어지는 질문에 대해 3개 선택지 중 가장 적절한 라벨을 예측하며, 판단 근거가 부족한 경우 불확실성 선택지를 고르는 것을 목표로 한다.

현재 폴더는 대회 종료 후 다음 목적에 맞게 정리되어 있다.

- 최종 `final_submission.csv` 재현 및 검증
- 모델/프롬프트/근거 게이트 파이프라인 설명
- 최종 공개/비공개 리더보드 결과와 일반화 실패 원인 분석

> GitHub 공개본에서는 대회 제공 원본 데이터(`data/`), 모델 가중치(`models/`), 최종 제출 CSV(`final_submission.csv`), 원시 추론 로그(`*.jsonl`), 내부 산출물(`final_runs/`)을 포함하지 않는다. 원본 데이터와 모델 가중치는 대회 및 모델 배포처의 조건에 맞춰 별도로 준비해야 한다.

---

## 1. 최종 결과 요약

| 항목 | 값 |
| --- | ---: |
| Public score | `0.99558` |
| Public rank | `65` |
| Private score | `0.72261` |
| Private rank | `83` |

최종 제출본은 public score에서는 매우 높았지만 private score에서 크게 하락했다. 따라서 이 제출물은 “private 일반화가 충분히 검증된 모델”이라기보다, **public leaderboard에서 강했던 재현 가능한 최종 제출 아티팩트**로 해석해야 한다.

공개/비공개 점수 차이:

```text
0.99558 - 0.72261 = 0.27297
```

핵심 해석은 `docs/RESULT_ANALYSIS.md`에 정리되어 있다. 가장 큰 교훈은 모델 크기보다 검증 전략이며, public score만으로 최종 후보를 고르면 private 분포의 모호성, 이미지-텍스트 불일치, 사회적 편향 함정을 충분히 방어하기 어렵다는 점이다.

---

## 2. 대회 목표와 평가 방식

대회 목표는 이미지와 텍스트로 구성된 질의응답 데이터를 이해하고, 주어진 질문에 대해 가장 적절한 선택지 번호를 예측하는 것이다.

핵심 요구사항:

- 명확한 근거가 있으면 해당 인물/선택지를 고른다.
- 판단 근거가 부족하면 불확실성 선택지를 고른다.
- 성별, 인종, 나이, 외모, 복장, 사회적 배경만으로 능력, 성격, 책임, 위험성 등을 추론하지 않는다.
- 평가 데이터셋 패턴을 분석해 규칙, 의사 라벨, 테스트 전용 프롬프트를 만들지 않는다.

평가 지표는 균형 정확도(Balanced Accuracy)다.

```text
Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2
```

Leaderboard 구조:

| 구간 | 설명 |
| --- | --- |
| Public score | 전체 테스트 데이터 중 사전 샘플링된 약 60% |
| Private score | 나머지 약 40% |

---

## 3. 데이터 구조

공식 데이터는 다음 구조를 갖는다.

```text
open.zip
├── train/
│   ├── images/
│   │   └── train_img_0000.jpg
│   └── train.csv
├── test/
│   ├── images/
│   │   ├── test_img_0000.jpg
│   │   ├── ...
│   │   └── test_img_8499.jpg
│   └── test.csv
└── sample_submission.csv
```

주요 파일:

| 파일 | 역할 |
| --- | --- |
| `data/train/train.csv` | 형식 예시용 train 샘플 1개. `label` 포함 |
| `data/test/test.csv` | 평가용 test 샘플 8,500개 |
| `data/sample_submission.csv` | 제출 템플릿. 행 순서 유지 필요 |
| `data/test/images/` | test 이미지 8,500개 |

주요 컬럼:

| 컬럼 | 설명 |
| --- | --- |
| `sample_id` | 샘플 식별자 |
| `image_path` | 이미지 상대 경로 |
| `context` | 질문 판단에 필요한 텍스트 맥락 |
| `question` | 질문 |
| `answers` | 3개 선택지 JSON 문자열 |
| `label` | 정답 라벨. train에만 존재 |

제출 파일은 `sample_id,label` 두 열을 가져야 하며, `label`은 `0`, `1`, `2` 중 하나다.

---

## 4. 최종 제출 파일

현재 로컬 작업 폴더의 최종 제출 관련 파일은 다음과 같다.

| 경로 | 역할 |
| --- | --- |
| `final_submission.csv` | 최종 업로드용 제출 파일 |
| `final_runs/qwen35_evidencegate/raw_outputs_qwen35_highpix401k_evidencegate.jsonl` | 최종 원시 모델 출력 |

제출 CSV는 최종본인 `final_submission.csv` 하나만 유지한다. 중복 사본과 중간 후보 제출물은 정리했다.


---

## 5. 빠른 재현 및 검증

기존 최종 산출물을 다시 추론하지 않고 검증한다.

```bash
.venv/bin/python run_evidence_gate_pipeline.py --resume
```

실행될 전체 명령을 확인한다.

```bash
.venv/bin/python run_evidence_gate_pipeline.py --dry_run
```

데이터 형식만 검증한다.

```bash
.venv/bin/python infer.py \
  --data_dir data \
  --split test \
  --expected_test_rows 8500 \
  --validate_only
```

최종 제출본을 감사한다.

```bash
.venv/bin/python tools/audit_final_candidate.py \
  --submission final_submission.csv \
  --raw_output final_runs/qwen35_evidencegate/raw_outputs_qwen35_highpix401k_evidencegate.jsonl \
  --report /tmp/final_submission_audit.json
```

필요 시 전체 추론을 다시 생성하고 최종 위치에 반영한다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_evidence_gate_pipeline.py --overwrite --cuda 0
```

검증된 제출 형식:

```text
rows: 8500
columns: sample_id,label
sample order: data/sample_submission.csv와 동일
labels: 0, 1, 2
label distribution: {0: 2961, 1: 2732, 2: 2807}
repair count: 18/8500
```

---

## 6. 실행 환경

기준 평가 환경:

```text
GPU: RTX A6000 48GB
OS: Ubuntu 20.04
Python: 3.10
CUDA: 12.4
PyTorch: 2.6.0
```

로컬 리허설 환경:

```text
Python: 3.10.12
PyTorch: 2.6.0+cu124
CUDA runtime: 12.4
transformers: 5.9.0
accelerate: 1.13.0
```

권장 환경 구성:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

모델은 다음 경로에 배치한다고 가정한다.

```text
models/Qwen3.5-9B
```

---

## 7. 프로젝트 구조

```text
.
├── README.md
├── requirements.txt
├── infer.py
├── run_evidence_gate_pipeline.py
├── tools/
│   ├── audit_final_candidate.py
│   └── adjudicate_resolution_disagreements.py
├── docs/
│   ├── competition_overview.md
│   ├── model_and_reproducibility.md
│   └── RESULT_ANALYSIS.md
├── data/
├── final_runs/
└── final_submission.csv
```

공개 저장소에 올릴 때는 `data/`, `final_runs/`, `final_submission.csv`를 제외한다.

주요 파일 역할:

| 경로 | 역할 |
| --- | --- |
| `infer.py` | 로컬 VLM 추론, 출력 파싱, 복구 프롬프트, 제출 생성 |
| `run_evidence_gate_pipeline.py` | 최종 3단계 파이프라인 실행 및 최종본 검증 |
| `tools/adjudicate_resolution_disagreements.py` | 기본 해상도/고해상도 불일치 샘플 재판단 |
| `tools/audit_final_candidate.py` | 최종 제출 파일 및 원시 출력 감사 |
| `docs/competition_overview.md` | 대회 목표, 데이터 구조, 평가 방식, 제출 규칙 |
| `docs/model_and_reproducibility.md` | 최종 모델, 실행 환경, 재현 및 검증 명령 |
| `docs/RESULT_ANALYSIS.md` | 최종 리더보드 결과와 private 사후 분석 |

---

## 8. 최종 모델 파이프라인

최종 솔루션은 로컬 `Qwen/Qwen3.5-9B` 가중치를 직접 로드해 세 단계로 추론한다.

```text
test CSV 로드
-> image/context/question/answers 검증
-> 기본 해상도 직접 근거 추론
-> 고해상도 직접 근거 추론
-> 두 결과가 다른 샘플만 evidence-gate 재판단
-> 최종 label 파싱 및 repair
-> sample_submission 순서 검증
-> final_submission.csv 생성
```

단계별 설정:

| 단계 | 내용 |
| --- | --- |
| 1 | `image_max_pixels=200704` 직접 근거 추론 |
| 2 | `image_max_pixels=401408` 고해상도 직접 근거 추론 |
| 3 | 1단계와 2단계가 다른 샘플만 로컬 VLM 판정기로 재판단 |

동일한 샘플은 기본 해상도 결과를 유지한다. 불일치 샘플은 같은 로컬 VLM이 이미지, 맥락, 질문, 선택지, 두 이전 답변을 다시 보고 최종 `{"label": n}` 텍스트를 생성한다.

이 방식은 다음에 해당하지 않는다.

- 원격 추론 API
- 다수결
- 평균 앙상블
- 하드코딩된 라벨 매핑
- 공개 점수 기반 라벨 수정
- 테스트셋 패턴 기반 규칙

---

## 9. 프롬프트와 파싱 원칙

프롬프트는 일반적인 근거 판단 규칙만 사용한다.

- 맥락에 명시된 행동, 발언, 역할, 사실은 근거로 사용한다.
- 이미지에 명확히 보이는 객체, 행동, 공간 관계, 읽을 수 있는 글자는 근거로 사용한다.
- 성별, 인종, 나이, 외모, 복장, 사회적 배경만으로 능력, 성격, 책임, 위험성 등을 추론하지 않는다.
- 근거가 부족하면 불확실성 선택지를 고른다.
- 테스트셋 패턴, 답변 분포, 샘플 전용 규칙은 사용하지 않는다.

모델은 JSON 형태의 라벨을 생성하도록 지시받는다.

```json
{"label": 0}
```

파싱이 실패하면 복구 프롬프트를 한 번 사용한다. 최종 evidence-gate 실행의 복구 횟수는 `18/8500`이다.

---

## 10. 모델 및 라이선스

| 항목 | 값 |
| --- | --- |
| 모델 | `Qwen/Qwen3.5-9B` |
| 로컬 경로 | `models/Qwen3.5-9B` |
| 라이선스 | Apache-2.0 |
| 모델 URL | https://huggingface.co/Qwen/Qwen3.5-9B |

최종 솔루션은 원격 추론 API를 사용하지 않고, 로컬 모델 가중치를 직접 로드한다.

---

## 11. Private score 사후 분석 요약

최종 제출본은 public subset에서는 매우 강했지만 private subset에서는 충분히 일반화되지 못했다.

주요 원인 가설:

1. Public subset에 대한 간접 과적합
2. Private의 모호성/명확성 판정 기준 차이
3. 지칭 불명확성, 이미지-텍스트 불일치, 사회적 단서 유도에 대한 과잉 추론
4. 기본 해상도와 고해상도 추론이 같은 방향으로 틀리는 샘플을 evidence-gate가 잡지 못함
5. 불확실성 선택 기준이 private에서 충분히 보수적이지 않았음
6. public score feedback을 활용한 후보 선택 과정이 private 일반화보다 leaderboard 적합도를 더 크게 반영함

정확한 해석:

```text
public leaderboard에서 강했던 최종 제출 아티팩트.
재현은 가능하지만, private 결과상 public-overfit 위험이 확인된 파일.
```

---

## 12. 문서 구조

문서는 README를 중심으로 두되, 세부 내용은 다음처럼 나눠 둔다.

| 문서 | 내용 |
| --- | --- |
| `README.md` | 프로젝트 전체 요약, 최종 결과, 빠른 재현, 공개 주의사항 |
| `docs/competition_overview.md` | 대회 목표, 데이터 구조, 평가 방식, 제출 규칙 |
| `docs/model_and_reproducibility.md` | 최종 모델, 파이프라인, 검증 명령, 재생성 방법 |
| `docs/RESULT_ANALYSIS.md` | public/private 결과 분석과 private 사후 분석 |
