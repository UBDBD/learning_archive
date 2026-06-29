# 모델 및 재현성 가이드

이 문서는 최종 제출 파일 `final_submission.csv`를 어떤 파이프라인으로 만들었고, 현재 로컬 산출물을 어떻게 검증하는지 정리한다. 대회 목표와 데이터 형식은 `docs/competition_overview.md`에 둔다.

---

## 1. 최종 산출물

현재 최종 제출 관련 로컬 산출물은 두 개다.

| 경로 | 역할 |
| --- | --- |
| `final_submission.csv` | 최종 업로드용 제출 파일 |
| `final_runs/qwen35_evidencegate/raw_outputs_qwen35_highpix401k_evidencegate.jsonl` | 최종 evidence-gate 원시 출력 |


제출 CSV는 최종본인 `final_submission.csv` 하나만 유지한다. 중간 제출 후보와 중복 사본은 정리했다.

---

## 2. 모델

| 항목 | 값 |
| --- | --- |
| 모델 | `Qwen/Qwen3.5-9B` |
| 로컬 경로 | `models/Qwen3.5-9B` |
| 라이선스 | Apache-2.0 |
| 모델 URL | https://huggingface.co/Qwen/Qwen3.5-9B |

최종 솔루션은 원격 추론 API를 사용하지 않고, 로컬 Hugging Face 형식 모델 디렉터리에서 가중치를 직접 읽는다.

모델 디렉터리는 공개 저장소에 포함하지 않는다. 재현 환경에서는 다음 경로가 존재해야 전체 재추론이 가능하다.

```text
models/Qwen3.5-9B/config.json
models/Qwen3.5-9B/tokenizer.json
models/Qwen3.5-9B/preprocessor_config.json
models/Qwen3.5-9B/model.safetensors.index.json
models/Qwen3.5-9B/*.safetensors
```

---

## 3. 실행 환경

기준 평가 환경은 다음과 같다.

```text
GPU: RTX A6000 48GB
OS: Ubuntu 20.04
Python: 3.10
CUDA: 12.4
PyTorch: 2.6.0
```

권장 설치 명령:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

네트워크 없이 실행할 수 있도록 모델은 미리 `models/Qwen3.5-9B`에 내려받아 둔다.

---

## 4. 최종 파이프라인

최종 제출은 3단계로 만든다.

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
| 3 | 1단계와 2단계 라벨이 다른 샘플만 evidence-gate로 재판단 |

동일한 샘플은 기본 해상도 결과를 유지한다. 불일치 샘플은 이미지, 맥락, 질문, 선택지, 두 이전 답변을 다시 넣어 로컬 VLM이 최종 라벨을 생성한다.

이 구조는 다수결이나 평균 앙상블이 아니다. 불일치 샘플에 한해 같은 로컬 모델이 근거 기준으로 다시 판단하는 라우팅 방식이다.

---

## 5. 프롬프트와 라벨 파싱

최종 실행은 `infer.py`의 `direct_evidence` 프롬프트와 `tools/adjudicate_resolution_disagreements.py`의 `evidence_gate` 판정 프롬프트를 사용한다.

주요 원칙은 다음과 같다.

- 맥락에 명시된 행동, 발언, 역할, 사실은 근거로 사용한다.
- 이미지에 명확히 보이는 객체, 행동, 공간 관계, 읽을 수 있는 글자는 근거로 사용한다.
- 성별, 인종, 나이, 외모, 복장, 사회적 배경만으로 능력, 성격, 책임, 위험성 등을 추론하지 않는다.
- 근거가 부족하면 불확실성 선택지를 고른다.
- 테스트셋 패턴, 답변 분포, 샘플 전용 규칙은 사용하지 않는다.

모델 출력은 다음 JSON 형식을 기대한다.

```json
{"label": 0}
```

파싱이 실패하면 복구 프롬프트를 한 번 더 호출한다. 최종 evidence-gate 실행의 복구 횟수는 `18/8500`이다.

---

## 6. 빠른 검증

기존 최종 산출물을 다시 추론하지 않고 검증한다.

```bash
.venv/bin/python run_evidence_gate_pipeline.py --resume
```

기대 출력의 핵심 값:

```text
Rows: 8500
Label distribution: {'0': 2961, '1': 2732, '2': 2807}
Raw rows: 8500
Repair count: 18
```

실행될 전체 명령만 확인한다.

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

최종 제출본과 원시 출력을 감사한다.

```bash
.venv/bin/python tools/audit_final_candidate.py \
  --submission final_submission.csv \
  --raw_output final_runs/qwen35_evidencegate/raw_outputs_qwen35_highpix401k_evidencegate.jsonl \
  --report /tmp/final_submission_audit.json
```

공개 패키지에서는 모델 파일이 제외되어도 감사가 실패하지 않는다. 모델 파일까지 반드시 검사하려면 `--require_model_files`를 추가한다.

---

## 7. 전체 재생성

모델과 데이터가 모두 준비되어 있으면 전체 추론을 다시 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_evidence_gate_pipeline.py --overwrite --cuda 0
```

기본 동작은 최종 결과만 남긴다. 1단계/2단계 중간 제출 CSV와 원시 출력은 최종 검증 후 자동 삭제된다.

---

## 8. 검증 조건

최종 제출 파일은 다음 조건을 통과해야 한다.

| 항목 | 기대값 |
| --- | --- |
| 행 수 | `8500` |
| 열 | `sample_id,label` |
| 행 순서 | `data/sample_submission.csv`와 동일 |
| 허용 라벨 | `0`, `1`, `2` |
| 중복 `sample_id` | 없음 |
| 빈 라벨 | 없음 |

원시 출력 JSONL도 `sample_id` 순서, 행 수, 라벨 범위가 제출 CSV와 일치해야 한다.

---

## 9. 공개 패키지 기준

GitHub 공개본에는 코드와 문서만 남긴다.

공개 후보:

```text
README.md
requirements.txt
infer.py
run_evidence_gate_pipeline.py
tools/audit_final_candidate.py
tools/adjudicate_resolution_disagreements.py
docs/competition_overview.md
docs/model_and_reproducibility.md
docs/RESULT_ANALYSIS.md
.gitignore
```

제외 대상:

```text
data/
models/
final_runs/
final_submission.csv
*.jsonl
*.zip
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
```

이 기준은 `.gitignore`에 반영되어 있다.
