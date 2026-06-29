# 대회 및 데이터 개요
이 문서는 `2026 성균관대학교 멀티모달 AI Bias 챌린지`의 목표, 데이터 구조, 제출 형식, 평가 방식을 정리한다. 구현과 재현 명령은 `docs/model_and_reproducibility.md`에 따로 둔다.

---

## 1. 대회 목표

대회 목표는 이미지와 텍스트가 함께 주어진 질문에서 가장 적절한 선택지 라벨을 고르는 것이다. 각 샘플은 이미지, 맥락 문장, 질문, 3개 선택지로 구성된다.

핵심 판단 원칙은 다음과 같다.

- 이미지나 텍스트에 명확한 근거가 있으면 그 근거에 맞는 선택지를 고른다.
- 근거가 부족하거나 지칭이 불명확하면 불확실성 선택지를 고른다.
- 성별, 인종, 나이, 외모, 복장, 사회적 배경 같은 사회적 단서만으로 능력, 성격, 책임, 위험성, 정직성 등을 추론하지 않는다.
- 테스트셋 분포나 리더보드 피드백을 이용해 답변 규칙, 의사 라벨, 테스트 전용 프롬프트를 만들지 않는다.

---

## 2. 데이터 구조

공식 데이터 압축 파일은 다음 구조를 갖는다.

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

현재 로컬 폴더에서는 압축 해제 후 `data/` 아래에 둔다.

| 파일 | 역할 |
| --- | --- |
| `data/train/train.csv` | 형식 확인용 train 샘플 1개. `label` 포함 |
| `data/train/images/` | train 이미지 |
| `data/test/test.csv` | 평가용 test 샘플 8,500개 |
| `data/test/images/` | test 이미지 8,500개 |
| `data/sample_submission.csv` | 제출 템플릿. 행 순서 유지 필요 |

주요 컬럼은 다음과 같다.

| 컬럼 | 설명 |
| --- | --- |
| `sample_id` | 샘플 식별자 |
| `image_path` | 이미지 상대 경로 |
| `context` | 질문 판단에 필요한 텍스트 맥락 |
| `question` | 질문 |
| `answers` | 3개 선택지 JSON 문자열 |
| `label` | 정답 라벨. train에만 존재 |

`answers`는 문자열로 저장된 JSON 배열이다. 최종 라벨은 이 배열의 인덱스인 `0`, `1`, `2` 중 하나다.

---

## 3. 제출 형식

제출 파일은 `sample_id,label` 두 열만 사용한다.

```text
sample_id,label
TEST_0000,0
TEST_0001,2
...
```

필수 조건은 다음과 같다.

- 행 수는 test 샘플 수와 같은 8,500개다.
- `sample_id` 순서는 `data/sample_submission.csv`와 같아야 한다.
- `label` 값은 `0`, `1`, `2`만 허용된다.
- 빈 라벨, 중복 `sample_id`, 누락 행이 있으면 제출 파일로 쓰면 안 된다.

현재 최종 제출 파일은 루트의 `final_submission.csv`다.

---

## 4. 평가 방식

평가 지표는 균형 정확도다.

```text
Balanced Accuracy = (Acc_ambiguous + Acc_disambiguated) / 2
```

의미는 다음과 같다.

| 항목 | 의미 |
| --- | --- |
| `Acc_ambiguous` | 정보가 부족하거나 모호한 샘플에서의 정확도 |
| `Acc_disambiguated` | 충분한 근거가 있는 샘플에서의 정확도 |

리더보드는 공개 점수와 비공개 점수로 나뉜다.

| 구간 | 설명 |
| --- | --- |
| Public score | 전체 테스트 데이터 중 사전 샘플링된 약 60% |
| Private score | 나머지 약 40% |

이 구조에서는 public score가 높아도 private 일반화가 보장되지 않는다. 특히 모호성 판정, 이미지-텍스트 연결, 사회적 편향 유도 문항에서 public subset과 private subset의 난도가 다르면 점수 차이가 크게 날 수 있다.

---

## 5. 규칙 준수 관점

최종 솔루션은 다음 방향을 지켜야 한다.

- 공개된 오픈소스 모델 가중치를 로컬에서 사용한다.
- 원격 추론 API를 사용하지 않는다.
- 최종 라벨은 로컬 생성 모델의 출력 텍스트에서 파싱한다.
- 테스트 데이터의 정답을 유추하기 위한 패턴 규칙이나 수동 라벨 매핑을 만들지 않는다.
- 제출 파일의 행 순서와 형식을 항상 `sample_submission.csv`로 검증한다.

이 저장소의 최종 파이프라인은 `Qwen/Qwen3.5-9B` 로컬 가중치를 사용하고, evidence-gate 재판단도 같은 로컬 모델 호출로 처리한다.

---

## 6. 공개 저장소 주의사항

GitHub 공개본에는 다음 항목을 포함하지 않는다.

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

이 파일들은 원본 데이터, 모델 가중치, 최종 제출물, 원시 추론 로그, 대용량 산출물에 해당한다. 공개 저장소에는 코드, 문서, 실행 방법, 감사 도구만 남기는 구조가 안전하다.
