# 솔루션 상세 설명

## 1. 문제 정의

이 대회는 자유 형식 텍스트 생성 문제가 아니라 구조화된 agent 판단 문제다. 각 task에는 prompt, visible history, session state, object, 구조화된 record가 들어 있으며, harness는 다음 여섯 개의 점수 핵심 필드를 포함한 answer object를 생성해야 한다.

1. `focal_id`: 현재 처리할 중심 object
2. `target`: 수신처, 채널, 앱 또는 내부 목적지
3. `control`: `proceed`, `amend`, `hold`, `ask`
4. `content_scope`: 허용·제외 정보 필드
5. `policy`: 위험 신호, 위반, 확인 상태
6. `plan_events`: 실행 계획 이벤트 순서

필드 간 의존 순서가 중요하다. focal object가 틀리면 target과 control의 점수가 무효가 되고, 이어서 scope, policy, plan 점수도 막힌다. 따라서 구현은 policy 판단보다 먼저 focal을 해결한다.

## 2. 전체 처리 흐름

```text
task JSON
  -> 로컬 근거 추출
  -> focal 해결
  -> 문맥 분류
  -> target 추론
  -> control 판단
  -> scope + policy 구성
  -> plan 이벤트 템플릿 적용
  -> 스키마 검증
  -> 단일 셀 UTF-8 submission.csv
```

`harness.py`의 `FinalHarness.answer_task()`가 이 흐름을 담당한다. `make_submission.py`는 task를 session과 turn 순으로 정렬하고, session별 상태를 유지하며, 각 답안을 검증한 뒤 제출 CSV를 쓴다.

## 3. Focal 해결

### Marker trace 우선

task에 `focal_marker_refs`와 `focal_resolution_trace`가 있으면 object 배열 순서 대신 trace를 따른다.

1. `route_binding_order`를 읽는다.
2. `latest_phase_rule`에서 phase를 선택하고, 없으면 `latest_phase`를 사용한다.
3. `phase_to_marker`로 phase에 대응하는 marker를 찾는다.
4. `marker_to_ref`로 marker를 `WM-xxxx` 참조값으로 바꾼다.
5. `attrs.ref_code`가 해당 참조값과 일치하는 object를 고른다.

### History 및 의미 기반 fallback

완전한 marker trace가 없으면 최근 visible history의 binding, record의 직접 object 참조, 명시적 `WM-xxxx` 언급, session 연속성, 마지막으로 결정적 prompt/object 적합도 점수를 차례로 확인한다. history parser는 승인된 참조, 유일한 통과 항목, 이전 메모리 binding 같은 표현을 인식한다. 여러 object가 같은 type과 title을 가질 수 있으므로 이 과정이 필요하다.

## 4. 문맥 분류와 Target 추론

`classify_context()`는 행동을 바꾸는 다음 근거를 추출한다.

- local-update와 status-update 경계
- 명시적 ask, 무효화, redaction 지시
- consent, security, safety, precondition 상태
- route authority, boundary, candidate snapshot 상태
- target/focal 모호성과 민감 필드

target 추론 우선순위는 다음과 같다.

1. local update와 persistent memory write는 `memory_store`로 보낸다.
2. 유효한 최신 target-change record는 오래된 문맥보다 우선한다.
3. `resolved_target`은 화면에 보이는 recipient보다 우선한다.
4. 최신 route가 없을 때에만 persistent memory가 channel 후보를 제공한다.
5. focal attribute와 이전 session target을 결정적 fallback으로 사용한다.

이 분리는 화면에 보이는 recipient가 더 최근의 승인 route record를 덮어쓰지 못하게 한다.

## 5. Control 정책

control 판단에는 명시적인 우선순위가 있다. 뒤늦은 허용 신호가 더 우선인 safety 또는 consent 조건을 우회할 수 없다.

| 우선순위 | 조건군 | 결과 |
| ---: | --- | --- |
| 1 | local-only update | `memory_store`로 `proceed` |
| 2 | consent 철회, security, safety, 무효화된 precondition | `hold` |
| 3 | 외부 목적지에 대한 명시적 redaction 허용 | `amend` |
| 4 | 명시적 확인 지시 또는 미해결 route | `ask` |
| 5 | authority가 미완료인 guardrail ladder | `hold` |
| 6 | calendar, amount, merchant, duration, memory conflict | `ask` |
| 7 | 임시 privacy override | `amend` |
| 8 | `doctor_note_forbidden`인 외부 health record | `hold` |
| 9 | 남은 민감 공유 | `amend`, 그 외는 `proceed` |

이 순서는 공개 dev set에서 검증한 뒤 결정적으로 유지했다. 어떤 규칙도 task ID나 session ID를 분기 조건으로 사용하지 않는다.

## 6. Scope, Policy, Plan 구성

### Content scope

- `hold`는 `none`을 만든다.
- local update는 좁은 allowlist를 가진 `status_only`를 만든다.
- `amend`는 `redacted`를 만들고 external policy와 focal content에 따라 민감 필드를 제외한다.
- `ask`는 보통 summary만 노출하되, policy가 요구하면 `redacted` 또는 `none`을 사용한다.
- 일반적인 안전 dispatch는 사용 가능한 필드에 따라 `raw` 또는 `summary`를 사용한다.

ambiguous target의 확인 불리언은 dev에서 일관되게 나타난 상태표를 따른다. `ask`는 policy와 scope 확인을 모두 요구하고, `amend`도 둘 다 요구하며, `proceed`는 policy 확인만 요구하고, `hold`는 둘 다 요구하지 않는다.

### Policy

위험 flag는 task에서 관찰 가능한 근거가 있을 때만 넣는다. 공개 scorer는 집합 유사도를 사용하므로, 추측성 flag를 늘리면 precision이 낮아진다. harness는 strict sharing, external sharing, sensitive content, local-only 처리, minimal disclosure, 모호성, clarification, 확인된 precondition change에 대해서만 근거가 있을 때 flag를 추가한다.

### Plan 템플릿

plan builder는 문장 생성 대신 작은 event ontology를 사용한다.

| 결과 | Plan |
| --- | --- |
| Local update | `read -> verify -> update` |
| Hold | `read -> guard` |
| Ask | `read -> clarify` |
| Redacted dispatch | `read -> redact -> dispatch` |
| Raw dispatch | `read -> dispatch` |
| Summary dispatch | `read -> summarize -> dispatch` |

고정 템플릿을 사용해 verb, target, argument가 제공된 plan ontology와 일치하도록 했다.

## 7. 결정적 실행과 검증

`make_submission.py`는 CSV를 쓰기 전에 다음을 확인한다.

- screening task ID 전체가 정확히 포함되는지
- `scpc.final.answer.v1` 스키마 표식과 필수 metadata가 맞는지
- control과 content_scope 값이 유효한지
- list와 boolean 필드의 타입이 맞는지
- event shape와 event 수 상한이 맞는지
- UTF-8 단일 열, 단일 데이터 행 CSV로 직렬화되는지

`scripts/verify_reproducibility.py`는 최종 회귀 검증 게이트다. 전체 screening payload를 두 번 생성하고 루트의 working submission과 변경 불가 최종 archive가 바이트 단위로 같은지 확인한 뒤, 로컬 dev 점수가 `0.9446`으로 유지되는지 확인한다.

## 8. 결과

최종 보고 순위는 133등이다. 마지막 공개 점수는 `0.8459`였으며, focal parser와 명시적 redaction/route 처리를 보강한 뒤 크게 상승했다. 점수 이력과 로컬 점수의 한계는 `result_analysis.md`에서 설명한다.
