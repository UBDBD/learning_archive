# 결과 분석

## 결과

- 최종 보고 순위: **133등**
- 최종 점수: 로컬 workspace에 기록되지 않음
- 마지막으로 기록된 공개 점수: `0.8459`, 299등
- 최종 후보의 로컬 dev 점수: `0.9446`

최종 순위를 로컬 dev 점수의 직접적인 변환값으로 해석해서는 안 된다. dev scorer는 공개 근사치이고, 리더보드는 서버 보관 정답 및 다른 task 조합을 사용한다.

## 점수 이력

| 확인된 점수 | 상황 |
| ---: | --- |
| `0.3626` | 초기 결정적 baseline 반복 |
| `0.5990` | 첫 routing 및 구조 개선 |
| `0.7029` | focal/target/control 교정 지속 |
| `0.7535` | 좁은 실험 전의 안정 공개 baseline |
| `0.7475` | 넓은 변경 실험에 따른 회귀 |
| `0.7537` | 더 좁은 후보로 회복 |
| `0.7546` | 집중된 policy 변경만으로 얻은 최고 결과 |
| `0.8459` | focal parser와 명시적 redaction/route 보정 후 큰 상승 |
| 133등 | 마지막 제출 이후 확인된 최종 순위 |

## 잘 작동한 점

### 1. Focal 해결을 최우선 문제로 둔 것

가장 큰 공개 점수 상승은 non-marker history binding을 수정하고, 표면적인 object 유사도보다 `focal_marker_refs`와 `focal_resolution_trace`를 우선한 뒤 발생했다. 이는 gated dependency를 복구했다. `focal_id`가 맞으면 target, control, scope, policy, plan이 모두 점수를 얻을 수 있다.

### 2. 최신 명시 지시를 존중한 것

가장 강한 control 규칙은 일반 키워드 매칭이 아니었다. 최신 prompt/history 문맥에서 상태를 바꾸는 지시, 즉 local-only update, clarification 필요, 무효화된 precondition, 명시적 redaction 허용을 인식했다. 이들은 의미가 다른 상태 전이이므로 각각 다른 우선순위 위치가 필요했다.

### 3. Policy flag를 희소하게 유지한 것

policy 점수는 집합 유사도를 사용한다. task record가 뒷받침하는 좁은 추가는 효과가 있었지만, 그럴듯하나 근거 없는 flag를 많이 넣는 것은 효과가 없었다. plan event도 같은 원칙이 적용되어, 복잡한 계획보다 작은 표준 템플릿이 더 안정적이었다.

### 4. 강한 공개 후보를 보존한 것

다음 실험 전에 매 공개 점수 스냅샷을 보존했다. 덕분에 실제 개선과 로컬 전용 상승을 구분할 수 있었고, 나중에 다시 만들 필요 없이 rollback 후보를 유지할 수 있었다.

## 그대로 전이되지 않은 점

### 1. 로컬 dev 상승만으로 제출을 판단할 수 없었다

이전 실험 중에는 로컬 policy 또는 plan metric이 의미 있게 올랐지만 공개 점수는 거의 오르지 않거나 내려간 경우가 있었다. dev split은 규칙 검증에 필수였지만 screening의 모든 route 조합을 대표하지는 못했다. 따라서 공개 제출은 dev 점수의 단순 확인이 아니라 별도의 인과 근거로 다뤄야 했다.

### 2. 넓은 ask/policy/plan 재작성은 취약했다

수백 개의 policy flag 또는 clarification plan을 한 번에 바꾸면 원인 분리가 어려웠고, 한 반복에서는 공개 점수가 하락했다. 이후 변경은 record 근거가 있는 좁은 사례로 제한했다.

### 3. 최종 점수를 기록하지 못했다

workspace에는 최종 순위는 남아 있지만 마지막 제출의 수치 점수나 정확한 서버 응답은 남아 있지 않다. 따라서 마지막 후보의 한계 효과를 정확히 계산할 수 없다. 이 사후 분석은 확인된 공개 점수와 최종 순위를 구분하며, 없는 수치를 추정하지 않는다.

## 마지막 후보 평가

마지막 후보는 `0.8459` 공개 스냅샷의 focal과 target을 유지한 채, 미완료 authority, guardrail, `doctor_note_forbidden`, temporary override, calendar conflict, ambiguous-target 확인 불리언에 대한 일반 규칙을 적용했다. 로컬 dev overall은 `0.9409`에서 `0.9446`으로 올랐고, dev의 focal, target, control은 모두 `1.0`을 유지했다.

이는 합리적인 마지막 위험 선택이었다. 낮은 영향의 문구 변경 대신 남은 구조적 이득을 노렸다. 133등이라는 최종 순위는 접근법의 경쟁력을 보여 주지만, 최종 점수가 없으므로 마지막 38개 control 변경의 정확한 기여를 주장할 수는 없다.

## 재사용 가능한 교훈

1. policy를 결정하기 전에 referent를 해결해야 한다. 이후 필드는 그 결정에 조건부로 연결된다.
2. 시간적 우선순위를 명시적으로 구현해야 한다. 최신 invalidation, local-update, confirmation 신호는 오래된 문맥을 덮어야 한다.
3. 집합 점수형 metadata에서는 recall만큼 precision도 중요하다.
4. 가능하면 실험을 한 축으로 제한하고, 제출 산출물마다 hash를 남겨야 한다.
5. 매 제출 직후 서버 점수, 순위, 제출 시각, 후보 hash를 함께 기록해야 한다.