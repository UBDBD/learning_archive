# Private Score Postmortem

정리 기준일: 2026-06-26  
최종 public score: `0.5726881984`  
최종 private score: `0.61533`  
최종 private leaderboard rank: `324`

이 문서는 최종 제출 후 확인된 private 결과를 기준으로, 현재 솔루션이 왜 public에는 강했지만 private에서는 무너졌는지 정리한 사후 분석이다.

---

## 1. 핵심 결론

최종 결과는 public subset에 과하게 맞춰진 결과로 보는 것이 타당하다.

최종 제출 파일은 public 기준으로는 `0.5726881984`까지 개선되었지만, private 전체 score는 `0.61533`이었다. 이는 단순한 순위 변동이나 약간의 hidden mismatch가 아니라, hidden 56%에서 예측 품질이 상당히 나빠졌다는 의미다.

가장 중요한 원인은 다음이다.

```text
학습 모델의 일반화 성능 개선보다
public feedback을 이용한 target-wise 후처리와 calibration 조정이
반복적으로 누적되었다.
```

즉, 마지막 `S3` 조정 하나가 문제였다기보다, 그 이전까지 누적된 public-probe 기반 보정 체인 전체가 private 일반화 관점에서 위험해진 상태였다.

---

## 2. Public / Private 점수 분해

대회 규칙:

```text
Public Score  : 전체 테스트 데이터 중 44%
Private Score : 전체 테스트 데이터 100%
```

최종 점수:

```text
public  = 0.5726881984
private = 0.61533
```

private는 public 44%와 hidden 56%의 가중 평균으로 볼 수 있다.

```text
private = 0.44 * public + 0.56 * hidden
```

따라서 hidden 56%의 implied score는 다음과 같다.

```text
hidden = (0.61533 - 0.44 * 0.5726881984) / 0.56
       ≈ 0.64883
```

정리하면:

| 구간 | score |
| --- | ---: |
| public 44% | `0.57269` |
| implied hidden 56% | `0.64883` |
| private 100% | `0.61533` |
| hidden - public gap | `+0.07615` |

이 gap은 매우 크다. public에서 좋았던 calibration과 ranking이 hidden에서는 상당히 다르게 작동했다는 뜻이다.

---

## 3. 왜 public에는 좋아졌는가

public score는 대회 기간 동안 여러 차례 제출하면서 직접 관측할 수 있었다. 이 과정에서 우리는 다음 방식으로 후보를 개선했다.

```text
base submission 생성
-> public score 확인
-> 특정 target만 mean/prior/rank 방향으로 조정
-> public score가 좋아지면 그 파일을 새 base로 사용
-> 다른 target에서 같은 방식 반복
```

이 전략은 public leaderboard를 올리는 데 효과적이었다. 실제로 public score는 `0.584`대에서 `0.5726881984`까지 개선되었다.

하지만 이 방식은 public subset 44%의 label 분포와 우연한 target 방향성을 계속 학습하는 것과 비슷하다. 제출 횟수가 늘어날수록 public subset에 맞는 조정이 누적된다.

---

## 4. 최종 base가 이미 public-tuned 상태였다

최종 제출물 자체는 `S3` 하나만 stage2 방향으로 이동한 파일이다.

```text
S3_raw = clip(base_S3 + 0.70 * (stage2_S3 - base_S3))
S3_final = sigmoid(logit(S3_raw) + intercept)
```

다만 이때 사용한 base는 순수 학습 모델 output이 아니었다. 이미 여러 public-probe 개선이 누적된 파일이었다.

대표적인 누적 축:

| 축 | 의미 |
| --- | --- |
| anti-seed | 다른 seed/model 방향의 반대편이 public에서 맞는지 확인 |
| compact S1 | S1만 compact route 쪽으로 이동 |
| Q3 consensus | Q3 target 평균/consensus 조정 |
| S1/S2/S4 prior | sleep target별 prior/mean 보정 |
| Q1 anti-stage2 | Q1에서 stage2 반대 방향 반영 |
| S2 stage2 | S2에서 stage2 방향 반영 |
| Q2 stack-safe | Q2에서 stack-safe 방향 소량 반영 |
| S3 stage2 | 최종 S3 stage2 방향 반영 |

따라서 private 실패 원인을 마지막 후보 하나로만 보면 안 된다. 더 정확한 해석은 다음이다.

```text
최종 제출물은 public-probe로 누적 선택된 base 위에
S3 stage2 rank signal을 추가한 파일이며,
그 base 자체가 이미 public subset에 맞춰져 있었다.
```

---

## 5. Mean-lock과 prior 보정의 위험

최종 계열 후보는 target별 평균을 특정 값에 맞추거나 유지하는 방식이 많았다.

최종 제출물의 target mean:

| target | mean |
| --- | ---: |
| Q1 | `0.5098750579732658` |
| Q2 | `0.5768918549754322` |
| Q3 | `0.6000000000000000` |
| S1 | `0.6822222222222222` |
| S2 | `0.6409999952869762` |
| S3 | `0.6729949309496948` |
| S4 | `0.5599999996100536` |

이 평균이 public subset label rate와 가까우면 public logloss는 좋아진다. 하지만 hidden subset의 실제 label rate가 다르면 private logloss는 나빠진다.

이번 결과는 mean-lock 자체가 항상 나쁘다는 뜻은 아니다. 문제는 mean-lock 기준이 내부 검증이나 전체 test 분포 추정이 아니라, public feedback으로 선택된 값에 점점 가까워졌다는 점이다.

---

## 6. 과신 확률의 영향

최종 제출물의 확률 범위:

```text
min = 0.0312928347977511
max = 0.999999
```

`0.999999`는 logloss 관점에서 매우 공격적인 확률이다. public에서는 이런 확신이 맞으면 점수가 좋아지지만, hidden에서 틀리면 매우 큰 penalty가 발생한다.

private가 `0.61533`까지 악화된 것은 단순히 target mean이 조금 어긋난 문제만은 아닐 가능성이 크다. 일부 target/row에서 과신 확률이 hidden label과 반대로 작동하면서 logloss를 크게 밀어 올렸을 가능성이 있다.

---

## 7. 내부 CV가 막지 못한 이유

내부 CV는 train 450행만 사용한다. 반면 public score는 test public 44%에 대한 직접 피드백이다.

초기에는 내부 CV, GroupKFold, blocked time CV를 참고했다. 하지만 후반으로 갈수록 실제 후보 선택 기준은 다음에 가까워졌다.

```text
public에서 0.0001이라도 좋아지면 채택
public에서 나빠지면 폐기
```

이 선택 방식은 public leaderboard 최적화에는 맞지만, private 일반화 검증으로는 약하다.

특히 test가 250행뿐이고 public은 약 110행 수준이므로, public subset의 우연한 subject/target 구성이 전체 test를 대표하지 못할 수 있다.

---

## 8. 문서 해석 수정

기존 문서에서 다음 표현은 private 결과 이후 수정해서 해석해야 한다.

| 기존 해석 | private 이후 수정 해석 |
| --- | --- |
| final selected | public 기준 최종 선택 파일 |
| live public best | public subset에 가장 잘 맞은 파일 |
| final-safe fallback | public 기준 fallback이며 private 안전성을 보장하지 않음 |
| mean-lock으로 큰 private 악화 위험이 낮음 | 실제 private 결과에서는 충분히 낮지 않았음 |
| public에서 연속 개선된 방향은 신뢰 | 제출 반복이 많으면 public overfit 가능성이 커짐 |

앞으로 이 프로젝트의 최종 파일은 “private 일반화가 검증된 모델”이 아니라, 다음처럼 설명하는 것이 정확하다.

```text
public leaderboard feedback으로 선택된 최종 제출 아티팩트.
재현은 가능하지만, private 결과상 public-overfit이 확인된 파일.
```

---

## 9. 다음 대회에서의 기준

비슷한 작은 test/public-probe 대회에서는 다음 기준을 적용하는 것이 더 안전하다.

### 9.1 제출 후보 선택

public이 좋아졌다는 이유만으로 같은 축을 계속 미세 조정하지 않는다.

권장:

```text
public 개선 + 내부 CV/GroupCV/blocked CV 방향 일치
```

비권장:

```text
public 0.0001 개선만 보고 같은 target weight 계속 증가
```

### 9.2 확률 calibration

최종 제출 전에는 aggressive probability를 완화한 후보를 반드시 비교한다.

예시:

```text
p_final = 0.95 * p + 0.05 * target_prior
p_final = clip(p_final, 0.02, 0.98)
p_final = clip(p_final, 0.03, 0.97)
```

public에서는 조금 손해를 보더라도 private에서 큰 logloss 폭발을 막을 수 있다.

### 9.3 Public probe 예산 관리

남은 제출 횟수를 public 점수 미세 조정에만 쓰지 않는다.

권장 제출 슬롯:

1. public-upside 후보
2. conservative calibration 후보
3. independent model/seed ensemble 후보

이번 프로젝트에서는 1번 유형에 대부분의 제출이 집중되었다.

### 9.4 후보 family 분산

같은 base 위 target-wise 후처리만 반복하면 후보 간 error correlation이 높다. private 대응력을 높이려면 다음처럼 다른 family가 필요하다.

- 순수 CV-selected model
- subject/time prior를 약하게 둔 model
- 확률 clipping이 강한 conservative model
- public-tuned model
- 이들 간 low-weight ensemble

---

## 10. 최종 평가

이번 솔루션의 장점:

- feature engineering과 수면 episode 설계는 충분히 풍부했다.
- public leaderboard를 개선하는 probe 운용은 효과적이었다.
- 최종 파일은 재현 가능하도록 정리되었다.

이번 솔루션의 약점:

- 후반부 선택 기준이 public score에 과도하게 의존했다.
- target-wise mean/prior 보정이 hidden label 분포와 맞지 않았다.
- 확률이 너무 공격적인 row가 있었다.
- private를 방어할 conservative calibration 후보를 최종 선택 후보로 충분히 유지하지 못했다.

최종 결론:

```text
코드와 제출물의 재현성은 확보되었지만,
모델 선택 전략은 private 일반화보다 public leaderboard 최적화에 치우쳤다.
```
